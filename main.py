# main.py
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QTreeView, QFileSystemModel, QSplitter, QTextEdit)
from PyQt5.QtCore import Qt, QModelIndex
import pyqtgraph as pg
import pandas as pd

class MeasurementViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("测量波形查看器")
        self.resize(1200, 800)
        self.create_ui()
        # 读取特征文件
        self.features = pd.read_csv('features.tsv', sep='\t', dtype=str)
    
    def create_ui(self):
        # 创建主部件
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)
        
        # 创建文件树视图
        self.model = QFileSystemModel()
        self.model.setRootPath('')
        # 设置只显示 .tsv 文件，但排除 features.tsv
        self.model.setNameFilters(['*.tsv'])
        self.model.setNameFilterDisables(False)
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(os.getcwd()))
        self.tree.clicked.connect(self.on_tree_view_clicked)
        main_splitter.addWidget(self.tree)
        
        # 创建右侧布局
        right_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(right_splitter)
        
        # 创建绘图窗口
        self.plot_widget = pg.GraphicsLayoutWidget()
        right_splitter.addWidget(self.plot_widget)
        
        # 创建特征显示区域
        self.feature_text = QTextEdit()
        self.feature_text.setReadOnly(True)
        right_splitter.addWidget(self.feature_text)
        
        # 设置初始比例
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)
    
    def on_tree_view_clicked(self, index):
        file_path = self.model.filePath(index)
        if os.path.isfile(file_path) and file_path.endswith('.tsv') and os.path.basename(file_path) != 'features.tsv':
            self.load_measurement(file_path)
            self.load_features(file_path)
    
    def load_measurement(self, measurement_file_path):
        # 清空之前的图像
        self.plot_widget.clear()
        
        # 读取测量数据
        try:
            data = pd.read_csv(measurement_file_path, sep='\t')
        except Exception as e:
            self.feature_text.setText(f"无法读取测量数据文件。\n错误信息：{e}")
            return
        
        # 检查是否有数据列可供绘图
        available_columns = data.columns.tolist()
        if not available_columns:
            self.feature_text.setText("测量数据文件中没有可用的数据列。")
            return
        
        time = data['t'] if 't' in available_columns else data.index
        
        colors = {'ekg': 'r', 'optical': 'g', 'pressure': 'b',
                  'accel_x': 'c', 'accel_y': 'm', 'accel_z': 'y'}
        
        i = 0  # 行号
        for col in ['ekg', 'optical', 'pressure', 'accel_x', 'accel_y', 'accel_z']:
            if col in available_columns:
                p = self.plot_widget.addPlot(row=i, col=0, title=col.upper())
                p.plot(time, data[col], pen=colors[col])
                p.setLabel('left', col.upper())
                # 启用鼠标交互
                p.setMouseEnabled(x=True, y=True)
                # 设置 ViewBox 的缩放模式
                p.vb.setMouseMode(pg.ViewBox.RectMode)
                # 添加右键菜单
                p.vb.menu = None  # 禁用默认菜单
                p.vb.scene().contextMenuEvent = self.right_click_menu
                i += 1
        # 同步 X 轴
        self.sync_axes()
    
    def sync_axes(self):
        # 获取所有绘图项
        plots = [item for item in self.plot_widget.items() if isinstance(item, pg.PlotItem)]
        if not plots:
            return
        # 以第一个子图的 X 轴为主轴
        master_plot = plots[0]
        for p in plots[1:]:
            p.setXLink(master_plot)
    
    def right_click_menu(self, event):
        # 右键菜单，添加“重置视图”选项
        menu = pg.QtWidgets.QMenu()
        reset_action = menu.addAction("重置视图")
        action = menu.exec_(event.screenPos())
        if action == reset_action:
            for p in self.plot_widget.items():
                if isinstance(p, pg.PlotItem):
                    p.enableAutoRange()
    
    def load_features(self, measurement_file_path):
        # 解析文件名，提取 pid, phase, measurement
        filename = os.path.basename(measurement_file_path)
        name_parts = os.path.splitext(filename)[0].split('.')
        if len(name_parts) >= 3:
            pid = name_parts[0]
            phase = name_parts[1]
            measurement = name_parts[2].replace('_', ' ')
            # 进行特征匹配
            feature_rows = self.features[
                (self.features['pid'] == pid) &
                (self.features['phase'] == phase) &
                (self.features['measurement'] == measurement)
            ]
            if not feature_rows.empty:
                # 将特征信息显示在文本区域中
                feature_info = feature_rows.to_dict('records')[0]
                feature_text = '\n'.join([f"{k}: {v}" for k, v in feature_info.items()])
                self.feature_text.setText(feature_text)
            else:
                self.feature_text.setText("未找到对应的特征信息。")
        else:
            self.feature_text.setText("无法解析文件名以获取特征匹配信息。")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MeasurementViewer()
    viewer.show()
    sys.exit(app.exec_())