# main.py
import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeView, QFileSystemModel, QSplitter, QTextEdit
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from scipy import signal

class MeasurementViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("量測波形查看器")
        self.resize(1200, 800)

        # 讀取其他相關資訊的 TSV 檔案
        self.features = pd.read_csv('features.tsv', sep='\t', dtype=str)
        self.participants = pd.read_csv('participants.tsv', sep='\t', dtype=str)
        self.measurements_auscultatory = pd.read_csv('measurements_auscultatory.tsv', sep='\t', dtype=str)
        self.measurements_oscillometric = pd.read_csv('measurements_oscillometric.tsv', sep='\t', dtype=str)

        self.create_ui()

    def create_ui(self):
        # 建立主分割視窗：左側檔案樹，右側上部顯示波形，下部顯示文字資訊
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)

        # 左側：檔案樹 (僅顯示 .tsv 檔案)
        self.model = QFileSystemModel()
        self.model.setRootPath('')
        self.model.setNameFilters(['*.tsv'])
        self.model.setNameFilterDisables(False)
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(os.getcwd()))
        self.tree.clicked.connect(self.on_tree_view_clicked)
        main_splitter.addWidget(self.tree)

        # 右側：分上下兩區
        right_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(right_splitter)

        # 上部：波形圖顯示區 (使用 pyqtgraph 的 GraphicsLayoutWidget)
        self.plot_widget = pg.GraphicsLayoutWidget()
        right_splitter.addWidget(self.plot_widget)

        # 下部：資訊文字區
        self.feature_text = QTextEdit()
        self.feature_text.setReadOnly(True)
        right_splitter.addWidget(self.feature_text)

        # 調整比例
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

    def on_tree_view_clicked(self, index):
        file_path = self.model.filePath(index)
        # 只處理 .tsv 檔案，且排除 features.tsv（因為該檔案只存放特徵資料）
        if os.path.isfile(file_path) and file_path.endswith('.tsv') and os.path.basename(file_path) != 'features.tsv':
            self.load_file_content(file_path)

    def load_file_content(self, filepath):
        """
        根據使用者點選的量測檔，繪製分成多個子圖的訊號波形，
        並在下方文字區顯示該量測的相關資訊與特徵
        """
        filename = os.path.splitext(os.path.basename(filepath))[0]
        parts = filename.split('.')
        
        self.plot_widget.clear()
        self.feature_text.clear()

        if len(parts) < 3:
            self.feature_text.setText("檔名格式不符，無法解析 pid, phase, measurement")
            return

        pid = parts[0]
        phase = parts[1]
        measurement = parts[2]

        # 讀取 TSV 資料
        try:
            data = pd.read_csv(filepath, sep='\t', header=None)
            try:
                float(data.iloc[0, 0])
            except ValueError:
                data = pd.read_csv(filepath, sep='\t', header=0)
        except Exception as e:
            self.feature_text.setText(f"讀取量測檔時發生錯誤: {e}")
            return

        if data.shape[1] == 7:
            # 第一欄為時間
            x_data = pd.to_numeric(data.iloc[:, 0], errors='coerce').values
            # 後續依序：EKG, PPG, Tonometry, 加速度X, 加速度Y, 加速度Z
            signal_data = [pd.to_numeric(data.iloc[:, i], errors='coerce').values for i in range(1, 7)]
            x_label = "Time"
        elif data.shape[1] == 6:
            x_data = np.arange(len(data))
            signal_data = [pd.to_numeric(data.iloc[:, i], errors='coerce').values for i in range(6)]
            x_label = "Samples"
        else:
            self.feature_text.setText("量測檔的欄位數不符合預期，無法繪圖。")
            return

        # 在此印出三個主要訊號（EKG, PPG, Tonometry）的長度
        # signal_data[0] => EKG
        # signal_data[1] => PPG
        # signal_data[2] => Tonometry
        print(f"檔案: {filename}")
        print(f"  EKG訊號長度: {len(signal_data[0])}")
        print(f"  PPG訊號長度: {len(signal_data[1])}")
        print(f"  Tonometry訊號長度: {len(signal_data[2])}")

        # 根據時間序列估計取樣頻率 (假設時間單位為秒)
        fs = None
        if len(x_data) > 1:
            dt_vals = np.diff(x_data)
            if np.all(dt_vals > 0):
                dt = np.mean(dt_vals)
                if dt > 0:
                    fs = 1.0 / dt

        channel_titles = ["EKG", "PPG", "Tonometry", "Accelerometer X", "Accelerometer Y", "Accelerometer Z"]

        # 依序繪製各通道波形
        for i, title in enumerate(channel_titles):
            raw_chan = signal_data[i]
            # 如果能正常估計 fs，對 EKG, PPG, Tonometry 做濾波
            if fs is not None:
                if title == "EKG":
                    filtered_chan = self.filter_ekg(raw_chan, fs)
                elif title == "PPG":
                    filtered_chan = self.filter_ppg(raw_chan, fs)
                elif title == "Tonometry":
                    filtered_chan = self.filter_tonometry(raw_chan, fs)
                else:
                    filtered_chan = raw_chan
            else:
                filtered_chan = raw_chan

            p = self.plot_widget.addPlot(row=i, col=0, title=title)
            p.setLabel('bottom', x_label)
            p.setLabel('left', 'Amplitude')
            p.plot(x_data, filtered_chan, pen=pg.mkPen(color=pg.intColor(i, hues=6), width=2))
            p.showGrid(x=True, y=True)

        # 顯示文字資訊
        info_text = self.collect_info(pid, phase, measurement, filename)
        self.feature_text.setText(info_text)

    def filter_ekg(self, x, fs):
        """
        根據論文敘述：
        1. DC block (高通 0.1Hz)
        2. 7階 Elliptic 低通 (pass 40Hz, stop 45Hz, 0.1dB passband ripple)
        3. 6階 Chebyshev type I notch (中心 60Hz, Q=3, 0.1dB passband ripple)
        """
        # 1. DC block (可視為高通截止 0.1Hz)
        sos_dc = signal.iirfilter(
            N=2,    # 2階做示範，可自行調整
            Wn=0.1/(fs/2),
            btype='highpass',
            ftype='butter',
            output='sos'
        )
        y = signal.sosfiltfilt(sos_dc, x)

        # 2. 7階 Elliptic 低通：pass 40Hz，stop 45Hz
        #   這種較複雜的規範可用 iirdesign 或 ellip+對應參數，但這裡先給大略示範
        wp = 40.0 / (fs / 2.0)  # passband edge (正規化)
        ws = 45.0 / (fs / 2.0)  # stopband edge (正規化)
        # rp=0.1 dB passband ripple, rs可自行設定 40~60dB 看需求
        sos_lp = signal.iirdesign(
            wp=wp,
            ws=ws,
            gpass=0.1,
            gstop=40,   # 可調整
            ftype='ellip',
            output='sos'
        )
        y = signal.sosfiltfilt(sos_lp, y)

        # 3. notch 6階 Chebyshev type I 60Hz，Q=3 => 大約帶阻寬度 = 中心頻率/Q
        #   => 帶阻範圍 ~ [60-10, 60+10] = [50, 70], 不過實際可再校正
        w0 = 60.0 / (fs/2.0)
        bw = w0 / 3.0  # 以 Q=3 估計帶寬
        # 建議用 bandstop：w1=[w0 - bw/2, w0 + bw/2]
        w1 = [w0 - bw/2.0, w0 + bw/2.0]
        sos_notch = signal.iirfilter(
            N=6,
            rp=0.1,   # 0.1 dB passband ripple
            Wn=w1,
            btype='bandstop',
            ftype='cheby1',
            output='sos'
        )
        y = signal.sosfiltfilt(sos_notch, y)

        return y

    def filter_ppg(self, x, fs):
        """
        根據論文：
        1. 高通 Butterworth 4階 (0.25Hz cutoff)
        2. 低通 equiripple (10Hz pass, 12Hz stop, 1dB ripple, 60dB stop attenuation)
        """
        # 1. 高通 4階 (0.25Hz)
        sos_hp = signal.butter(
            N=4,
            Wn=0.25/(fs/2.0),
            btype='highpass',
            output='sos'
        )
        y = signal.sosfiltfilt(sos_hp, x)

        # 2. 低通 equiripple (這類可用 firwin2 或 remez 函式實現，這裡示範 iirdesign 不一定完全對應)
        #   pass: 10Hz, stop: 12Hz => 正規化
        wp = 10.0 / (fs / 2.0)
        ws = 12.0 / (fs / 2.0)
        # gpass=1dB, gstop=60dB
        sos_lp = signal.iirdesign(
            wp=wp, 
            ws=ws,
            gpass=1,
            gstop=60,
            ftype='ellip',  # 雖然原文提 equiripple FIR，這裡以 elliptical IIR 示範
            output='sos'
        )
        y = signal.sosfiltfilt(sos_lp, y)

        return y

    def filter_tonometry(self, x, fs):
        """
        根據論文：
        1. 高通 elliptical filter (0.2Hz stop, 0.3Hz pass, 60dB stop attenuation, 1dB passband ripple)
        2. 7階 elliptical 低通 (22Hz pass, 26Hz stop, 0.1dB ripple)
        """
        # 1. 高通 elliptical => 先簡化用 iirdesign，
        #   stop帶 <= 0.2Hz, pass帶 >= 0.3Hz
        ws = 0.2/(fs/2.0)
        wp = 0.3/(fs/2.0)
        sos_hp = signal.iirdesign(
            wp=wp,
            ws=ws,
            gpass=1,    # passband ripple
            gstop=60,   # stopband attenuation
            ftype='ellip',
            output='sos'
        )
        y = signal.sosfiltfilt(sos_hp, x)

        # 2. 7階 elliptical 低通 => pass 22Hz, stop 26Hz
        wp2 = 22.0 / (fs / 2.0)
        ws2 = 26.0 / (fs / 2.0)
        sos_lp = signal.iirdesign(
            wp=wp2,
            ws=ws2,
            gpass=0.1,   # passband ripple
            gstop=40,    # 可根據需求調整
            ftype='ellip',
            output='sos'
        )
        y = signal.sosfiltfilt(sos_lp, y)

        return y

    def collect_info(self, pid, phase, measurement, filename):
        """
        依據 PID, phase, measurement 等資訊，從其他 tsv 檔擷取 metadata，
        並組合成可顯示的文字內容。
        """
        def safe_str(val):
            """若 val 為 None 或 NaN，就回傳 'None'；否則回傳字串"""
            if val is None:
                return "None"
            if pd.isna(val):
                return "None"
            return str(val)

        # 假設 TSV 檔欄位中，把底線都變成空格，
        # 那我們就把來自檔名的 'measurement_20' 變成 'measurement 20' 再做比對
        measurement_for_compare = measurement.replace("_", " ")

        info_text_list = []
        info_text_list.append(f"檔案: {filename}")
        info_text_list.append(f"PID: {pid}")
        info_text_list.append(f"Phase: {phase}")
        info_text_list.append(f"Measurement: {measurement}")
        info_text_list.append("")

        # ========== 受試者資訊 (participants.tsv) ==========
        participant_rows = self.participants[self.participants['pid'] == pid]
        if not participant_rows.empty:
            p_dict = participant_rows.to_dict('records')[0]
            info_text_list.append("=== 受試者資訊 (participants) ===")
            info_text_list.append(f"Age: {safe_str(p_dict.get('age'))}")
            info_text_list.append(f"Gender: {safe_str(p_dict.get('gender'))}")
            info_text_list.append(f"Height: {safe_str(p_dict.get('height'))}")
            info_text_list.append(f"Weight: {safe_str(p_dict.get('weight'))}")
            info_text_list.append("")
        else:
            info_text_list.append("找不到受試者資訊（participants.tsv）")
        
        print(f'pid: {pid}, phase: {phase}, measurement: {measurement}')
        # ========== 量測資訊 (measurements_auscultatory / measurements_oscillometric) ==========
        ausc_rows = self.measurements_auscultatory[
            (self.measurements_auscultatory['pid'] == pid) &
            (self.measurements_auscultatory['phase'] == phase) &
            (self.measurements_auscultatory['measurement'] == measurement_for_compare)
        ]

        osc_rows = self.measurements_oscillometric[
            (self.measurements_oscillometric['pid'] == pid) &
            (self.measurements_oscillometric['phase'] == phase) &
            (self.measurements_oscillometric['measurement'] == measurement_for_compare)
        ]
        print(f'ausc_rows: {self.measurements_auscultatory}')
        print(f'osc_rows: {self.measurements_oscillometric}')
        # 若在測量檔auscultatory或oscillometric找到任何一筆，就顯示第一筆
        if not ausc_rows.empty:
            ausc_dict = ausc_rows.to_dict('records')[0]
            info_text_list.append("=== 量測資訊 (Auscultatory) ===")
            info_text_list.append(f"SBP: {safe_str(ausc_dict.get('sbp'))}")
            info_text_list.append(f"DBP: {safe_str(ausc_dict.get('dbp'))}")
            info_text_list.append(f"Duration: {safe_str(ausc_dict.get('duration'))}")
            info_text_list.append(f"Pressure Quality: {safe_str(ausc_dict.get('pressure_quality'))}")
            info_text_list.append(f"Optical Quality: {safe_str(ausc_dict.get('optical_quality'))}")
            info_text_list.append("")
        elif not osc_rows.empty:
            osc_dict = osc_rows.to_dict('records')[0]
            info_text_list.append("=== 量測資訊 (Oscillometric) ===")
            info_text_list.append(f"SBP: {safe_str(osc_dict.get('sbp'))}")
            info_text_list.append(f"DBP: {safe_str(osc_dict.get('dbp'))}")
            info_text_list.append(f"Duration: {safe_str(osc_dict.get('duration'))}")
            info_text_list.append(f"Pressure Quality: {safe_str(osc_dict.get('pressure_quality'))}")
            info_text_list.append(f"Optical Quality: {safe_str(osc_dict.get('optical_quality'))}")
            info_text_list.append("")
        else:
            info_text_list.append("找不到該次量測的基本資訊（Auscultatory / Oscillometric）")
            info_text_list.append("")

        # ========== 特徵資訊 (features.tsv) ==========
        print(f'self.features[pid]: {self.features["pid"]}')
        print(f'self.features[phase]: {self.features["phase"]}')
        print(f'self.features[measurement]: {self.features["measurement"]}')
        print(f'pid: {pid}, phase: {phase}, measurement: {measurement_for_compare}')
        feature_rows = self.features[
            (self.features['pid'] == pid) &
            (self.features['phase'] == phase) &
            (self.features['measurement'] == measurement_for_compare)
        ]
        if not feature_rows.empty:
            feat_dict = feature_rows.to_dict('records')[0]
            info_text_list.append("=== 特徵資訊 (Features) ===")
            info_text_list.append(f"Baseline SBP: {safe_str(feat_dict.get('baseline_sbp'))}")
            info_text_list.append(f"Baseline DBP: {safe_str(feat_dict.get('baseline_dbp'))}")
            info_text_list.append(f"Delta SBP: {safe_str(feat_dict.get('delta_sbp'))}")
            info_text_list.append(f"Delta DBP: {safe_str(feat_dict.get('delta_dbp'))}")
            info_text_list.append("")
        else:
            info_text_list.append("找不到特徵資訊（features.tsv）")

        return "\n".join(info_text_list)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MeasurementViewer()
    viewer.show()
    sys.exit(app.exec_())
