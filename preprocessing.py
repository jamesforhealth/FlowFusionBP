import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import scipy.signal as signal
from tqdm import tqdm

class SignalDataset(Dataset):
    def __init__(self, data_path, window_size=200, stride=100, max_samples=None):
        """
        Args:
            data_path: 数据文件路径
            window_size: 滑动窗口大小
            stride: 滑动步长
            max_samples: 最大样本数量，用于限制数据集大小
        """
        # 加载数据
        self.data = np.load(data_path, mmap_mode='r')  # 使用内存映射模式加载
        
        # 如果指定了最大样本数，则只使用部分数据
        if max_samples is not None:
            self.data = self.data[:max_samples]
            
        self.window_size = window_size
        self.stride = stride
        
        # 计算每个信号可以产生的窗口数量
        self.samples_per_signal = (self.data.shape[1] - self.window_size) // self.stride + 1
        self.num_signals = self.data.shape[0]
        
        print(f"数据形状: {self.data.shape}")
        print(f"每个信号可产生窗口: {self.samples_per_signal}")
        print(f"总窗口数: {len(self)}")

    def __len__(self):
        return self.num_signals * self.samples_per_signal

    def __getitem__(self, idx):
        signal_idx = idx // self.samples_per_signal
        window_idx = idx % self.samples_per_signal
        start = window_idx * self.stride
        end = start + self.window_size
        
        signal = self.data[signal_idx, start:end].copy()
        return torch.FloatTensor(signal).unsqueeze(0)

def get_data_loaders(optical_path, pressure_path, batch_size=64, window_size=200, 
                    stride=100, num_workers=0, max_samples=10000):
    """
    创建数据加载器
    
    Args:
        optical_path: 光学信号数据路径
        pressure_path: 压力信号数据路径
        batch_size: 批次大小
        window_size: 窗口大小
        stride: 滑动步长
        num_workers: 工作进程数
        max_samples: 最大样本数量
    """
    # 创建数据集
    optical_dataset = SignalDataset(optical_path, window_size, stride, max_samples)
    pressure_dataset = SignalDataset(pressure_path, window_size, stride, max_samples)
    
    # 创建数据加载器
    optical_loader = DataLoader(
        optical_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    pressure_loader = DataLoader(
        pressure_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return optical_loader, pressure_loader

def apply_bandpass_filter(data, fs=500, lowcut=0.5, highcut=15.0, order=4):
    """
    应用带通滤波器
    
    Args:
        data: 输入信号
        fs: 采样频率(Hz)
        lowcut: 低频截止(Hz)
        highcut: 高频截止(Hz)
        order: 滤波器阶数
    
    Returns:
        filtered_data: 滤波后的信号
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def preprocess_paired_signals(optical_data, pressure_data, fs=500, lowcut=0.5, highcut=15.0, order=4):
    """
    对成对的信号进行预处理
    
    Args:
        optical_data: 光学信号数据
        pressure_data: 压力信号数据
        fs: 采样频率
        lowcut: 带通滤波器低频截止
        highcut: 带通滤波器高频截止
        order: 滤波器阶数
    """
    # 确保长度相同
    min_length = min(len(optical_data), len(pressure_data))
    optical_data = optical_data[:min_length]
    pressure_data = pressure_data[:min_length]
    
    # 带通滤波
    optical_filtered = apply_bandpass_filter(optical_data, fs, lowcut, highcut, order)
    pressure_filtered = apply_bandpass_filter(pressure_data, fs, lowcut, highcut, order)
    
    # 分别进行Z-score标准化
    optical_norm = (optical_filtered - np.mean(optical_filtered)) / (np.std(optical_filtered) + 1e-8)
    pressure_norm = (pressure_filtered - np.mean(pressure_filtered)) / (np.std(pressure_filtered) + 1e-8)
    
    return optical_norm, pressure_norm

def sliding_window_paired(optical_data, pressure_data, window_size, step_size):
    """
    对成对信号进行滑动窗口分段
    """
    optical_segments = []
    pressure_segments = []
    
    for start_pos in range(0, len(optical_data) - window_size + 1, step_size):
        optical_segment = optical_data[start_pos:start_pos + window_size]
        pressure_segment = pressure_data[start_pos:start_pos + window_size]
        
        # 检查数据质量
        if (not np.isnan(optical_segment).any() and 
            not np.isnan(pressure_segment).any() and
            not np.isinf(optical_segment).any() and 
            not np.isinf(pressure_segment).any()):
            optical_segments.append(optical_segment)
            pressure_segments.append(pressure_segment)
    
    return np.array(optical_segments), np.array(pressure_segments)

def load_signals(data_folders, window_size=4096, step_size=512, save_folder='preprocessed_data'):
    """
    加载并预处理信号数据，保持成对关系
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    all_optical_segments = []
    all_pressure_segments = []
    
    # 记录处理的文件数和信号段数
    processed_files = 0
    total_segments = 0
    
    for data_folder in data_folders:
        file_paths = glob.glob(os.path.join(data_folder, '**', '*.tsv'), recursive=True)
        
        for file_path in file_paths:
            try:
                # 读取数据
                data = pd.read_csv(file_path, sep='\t')
                
                # 确保两种信号都存在
                if 'optical' not in data.columns or 'pressure' not in data.columns:
                    print(f"跳过文件 {file_path}: 缺少必要的信号列")
                    continue
                
                # 提取信号
                optical_data = data['optical'].values.astype(float)
                pressure_data = data['pressure'].values.astype(float)
                
                # 预处理成对信号
                optical_norm, pressure_norm = preprocess_paired_signals(optical_data, pressure_data)
                
                # 滑动窗口分段
                optical_segments, pressure_segments = sliding_window_paired(
                    optical_norm, pressure_norm, window_size, step_size
                )
                
                # 添加到总列表
                if len(optical_segments) > 0:
                    all_optical_segments.append(optical_segments)
                    all_pressure_segments.append(pressure_segments)
                    processed_files += 1
                    total_segments += len(optical_segments)
                
                # 定期打印进度
                if processed_files % 10 == 0:
                    print(f"已处理 {processed_files} 个文件，当前总段数: {total_segments}")
                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错：{e}")
                continue
    
    # 合并所有段
    if all_optical_segments:
        all_optical_segments = np.vstack(all_optical_segments)
        all_pressure_segments = np.vstack(all_pressure_segments)
        
        # 保存处理后的数据
        np.save(os.path.join(save_folder, 'optical_segments.npy'), all_optical_segments)
        np.save(os.path.join(save_folder, 'pressure_segments.npy'), all_pressure_segments)
        
        print(f"\n处理完成:")
        print(f"总处理文件数: {processed_files}")
        print(f"光学信号段形状: {all_optical_segments.shape}")
        print(f"压力信号段形状: {all_pressure_segments.shape}")
        
        # 计打印一些统计信
        print("\n数据统计:")
        print(f"光学信号 - 均值: {np.mean(all_optical_segments):.4f}, "
              f"标准差: {np.std(all_optical_segments):.4f}, "
              f"最小值: {np.min(all_optical_segments):.4f}, "
              f"最大值: {np.max(all_optical_segments):.4f}")
        print(f"压力信号 - 均值: {np.mean(all_pressure_segments):.4f}, "
              f"标准差: {np.std(all_pressure_segments):.4f}, "
              f"最小值: {np.min(all_pressure_segments):.4f}, "
              f"最大值: {np.max(all_pressure_segments):.4f}")
        
        # 计算相关性
        correlations = []
        for i in range(min(1000, len(all_optical_segments))):
            corr = np.corrcoef(all_optical_segments[i], all_pressure_segments[i])[0,1]
            if not np.isnan(corr):
                correlations.append(corr)
        print(f"信号相关性 - 均值: {np.mean(correlations):.4f}, 标准差: {np.std(correlations):.4f}")
        
    else:
        print("没有成功处理任何数据")

def load_bp_measurements(auscultatory_path, oscillometric_path, quality_threshold=0.65):
    """
    加载血压测量数据，并根据质量阈值筛选
    """
    # 读取两种测量方法的数据
    aus_data = pd.read_csv(auscultatory_path, sep='\t')
    osc_data = pd.read_csv(oscillometric_path, sep='\t')
    
    # 合并数据并筛选
    all_measurements = pd.concat([aus_data, osc_data], ignore_index=True)
    qualified_measurements = all_measurements[
        (all_measurements['pressure_quality'] >= quality_threshold) &
        (all_measurements['waveform_file_path'].notna()) &
        (all_measurements['sbp'].notna()) &
        (all_measurements['dbp'].notna()) &
        (all_measurements['waveforms_generated'] == 1)  # 确保波形文件已生成
    ]
    
    return qualified_measurements

def split_subjects(measurements_df, test_ratio=0.2, random_state=42):
    """
    按受试者ID分割训练集和测试集
    
    Args:
        measurements_df: 测量数据DataFrame
        test_ratio: 测试集比例
        random_state: 随机种子
    
    Returns:
        train_df, test_df: 训练集和测试集DataFrame
    """
    # 获取唯一的受试者ID
    unique_pids = measurements_df['pid'].unique()
    
    # 随机分割受试者ID
    np.random.seed(random_state)
    test_pids = np.random.choice(
        unique_pids, 
        size=int(len(unique_pids) * test_ratio), 
        replace=False
    )
    
    # 根据受试者ID分割数据
    test_df = measurements_df[measurements_df['pid'].isin(test_pids)]
    train_df = measurements_df[~measurements_df['pid'].isin(test_pids)]
    
    print(f"\n数据集分割:")
    print(f"总受试者数: {len(unique_pids)}")
    print(f"训练集受试者数: {len(unique_pids) - len(test_pids)}")
    print(f"测试集受试者数: {len(test_pids)}")
    print(f"训练集测量数: {len(train_df)}")
    print(f"测试集测量数: {len(test_df)}")
    
    return train_df, test_df

def process_waveforms_with_bp(measurements_df, base_dir, window_size=4096):
    """处理波形数据并关联血压值"""
    pressure_segments = []
    ekg_segments = []
    sbp_values = []
    dbp_values = []
    pids = []
    measurements = []
    segment_indices = []
    step_size = window_size // 8
    for _, row in tqdm(measurements_df.iterrows(), 
                      total=len(measurements_df), 
                      desc="处理波形文件"):
        if (pd.isna(row['sbp']) or pd.isna(row['dbp']) or 
            pd.isna(row['waveform_file_path']) or 
            row['pressure_quality'] < 0.5 or 
            row['waveforms_generated'] != 1):
            continue
            
        waveform_path = os.path.join(base_dir, row['waveform_file_path'])
        
        try:
            waveform_data = pd.read_csv(waveform_path, sep='\t')
            pressure_signal = waveform_data['pressure'].values
            ekg_signal = waveform_data['ekg'].values
            
            # 应用带通滤波和标准化
            pressure_filtered = apply_bandpass_filter(pressure_signal)
            ekg_filtered = apply_bandpass_filter(ekg_signal)
            
            # 标准化
            pressure_norm = (pressure_filtered - np.mean(pressure_filtered)) / (np.std(pressure_filtered) + 1e-8)
            ekg_norm = (ekg_filtered - np.mean(ekg_filtered)) / (np.std(ekg_filtered) + 1e-8)
            
            # 使用滑动窗口分段
            for i in range(0, len(pressure_norm) - window_size, step_size):
                pressure_segments.append(pressure_norm[i:i + window_size])
                ekg_segments.append(ekg_norm[i:i + window_size])
                sbp_values.append(row['sbp'])
                dbp_values.append(row['dbp'])
                pids.append(row['pid'])
                measurements.append(row['measurement'])
                segment_indices.append(i // (step_size))
                
        except Exception as e:
            print(f"处理文件时出错 {waveform_path}: {str(e)}")
            continue
    
    data_dict = {
        'pressure_segments': np.array(pressure_segments),
        'ekg_segments': np.array(ekg_segments),
        'sbp_values': np.array(sbp_values),
        'dbp_values': np.array(dbp_values),
        'pids': np.array(pids),
        'measurements': np.array(measurements),
        'segment_indices': np.array(segment_indices)
    }
    
    print(f"\n处理完成：")
    print(f"- 信号段数量: {len(pressure_segments)}")
    print(f"- 压力信号形状: {data_dict['pressure_segments'].shape}")
    print(f"- EKG信号形状: {data_dict['ekg_segments'].shape}")
    print(f"- 不同受试者数量: {len(np.unique(pids))}")
    print(f"- 不同测量类型数量: {len(np.unique(measurements))}")
    
    return data_dict

def save_processed_data(data_dict, save_dir):
    """保存处理后的数据"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存信号数据
    np.save(os.path.join(save_dir, 'pressure_segments.npy'), data_dict['pressure_segments'])
    np.save(os.path.join(save_dir, 'ekg_segments.npy'), data_dict['ekg_segments'])
    
    # 保存更详细的元数据
    metadata = pd.DataFrame({
        'pid': data_dict['pids'],
        'measurement': data_dict['measurements'],
        'segment_index': data_dict['segment_indices'],
        'sbp': data_dict['sbp_values'],
        'dbp': data_dict['dbp_values']
    })
    metadata.to_csv(os.path.join(save_dir, 'metadata.csv'), index=False)
    
    print(f"\n数据已保存至 {save_dir}")
    print(f"元数据示例：")
    print(metadata.head())

if __name__ == "__main__":
    # 原有的数据处理
    data_folders = ['measurements_auscultatory', 'measurements_oscillometric']
    # load_signals(data_folders)
    
    # 新增的血压预测数据处理
    print("\n开始处理血压预测数据...")
    
    # 加载并筛选测量数据
    measurements = load_bp_measurements(
        'measurements_auscultatory.tsv',
        'measurements_oscillometric.tsv',
        quality_threshold=0.65
    )
    
    # 分割训练集和测试集
    train_df, test_df = split_subjects(measurements)
    
    # 处理训练集数据
    print("\n处理训练集...")
    train_data = process_waveforms_with_bp(train_df, './')
    save_processed_data(train_data, 'preprocessed_data/train')
    
    # 处理测试集数据
    print("\n处理测试集...")
    test_data = process_waveforms_with_bp(test_df, './')
    save_processed_data(test_data, 'preprocessed_data/test')
