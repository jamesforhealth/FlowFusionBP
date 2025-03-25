import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class SignalTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=6, dim_feedforward=512):
        super().__init__()
        
        # 信号编码
        self.input_embedding = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 解码层
        self.output_layer = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, input_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # x shape: [batch_size, channels, seq_len]
        x = self.input_embedding(x)  # [batch_size, d_model, seq_len]
        x = x.transpose(1, 2)  # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x = self.output_layer(x)  # [batch_size, channels, seq_len]
        return x

class WaveformDataset(Dataset):
    """波形数据集"""
    def __init__(self, optical_segments, pressure_segments):
        """
        Args:
            optical_segments: ���处理好的光学信号段 (N, window_size)
            pressure_segments: 预处理好的压力信号段 (N, window_size)
        """
        self.optical = torch.FloatTensor(optical_segments)
        self.pressure = torch.FloatTensor(pressure_segments)
        
        # 确保形状正确
        assert self.optical.shape == self.pressure.shape, "光学信号和压力信号的形状不匹配"
        
    def __len__(self):
        return len(self.optical)
    
    def __getitem__(self, idx):
        return {
            'optical': self.optical[idx].unsqueeze(0),  # 添加通道维度 (1, window_size)
            'pressure': self.pressure[idx].unsqueeze(0)  # 添加通道维度 (1, window_size)
        }

class Conv1d_batchnorm(nn.Module):
    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride=1, activation='relu'):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv1d(
            in_channels=num_in_filters, 
            out_channels=num_out_filters, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding='same'
        )
        self.batchnorm = nn.BatchNorm1d(num_out_filters)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        if self.activation == 'relu':
            return nn.functional.relu(x)
        return x

class Multiresblock(nn.Module):
    def __init__(self, num_in_channels, num_filters, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha
        
        # 按原始比例分配滤波器
        self.filt_cnt_3x3 = int(self.W*0.167)
        self.filt_cnt_5x5 = int(self.W*0.333)
        self.filt_cnt_7x7 = int(self.W*0.5)
        self.num_out_filters = self.filt_cnt_3x3 + self.filt_cnt_5x5 + self.filt_cnt_7x7
        
        self.shortcut = Conv1d_batchnorm(num_in_channels, self.num_out_filters, kernel_size=1, activation='None')
        self.conv_3x3 = Conv1d_batchnorm(num_in_channels, self.filt_cnt_3x3, kernel_size=3, activation='relu')
        self.conv_5x5 = Conv1d_batchnorm(self.filt_cnt_3x3, self.filt_cnt_5x5, kernel_size=5, activation='relu')
        self.conv_7x7 = Conv1d_batchnorm(self.filt_cnt_5x5, self.filt_cnt_7x7, kernel_size=7, activation='relu')
        
        self.batch_norm1 = nn.BatchNorm1d(self.num_out_filters)
        self.batch_norm2 = nn.BatchNorm1d(self.num_out_filters)

    def forward(self, x):
        shortcut = self.shortcut(x)
        
        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)
        
        x = torch.cat([a, b, c], dim=1)
        x = self.batch_norm1(x)
        
        x = x + shortcut
        x = self.batch_norm2(x)
        x = nn.functional.relu(x)
        
        return x

class Respath(nn.Module):
    def __init__(self, num_in_filters, num_out_filters, respath_length):
        super().__init__()
        
        self.respath_length = respath_length
        self.shortcuts = nn.ModuleList([])
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        
        for i in range(self.respath_length):
            if i == 0:
                self.shortcuts.append(Conv1d_batchnorm(num_in_filters, num_out_filters, kernel_size=1, activation='None'))
                self.convs.append(Conv1d_batchnorm(num_in_filters, num_out_filters, kernel_size=3, activation='relu'))
            else:
                self.shortcuts.append(Conv1d_batchnorm(num_out_filters, num_out_filters, kernel_size=1, activation='None'))
                self.convs.append(Conv1d_batchnorm(num_out_filters, num_out_filters, kernel_size=3, activation='relu'))
            
            self.bns.append(nn.BatchNorm1d(num_out_filters))
    
    def forward(self, x):
        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)
            
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = nn.functional.relu(x)
            
            x = x + shortcut
            x = self.bns[i](x)
            x = nn.functional.relu(x)
        
        return x

class MultiResUNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, alpha=1.67):
        super().__init__()
        
        self.alpha = alpha
        
        # Encoder Path
        self.multiresblock1 = Multiresblock(input_channels, 32)
        self.in_filters1 = int(32*self.alpha*0.167) + int(32*self.alpha*0.333) + int(32*self.alpha*0.5)
        self.pool1 = nn.MaxPool1d(2)
        self.respath1 = Respath(self.in_filters1, 32, respath_length=4)
        
        self.multiresblock2 = Multiresblock(self.in_filters1, 32*2)
        self.in_filters2 = int(32*2*self.alpha*0.167) + int(32*2*self.alpha*0.333) + int(32*2*self.alpha*0.5)
        self.pool2 = nn.MaxPool1d(2)
        self.respath2 = Respath(self.in_filters2, 32*2, respath_length=3)
        
        self.multiresblock3 = Multiresblock(self.in_filters2, 32*4)
        self.in_filters3 = int(32*4*self.alpha*0.167) + int(32*4*self.alpha*0.333) + int(32*4*self.alpha*0.5)
        self.pool3 = nn.MaxPool1d(2)
        self.respath3 = Respath(self.in_filters3, 32*4, respath_length=2)
        
        self.multiresblock4 = Multiresblock(self.in_filters3, 32*8)
        self.in_filters4 = int(32*8*self.alpha*0.167) + int(32*8*self.alpha*0.333) + int(32*8*self.alpha*0.5)
        self.pool4 = nn.MaxPool1d(2)
        self.respath4 = Respath(self.in_filters4, 32*8, respath_length=1)
        
        self.multiresblock5 = Multiresblock(self.in_filters4, 32*16)
        self.in_filters5 = int(32*16*self.alpha*0.167) + int(32*16*self.alpha*0.333) + int(32*16*self.alpha*0.5)
        
        # Decoder path
        self.upsample6 = nn.ConvTranspose1d(self.in_filters5, 32*8, kernel_size=2, stride=2)
        self.concat_filters1 = 32*8*2
        self.multiresblock6 = Multiresblock(self.concat_filters1, 32*8)
        self.in_filters6 = int(32*8*self.alpha*0.167) + int(32*8*self.alpha*0.333) + int(32*8*self.alpha*0.5)
        
        self.upsample7 = nn.ConvTranspose1d(self.in_filters6, 32*4, kernel_size=2, stride=2)
        self.concat_filters2 = 32*4*2
        self.multiresblock7 = Multiresblock(self.concat_filters2, 32*4)
        self.in_filters7 = int(32*4*self.alpha*0.167) + int(32*4*self.alpha*0.333) + int(32*4*self.alpha*0.5)
        
        self.upsample8 = nn.ConvTranspose1d(self.in_filters7, 32*2, kernel_size=2, stride=2)
        self.concat_filters3 = 32*2*2
        self.multiresblock8 = Multiresblock(self.concat_filters3, 32*2)
        self.in_filters8 = int(32*2*self.alpha*0.167) + int(32*2*self.alpha*0.333) + int(32*2*self.alpha*0.5)
        
        self.upsample9 = nn.ConvTranspose1d(self.in_filters8, 32, kernel_size=2, stride=2)
        self.concat_filters4 = 32*2
        self.multiresblock9 = Multiresblock(self.concat_filters4, 32)
        self.in_filters9 = int(32*self.alpha*0.167) + int(32*self.alpha*0.333) + int(32*self.alpha*0.5)
        
        self.conv_final = Conv1d_batchnorm(self.in_filters9, num_classes, kernel_size=1, activation='None')
        
    def forward(self, x):
        x_multires1 = self.multiresblock1(x)
        x_pool1 = self.pool1(x_multires1)
        x_multires1 = self.respath1(x_multires1)
        
        x_multires2 = self.multiresblock2(x_pool1)
        x_pool2 = self.pool2(x_multires2)
        x_multires2 = self.respath2(x_multires2)
        
        x_multires3 = self.multiresblock3(x_pool2)
        x_pool3 = self.pool3(x_multires3)
        x_multires3 = self.respath3(x_multires3)
        
        x_multires4 = self.multiresblock4(x_pool3)
        x_pool4 = self.pool4(x_multires4)
        x_multires4 = self.respath4(x_multires4)
        
        x_multires5 = self.multiresblock5(x_pool4)
        
        up6 = torch.cat([self.upsample6(x_multires5), x_multires4], dim=1)
        x_multires6 = self.multiresblock6(up6)
        
        up7 = torch.cat([self.upsample7(x_multires6), x_multires3], dim=1)
        x_multires7 = self.multiresblock7(up7)
        
        up8 = torch.cat([self.upsample8(x_multires7), x_multires2], dim=1)
        x_multires8 = self.multiresblock8(up8)
        
        up9 = torch.cat([self.upsample9(x_multires8), x_multires1], dim=1)
        x_multires9 = self.multiresblock9(up9)
        
        out = self.conv_final(x_multires9)
        return out

def analyze_data_quality(optical_path, pressure_path, window_size=200):
    """分析数据质量"""
    optical_data = np.load(optical_path)
    pressure_data = np.load(pressure_path)
    
    print("数据基本统计：")
    print(f"光学信号形状: {optical_data.shape}")
    print(f"压力信号形状: {pressure_data.shape}")
    
    # 检查数值范围
    print("\n数值范围：")
    print(f"光学信号: min={optical_data.min():.4f}, max={optical_data.max():.4f}, mean={optical_data.mean():.4f}, std={optical_data.std():.4f}")
    print(f"压力信号: min={pressure_data.min():.4f}, max={pressure_data.max():.4f}, mean={pressure_data.mean():.4f}, std={pressure_data.std():.4f}")
    
    # 检查NaN和异常值
    print("\n数据质量检查：")
    print(f"光学信号NaN数量: {np.isnan(optical_data).sum()}")
    print(f"压力信号NaN数量: {np.isnan(pressure_data).sum()}")
    
    # 计算信号相关性
    correlations = []
    for i in range(0, min(1000, len(optical_data))):
        corr = np.corrcoef(optical_data[i], pressure_data[i])[0,1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    print(f"\n信号相关性: mean={np.mean(correlations):.4f}, std={np.std(correlations):.4f}")
    
    return optical_data, pressure_data

def calculate_initial_loss(model, data_loader, criterion, device):
    """计算初始loss"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="计算初始loss"):
            # 修正键名：从'optical'和'pressure'获取数据
            optical = batch['optical'].to(device)
            pressure = batch['pressure'].to(device)
            
            # 使用pressure作为目标
            outputs = model(optical)
            loss = criterion(outputs, pressure)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def train_model(model_type="multiresunet", optical_path=None, pressure_path=None, 
                num_epochs=500, batch_size=32, valid_split=0.2):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    optical_segments = np.load(optical_path)
    pressure_segments = np.load(pressure_path)
    print(f"光学信号形状: {optical_segments.shape}")
    print(f"压力信号形状: {pressure_segments.shape}")
    
    # 使用完整数据集
    dataset = WaveformDataset(
        optical_segments=optical_segments,
        pressure_segments=pressure_segments
    )
    
    # 创建数据加载器
    train_size = int((1 - valid_split) * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 创建模型、优化器和损失函数
    if model_type == "multiresunet":
        model = MultiResUNet().to(device)
    else:
        model = SignalTransformer().to(device)
    print(model)
    print(f"模型类型: {model_type}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    if os.path.exists(f'best_{model_type}_model.pth'):
        model.load_state_dict(torch.load(f'best_{model_type}_model.pth'))
        print("加载最佳模型")

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 添加L2正则化
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.L1Loss()
    
    # 计算初始loss
    print("\n计算初始loss...")
    initial_train_loss = calculate_initial_loss(model, train_loader, criterion, device)
    initial_valid_loss = calculate_initial_loss(model, valid_loader, criterion, device)
    print(f"初始训练Loss: {initial_train_loss:.6f}")
    print(f"初始验证Loss: {initial_valid_loss:.6f}")
    
    # 训练循环
    best_valid_loss = initial_valid_loss
    patience = 10
    patience_counter = 0
    train_losses = [initial_train_loss]
    valid_losses = [initial_valid_loss]
    
    print("\n开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            optical = batch['optical'].to(device)
            pressure = batch['pressure'].to(device)
            
            optimizer.zero_grad()
            outputs = model(optical)
            loss = criterion(outputs, pressure)
            loss.backward()
            
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Valid]'):
                optical = batch['optical'].to(device)
                pressure = batch['pressure'].to(device)
                
                outputs = model(optical)
                loss = criterion(outputs, pressure)
                valid_loss += loss.item()
                
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Valid Loss: {avg_valid_loss:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 更新学习率
        scheduler.step(avg_valid_loss)
        
        # 早停检查
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), f'best_{model_type}_model.pth')
            patience_counter = 0
            print(f"保存新的最佳模型 (Loss: {best_valid_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n验证loss连续{patience}个epoch没有改善，停止训练")
                break
            
        # 每个epoch都保存预测结果
        visualize_predictions(model, valid_loader, device, epoch)
    
    return train_losses, valid_losses

def visualize_predictions(model, valid_loader, device, epoch):
    """可视化预测结果"""
    model.eval()
    with torch.no_grad():
        # 获取一个批次的数据
        batch = next(iter(valid_loader))
        optical = batch['optical'].to(device)
        pressure = batch['pressure'].to(device)
        
        # 预测
        outputs = model(optical)
        
        # 转回CPU并转为numpy数组
        outputs = outputs.cpu().numpy()
        pressure = pressure.cpu().numpy()
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 随机选择一些样本进行可视化
        n_samples = min(5, len(outputs))
        for i in range(n_samples):
            plt.subplot(n_samples, 1, i+1)
            plt.plot(pressure[i, 0], label='实际信号')
            plt.plot(outputs[i, 0], label='预测信号')
            plt.legend()
            plt.title(f'样本 {i+1}')
        
        plt.tight_layout()
        plt.savefig(f'predictions_epoch_{epoch+1}.png')
        plt.close()
        
        # 计算并打印评估指标
        mae = np.mean(np.abs(outputs - pressure))
        print(f"\n预测评估 (Epoch {epoch+1}):")
        print(f"平均绝对误差: {mae:.4f}")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')  # 设置多进程启动方法
    
    optical_path = 'preprocessed_data/optical_segments.npy'
    pressure_path = 'preprocessed_data/pressure_segments.npy'
    
    print("训练 MultiResUNet...")
    train_losses_unet, valid_losses_unet = train_model(
        model_type="multiresunet",
        optical_path=optical_path,
        pressure_path=pressure_path
    )
    
    # print("\n训练 Transformer...")
    # train_losses_transformer, valid_losses_transformer = train_model(
    #     model_type="transformer",
    #     optical_path=optical_path,
    #     pressure_path=pressure_path
    # )