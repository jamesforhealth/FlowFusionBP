import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR

class BPDataset(Dataset):
    """血压预测数据集"""
    def __init__(self, pressure_segments, ekg_segments, metadata):
        """
        Args:
            pressure_segments: 压力信号段数据 (.npy文件)
            ekg_segments: EKG信号段数据 (.npy文件)
            metadata: 包含血压值的元数据 (.csv文件)
        """
        self.pressure = torch.FloatTensor(np.load(pressure_segments))
        self.ekg = torch.FloatTensor(np.load(ekg_segments))
        self.metadata = pd.read_csv(metadata)
        
        print(f"\n数据集信息:")
        print(f"压力信号形状: {self.pressure.shape}")  # 应该是 (N, 4096)
        print(f"EKG信号形状: {self.ekg.shape}")       # 应该是 (N, 4096)
        
        # 添加形状检查
        assert self.pressure.shape[1] == 4096, f"压力信号长度应为4096，实际为{self.pressure.shape[1]}"
        assert self.ekg.shape[1] == 4096, f"EKG信号长度应为4096，实际为{self.ekg.shape[1]}"
        
        # 血压值不需要标准化，因为它们本身就是有意义的物理量
        self.sbp = torch.FloatTensor(self.metadata['sbp'].values)
        self.dbp = torch.FloatTensor(self.metadata['dbp'].values)
        
        # 打印一些统计信息
        print("\n血压值统计信息:")
        print(f"SBP - 均值: {self.sbp.mean():.2f}, 标准差: {self.sbp.std():.2f}")
        print(f"SBP - 最小值: {self.sbp.min():.2f}, 最大值: {self.sbp.max():.2f}")
        print(f"DBP - 均值: {self.dbp.mean():.2f}, 标准差: {self.dbp.std():.2f}")
        print(f"DBP - 最小值: {self.dbp.min():.2f}, 最大值: {self.dbp.max():.2f}")
        
    def __len__(self):
        return len(self.pressure)
    
    def __getitem__(self, idx):
        return {
            'pressure': self.pressure[idx].unsqueeze(0),
            'ekg': self.ekg[idx].unsqueeze(0),
            'sbp': self.sbp[idx],
            'dbp': self.dbp[idx]
        }

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super().__init__()
        
        print(f"初始化 MultiScaleBlock: in_channels={in_channels}, out_channels={out_channels}")
        
        self.out_channels = out_channels
        self.branch_channels = out_channels // 6
        
        # 确保分组数合适
        groups = min(4, self.branch_channels, in_channels)
        while (in_channels % groups != 0) or (self.branch_channels % groups != 0):
            groups -= 1
            
        print(f"分支通道数: {self.branch_channels}, 分组数: {groups}")
        
        # 1x1 卷积分支
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, self.branch_channels, kernel_size=1),
            nn.BatchNorm1d(self.branch_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate/2)
        )
        
        # 3x3 卷积分支
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, self.branch_channels, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(self.branch_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate/2)
        )
        
        # 5x5 卷积分支
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels, self.branch_channels, kernel_size=5, padding=2, groups=groups),
            nn.BatchNorm1d(self.branch_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate/2)
        )
        
        # 7x7 卷积分支
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels, self.branch_channels, kernel_size=7, padding=3, groups=groups),
            nn.BatchNorm1d(self.branch_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate/2)
        )
        
        # 11x11 卷积分支
        self.conv11 = nn.Sequential(
            nn.Conv1d(in_channels, self.branch_channels, kernel_size=11, padding=5, groups=groups),
            nn.BatchNorm1d(self.branch_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate/2)
        )
        
        # 全局信息分支 - 修改为保持时间维度
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, self.branch_channels, 1),
            nn.BatchNorm1d(self.branch_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate/2)
        )
        
        # 通道调整层
        self.channel_adj = nn.Sequential(
            nn.Conv1d(self.branch_channels * 6, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate)
        )
        
        # SE注意力
        self.se = SELayer(out_channels, reduction=8)
        
        # 残差连接
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        # print(f"MultiScaleBlock input shape: {x.shape}")
        
        # 保存残差连接
        identity = x
        
        # 各分支特征提取
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x11 = self.conv11(x)
        
        # 全局信息分支 - 扩展时间维度以匹配其他分支
        xg = self.global_branch(x)
        xg = xg.expand(-1, -1, x.size(2))  # 扩展到与输入相同的时间维度
        
        # print(f"Branch shapes: {x1.shape}, {x3.shape}, {x5.shape}, {x7.shape}, {x11.shape}, {xg.shape}")
        
        # 特征拼接
        x_cat = torch.cat([x1, x3, x5, x7, x11, xg], dim=1)
        # print(f"Concatenated shape: {x_cat.shape}")
        
        # 调整通道数
        x = self.channel_adj(x_cat)
        # print(f"After channel adjustment: {x.shape}")
        
        # SE注意力
        x = self.se(x)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity
        
        # print(f"Output shape: {x.shape}")
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # print(f"SELayer init - channel: {channel}, reduction: {reduction}")
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )
        
        # print(f"SE fc layers: {channel} -> {channel // reduction} -> {channel}")

    def forward(self, x):
        b, c, _ = x.size()
        # print(f"SE input shape: {x.shape}")
        
        y = self.avg_pool(x).view(b, c)
        # print(f"SE after pooling shape: {y.shape}")
        
        y = self.fc(y).view(b, c, 1)
        # print(f"SE after fc shape: {y.shape}")
        
        return x * y.expand_as(x)

class ImprovedDeepBPNet(nn.Module):
    """改进的深度血压预测模型 - 使用多尺度特征提取"""
    def __init__(self, input_size=4096):
        super().__init__()
        
        # 压力信号处理分支
        self.pressure_net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 4096 -> 2048
            
            MultiScaleBlock(64, 128),
            nn.MaxPool1d(2),  # 2048 -> 1024
            
            MultiScaleBlock(128, 256),
            nn.MaxPool1d(2),  # 1024 -> 512
            
            MultiScaleBlock(256, 512),
            nn.MaxPool1d(2),  # 512 -> 256
            
            MultiScaleBlock(512, 512),
            nn.MaxPool1d(2),  # 256 -> 128
            
            nn.AdaptiveAvgPool1d(1)  # -> [B, 512, 1]
        )
        
        # EKG信号处理分支
        self.ekg_net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            MultiScaleBlock(64, 128),
            nn.MaxPool1d(2),
            
            MultiScaleBlock(128, 256),
            nn.MaxPool1d(2),
            
            MultiScaleBlock(256, 512),
            nn.MaxPool1d(2),
            
            MultiScaleBlock(512, 512),
            nn.MaxPool1d(2),
            
            nn.AdaptiveAvgPool1d(1)  # -> [B, 512, 1]
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Flatten(),  # [B, 1024, 1] -> [B, 1024]
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 2)  # 输出SBP和DBP
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, pressure, ekg):
        # print(f"\nInput shapes - Pressure: {pressure.shape}, EKG: {ekg.shape}")
        
        # 特征提取
        pressure_features = self.pressure_net(pressure)
        #   print(f"Pressure features: {pressure_features.shape}")
        
        ekg_features = self.ekg_net(ekg)
        # print(f"EKG features: {ekg_features.shape}")
        
        # 特征融合
        combined_features = torch.cat([pressure_features, ekg_features], dim=1)
        # print(f"Combined features: {combined_features.shape}")
        
        # 预测
        output = self.fusion(combined_features)
        # print(f"Output: {output.shape}")
        
        return output

def calculate_initial_metrics(model, data_loader, device):
    """计算初始指标"""
    model.eval()
    total_loss = 0
    sbp_errors = []
    dbp_errors = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="计算初始指标"):
            pressure = batch['pressure'].to(device)
            ekg = batch['ekg'].to(device)
            targets = torch.stack([batch['sbp'], batch['dbp']], dim=1).to(device)
            # print(f"targets: {targets}")
            outputs = model(pressure, ekg)
            # input(f"outputs: {outputs}")
            loss = nn.MSELoss()(outputs, targets)
            total_loss += loss.item()
            
            # 计算MAE
            sbp_errors.extend(abs(outputs[:, 0].cpu() - targets[:, 0].cpu()).numpy())
            dbp_errors.extend(abs(outputs[:, 1].cpu() - targets[:, 1].cpu()).numpy())
    
    avg_loss = total_loss / len(data_loader)
    sbp_mae = np.mean(sbp_errors)
    dbp_mae = np.mean(dbp_errors)
    
    return avg_loss, sbp_mae, dbp_mae

def train_model(train_loader, valid_loader, device, num_epochs=100):
    """训练模型"""
    model = ImprovedDeepBPNet().to(device)
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    input(f"模型参数数量: {total_params:,}, model: {model}")


    # 加载预训练模型
    model.load_state_dict(torch.load('best_bp_model.pth'))

    # 使用更强的权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    
    # 使用One Cycle学习率调度
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # 使用组合损失函数
    def combined_loss(outputs, targets):
        mse_loss = F.mse_loss(outputs, targets)
        mae_loss = F.l1_loss(outputs, targets)
        return 0.5 * mse_loss + 0.5 * mae_loss
    
    # EMA模型
    ema_model = AveragedModel(model, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: 
        0.999 * averaged_model_parameter + 0.001 * model_parameter)
    
    # 计算初始指标
    print("\n计算初始指标...")
    init_loss, init_sbp_mae, init_dbp_mae = calculate_initial_metrics(model, valid_loader, device)
    print(f"初始验证Loss: {init_loss:.6f}")
    print(f"初始SBP MAE: {init_sbp_mae:.2f} mmHg")
    print(f"初始DBP MAE: {init_dbp_mae:.2f} mmHg")
    
    best_valid_loss = init_loss
    patience = 10
    patience_counter = 0
    train_losses = []
    valid_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_sbp_errors = []
        train_dbp_errors = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            pressure = batch['pressure'].to(device)
            ekg = batch['ekg'].to(device)
            targets = torch.stack([batch['sbp'], batch['dbp']], dim=1).to(device)
            
            optimizer.zero_grad()
            outputs = model(pressure, ekg)
            loss = combined_loss(outputs, targets)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            # 计算训练MAE
            train_sbp_errors.extend(abs(outputs[:, 0].detach().cpu() - targets[:, 0].cpu()).numpy())
            train_dbp_errors.extend(abs(outputs[:, 1].detach().cpu() - targets[:, 1].cpu()).numpy())
        
        # 验证阶段
        model.eval()
        valid_loss = 0
        valid_sbp_errors = []
        valid_dbp_errors = []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Valid]'):
                pressure = batch['pressure'].to(device)
                ekg = batch['ekg'].to(device)
                targets = torch.stack([batch['sbp'], batch['dbp']], dim=1).to(device)
                
                outputs = model(pressure, ekg)
                loss = combined_loss(outputs, targets)
                valid_loss += loss.item()
                
                # 计算验证MAE
                valid_sbp_errors.extend(abs(outputs[:, 0].cpu() - targets[:, 0].cpu()).numpy())
                valid_dbp_errors.extend(abs(outputs[:, 1].cpu() - targets[:, 1].cpu()).numpy())
        
        # 计算平均指标
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        train_sbp_mae = np.mean(train_sbp_errors)
        train_dbp_mae = np.mean(train_dbp_errors)
        valid_sbp_mae = np.mean(valid_sbp_errors)
        valid_dbp_mae = np.mean(valid_dbp_errors)
        
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}")
        print(f"Train MAE - SBP: {train_sbp_mae:.2f} mmHg, DBP: {train_dbp_mae:.2f} mmHg")
        print(f"Valid MAE - SBP: {valid_sbp_mae:.2f} mmHg, DBP: {valid_dbp_mae:.2f} mmHg")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 更新学习率
        scheduler.step()
        
        # 早停检查
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), 'best_bp_model.pth')
            patience_counter = 0
            print("保存新最佳模型")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n验证loss连续{patience}个epoch没有改善，停止训练")
                break
    
    return train_losses, valid_losses

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载训练集
    train_dataset = BPDataset(
        'preprocessed_data/train/pressure_segments.npy',
        'preprocessed_data/train/ekg_segments.npy',
        'preprocessed_data/train/metadata.csv'
    )
    
    # 加载验证集
    valid_dataset = BPDataset(
        'preprocessed_data/test/pressure_segments.npy',
        'preprocessed_data/test/ekg_segments.npy',
        'preprocessed_data/test/metadata.csv'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    
    # 训练模型
    train_losses, valid_losses = train_model(train_loader, valid_loader, device)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()

if __name__ == '__main__':
    main()