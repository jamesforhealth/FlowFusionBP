#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【方向二 – 表示學習＋自監督模型】 (改版)

本程式將每個 h5 檔案視為一個樣本，
讀取所有 sliding window（形狀 (N, WINDOW_LENGTH)），
將 EKG 與所選 sensor（PPG 或 Tonometry）合併成 2 通道輸入，
利用深層 CNN (DeepWindowEncoder) 逐層 pooling 萃取特徵，
並在同一筆量測內假定短時間內用以預測血壓的潛在特徵是不變的，
因此分兩階段訓練：
  Phase 1：僅訓練 encoder，使各 window 的 representation 一致（consistency loss）
  Phase 2：端到端訓練回歸任務，預測 measurement_sbp 與 measurement_dbp

訓練中若驗證損失出現 nan 或 inf，則提前停止訓練。

注意：
  - 本範例僅回歸兩個目標值（經正規化後）。
  - 若要同時比較 PPG 與 Tonometry，可分別用 BPWholeDataset(sensor='ppg') 與 BPWholeDataset(sensor='tonometry') 建立 DataLoader，然後用相同模型架構訓練。
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

#############################################
# 參數設定
#############################################
WINDOW_LENGTH = 4096
WINDOW_STEP = 256
H5_FOLDER = 'output_h5_folder'  # 請調整至你的 h5 資料夾路徑

#############################################
# Dataset：每個 h5 檔案作為一個樣本
#############################################
class BPWholeDataset(Dataset):
    def __init__(self, h5_folder, sensor='ppg', blacklist_file='h5_nan_blacklist.txt'):
        """
        h5_folder: 存放 h5 檔案的資料夾
        sensor: 'ppg' 或 'tonometry'
        blacklist_file: 黑名單文件路徑，用於存放包含 NaN 的檔案名
        """
        self.h5_folder = h5_folder
        self.sensor = sensor.lower()
        all_files = [os.path.join(h5_folder, f) for f in os.listdir(h5_folder) if f.endswith('.h5')]
        
        # 讀取黑名單
        self.blacklist = set()
        if os.path.exists(blacklist_file):
            with open(blacklist_file, 'r') as f:
                self.blacklist = set(line.strip() for line in f)
            print(f"已從 {blacklist_file} 讀取 {len(self.blacklist)} 個黑名單文件")
        
        # 過濾掉黑名單中的文件
        self.files = [f for f in all_files if f not in self.blacklist]
        print(f"總文件數: {len(all_files)}, 過濾後: {len(self.files)}")
        
        if len(self.files) == 0:
            raise ValueError("沒有找到有效的 h5 檔案，請檢查資料夾。")
        
        # 檢查剩餘文件並添加新的問題文件到黑名單
        new_blacklist_entries = set()
        filtered_count = 0
        
        for f in self.files[:]:  # 使用副本進行遍歷，以便在遍歷過程中修改列表
            try:
                with h5py.File(f, 'r') as hf:
                    # 檢查必要的數據集是否存在
                    if 'EKG' not in hf or self.sensor.upper() not in hf:
                        print(f"檔案 {os.path.basename(f)} 缺少必要的數據集，加入黑名單")
                        new_blacklist_entries.add(f)
                        self.files.remove(f)
                        filtered_count += 1
                        continue
                    
                    # 檢查屬性是否存在且不為 NaN
                    try:
                        attr_age = hf.attrs.get('participant_age', None)
                        attr_gender = hf.attrs.get('participant_gender', None)
                        attr_height = hf.attrs.get('participant_height', None)
                        attr_weight = hf.attrs.get('participant_weight', None)
                        attr_sbp = hf.attrs.get('measurement_sbp', None)
                        attr_dbp = hf.attrs.get('measurement_dbp', None)
                        
                        # 檢查所有屬性是否存在
                        if None in (attr_age, attr_gender, attr_height, attr_weight, attr_sbp, attr_dbp):
                            print(f"檔案 {os.path.basename(f)} 屬性缺失，加入黑名單")
                            new_blacklist_entries.add(f)
                            self.files.remove(f)
                            filtered_count += 1
                            continue
                        
                        # 檢查數值屬性是否為 NaN
                        if (np.isnan(float(attr_age)) or np.isnan(float(attr_height)) or
                            np.isnan(float(attr_weight)) or np.isnan(float(attr_sbp)) or
                            np.isnan(float(attr_dbp))):
                            print(f"檔案 {os.path.basename(f)} 屬性有 NaN 值，加入黑名單")
                            new_blacklist_entries.add(f)
                            self.files.remove(f)
                            filtered_count += 1
                            continue
                    except (ValueError, TypeError) as e:
                        print(f"檔案 {os.path.basename(f)} 屬性轉換錯誤: {e}，加入黑名單")
                        new_blacklist_entries.add(f)
                        self.files.remove(f)
                        filtered_count += 1
                        continue
                    
                    # 檢查數據集中是否有 NaN 值
                    ekg_data = hf['EKG'][:]
                    if self.sensor.upper() == 'PPG':
                        sensor_data = hf['PPG'][:]
                    elif self.sensor.upper() == 'TONOMETRY':
                        sensor_data = hf['Tonometry'][:]
                    else:
                        raise ValueError(f"未知的傳感器類型: {self.sensor}")
                    
                    if np.isnan(ekg_data).any() or np.isnan(sensor_data).any():
                        print(f"檔案 {os.path.basename(f)} 數據中有 NaN 值，加入黑名單")
                        new_blacklist_entries.add(f)
                        self.files.remove(f)
                        filtered_count += 1
                        continue
            except (IOError, OSError, ValueError, KeyError, AttributeError) as e:
                print(f"處理檔案 {os.path.basename(f)} 時出錯: {str(e)}，加入黑名單")
                new_blacklist_entries.add(f)
                if f in self.files:
                    self.files.remove(f)
                filtered_count += 1
        
        # 更新黑名單
        if new_blacklist_entries:
            # 讀取原有的黑名單內容
            existing_blacklist = set()
            if os.path.exists(blacklist_file):
                with open(blacklist_file, 'r') as f:
                    existing_blacklist = set(line.strip() for line in f)
            
            # 合併原有的黑名單和新的黑名單項目
            updated_blacklist = existing_blacklist.union(new_blacklist_entries)
            
            # 寫入更新後的黑名單
            with open(blacklist_file, 'w') as f:
                for item in updated_blacklist:
                    f.write(f"{item}\n")
            
            print(f"添加了 {len(new_blacklist_entries)} 個新的黑名單項目")
            self.blacklist.update(new_blacklist_entries)
        
        print(f"剩餘有效文件: {len(self.files)}")
        print(f"總共過濾掉 {filtered_count} 個檔案，黑名單總數: {len(self.blacklist)}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            f = self.files[idx]
            with h5py.File(f, 'r') as hf:
                # 讀取信號數據
                ekg_windows = hf['EKG'][:]
                if self.sensor.upper() == 'PPG':
                    sensor_windows = hf['PPG'][:]
                elif self.sensor.upper() == 'TONOMETRY':
                    sensor_windows = hf['Tonometry'][:]
                else:
                    raise ValueError(f"未知的傳感器類型: {self.sensor}")
                
                # 正規化每個資料集（以各自均值與標準差）
                ekg_windows = (ekg_windows - np.mean(ekg_windows)) / (np.std(ekg_windows) + 1e-8)
                sensor_windows = (sensor_windows - np.mean(sensor_windows)) / (np.std(sensor_windows) + 1e-8)
                windows = np.stack([ekg_windows, sensor_windows], axis=1)  # shape: (N, 2, WINDOW_LENGTH)
                
                # 讀取參與者資訊
                age = float(hf.attrs.get('participant_age', 0)) / 100.0
                gender = hf.attrs.get('participant_gender', "F")
                gender = 1.0 if str(gender).strip().upper() == "M" else 0.0
                height = float(hf.attrs.get('participant_height', 0)) / 200.0
                weight = float(hf.attrs.get('participant_weight', 0)) / 150.0
                personal_info = np.array([age, gender, height, weight], dtype=np.float32)
                
                # 讀取目標血壓值
                sbp = float(hf.attrs.get('measurement_sbp', 0)) / 200.0
                dbp = float(hf.attrs.get('measurement_dbp', 0)) / 200.0
                target = np.array([sbp, dbp], dtype=np.float32)
            
            windows = windows.astype(np.float32)
            return torch.tensor(windows, dtype=torch.float32), torch.tensor(personal_info, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
        
        except Exception as e:
            print(f"獲取樣本 {idx} 時發生錯誤: {str(e)}")
            # 如果出錯，嘗試返回下一個樣本
            return self.__getitem__((idx + 1) % len(self))

#############################################
# 深層 CNN Encoder (DeepWindowEncoder)
#############################################
class DeepWindowEncoder(nn.Module):
    def __init__(self, in_channels=2, base_channels=32, representation_dim=32):
        super(DeepWindowEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(base_channels, base_channels*2, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(base_channels*2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(base_channels*2, base_channels*4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(base_channels*4)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv1d(base_channels*4, base_channels*8, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(base_channels*8)
        self.pool4 = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(base_channels*8, representation_dim)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x).squeeze(-1)
        rep = self.fc(x)
        return rep

class PersonalInfoEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16):
        super(PersonalInfoEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

#############################################
# 模型：深層 Encoder + Regression Head
#############################################
class RepresentationBPModel(nn.Module):
    def __init__(self, in_channels=2, representation_dim=32, base_channels=32):
        super(RepresentationBPModel, self).__init__()
        self.encoder = DeepWindowEncoder(in_channels=in_channels, base_channels=base_channels, representation_dim=representation_dim)
        self.personal_encoder = PersonalInfoEncoder(input_dim=4, hidden_dim=16)
        self.regressor = nn.Sequential(
            nn.Linear(representation_dim + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 只預測 sbp 與 dbp
        )
    
    def forward(self, windows, personal_info):
        # windows: (B, 2, WINDOW_LENGTH) – B 為 mini-batch size (來自同一 h5 的 overlapping mini-batch)
        reps = self.encoder(windows)  # (B, representation_dim)
        rep_mean = reps.mean(dim=0, keepdim=True)  # (1, representation_dim)
        personal_feat = self.personal_encoder(personal_info.unsqueeze(0))  # (1, 16)
        combined = torch.cat([rep_mean, personal_feat], dim=1)  # (1, representation_dim + 16)
        bp_pred = self.regressor(combined)  # (1, 2)
        return reps, rep_mean, bp_pred

#############################################
# Overlapping mini-batch 切分
#############################################
def get_overlapping_batches(windows, inner_batch_size=32, stride=16):
    batches = []
    N = windows.shape[0]
    if N < inner_batch_size:
        return [windows]
    for start in range(0, N - inner_batch_size + 1, stride):
        batch = windows[start:start+inner_batch_size]
        batches.append(batch)
    last_complete_end = ((N - inner_batch_size) // stride) * stride + inner_batch_size
    if last_complete_end < N:
        last_batch = windows[-inner_batch_size:]
        if len(batches) == 0 or not torch.equal(last_batch, batches[-1]):
            batches.append(last_batch)
    return batches

#############################################
# 分階段訓練：
# Phase 1：僅訓練 encoder (Consistency training)
# Phase 2：端到端回歸訓練 (Regression training)
#############################################
def train_consistency_epoch(model, dataloader, optimizer, device, inner_batch_size=32, inner_stride=16):
    model.train()
    mse_loss_fn = nn.MSELoss()
    total_loss = 0.0
    count = 0
    for windows_all, personal_info, _ in tqdm(dataloader, desc="Consistency Training"):
        windows_all = windows_all.squeeze(0)  # (N, 2, WINDOW_LENGTH)
        personal_info = personal_info.squeeze(0)
        mini_batches = get_overlapping_batches(windows_all, inner_batch_size, inner_stride)
        for mini_batch in mini_batches:
            mini_batch = mini_batch.to(device)
            p_info = personal_info.to(device)
            optimizer.zero_grad()
            reps, rep_mean, _ = model(mini_batch, p_info)
            # 若出現 nan，跳過該 mini-batch
            if torch.isnan(reps).any() or torch.isnan(rep_mean).any():
                continue
            loss = mse_loss_fn(reps, rep_mean.expand_as(reps))
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else float('inf')

def train_regression_epoch(model, dataloader, optimizer, device, inner_batch_size=32, inner_stride=16):
    model.train()
    l1_loss_fn = nn.L1Loss()
    total_loss = 0.0
    count = 0
    for windows_all, personal_info, target in tqdm(dataloader, desc="Regression Training"):
        windows_all = windows_all.squeeze(0)
        personal_info = personal_info.squeeze(0)
        target = target.squeeze(0)  # (2,)
        mini_batches = get_overlapping_batches(windows_all, inner_batch_size, inner_stride)
        for mini_batch in mini_batches:
            mini_batch = mini_batch.to(device)
            p_info = personal_info.to(device)
            t_target = target.to(device)
            optimizer.zero_grad()
            _, _, bp_pred = model(mini_batch, p_info)
            # 將 bp_pred 從 (1,2) 轉為 (2,)
            bp_pred = bp_pred[0]
            if torch.isnan(bp_pred).any():
                continue
            loss = l1_loss_fn(bp_pred, t_target)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else float('inf')

def validate_epoch(model, dataloader, device, inner_batch_size=32, inner_stride=16):
    model.eval()
    l1_loss_fn = nn.L1Loss()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for windows_all, personal_info, target in tqdm(dataloader, desc="Validation"):
            windows_all = windows_all.squeeze(0)
            personal_info = personal_info.squeeze(0)
            target = target.squeeze(0)
            target = target.to(device)
            mini_batches = get_overlapping_batches(windows_all, inner_batch_size, inner_stride)
            sample_loss = 0.0
            for mini_batch in mini_batches:
                mini_batch = mini_batch.to(device)
                p_info = personal_info.to(device)
                _, _, bp_pred = model(mini_batch, p_info)
                bp_pred = bp_pred[0]
                loss = l1_loss_fn(bp_pred, target)
                sample_loss += loss.item()
                count += 1
            total_loss += sample_loss / len(mini_batches)
    return total_loss / len(dataloader)

#############################################
# 主訓練流程：每個 epoch 同時包含 Phase 1 與 Phase 2
#############################################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 若要比較 PPG 與 Tonometry，只需分別建立不同的 DataLoader，例如：
    # dataset_ppg = BPWholeDataset(h5_folder=H5_FOLDER, sensor='ppg')
    # dataset_tono = BPWholeDataset(h5_folder=H5_FOLDER, sensor='tonometry')
    # 這裡示範以 ppg 為例
    dataset = BPWholeDataset(h5_folder=H5_FOLDER, sensor='ppg')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    
    model = RepresentationBPModel(in_channels=2, representation_dim=32, base_channels=32).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"總參數量: {total_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 初始驗證
    init_val_loss = validate_epoch(model, dataloader, device)
    print(f"初始驗證回歸損失: {init_val_loss:.4f}")
    if np.isnan(init_val_loss) or init_val_loss == float('inf'):
        print("初始驗證損失為 nan 或 inf，請檢查資料與模型設定，提前停止訓練！")
        return
    
    num_epochs = 60
    best_val_loss = float('inf')
    best_epoch = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        cons_loss = train_consistency_epoch(model, dataloader, optimizer, device)
        reg_loss = train_regression_epoch(model, dataloader, optimizer, device)
        val_loss = validate_epoch(model, dataloader, device)
        print(f"  Consistency Loss: {cons_loss:.4f}, Regression Loss: {reg_loss:.4f}, Val Loss: {val_loss:.4f}")
        if np.isnan(val_loss) or val_loss == float('inf'):
            print("驗證損失出現 nan 或 inf，提前停止訓練！")
            break
        if val_loss < best_val_loss:
            best_val_loss = val_loss            
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_representation_model.pth')
            print(f"保存最佳模型 (epoch {best_epoch}, val_loss: {best_val_loss:.4f})")
    
    print(f"\n訓練完成！最佳模型在 epoch {best_epoch}，驗證損失: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
