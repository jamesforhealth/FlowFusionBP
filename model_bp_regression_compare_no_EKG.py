#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比較 PPG、Tonometry 與 Fusion (PPG+Tonometry) 三種方案對血壓預測的效能與穿戴訊號對於輸出預測解釋性的影響

採用兩種方法：
1. End-to-End 回歸模型：直接將一維訊號與個人資訊融合後進行血壓回歸
2. 基於對比表示學習的模型：先對信號進行對比學習，使嵌入空間反映血壓值分布遠近，再融合個人資訊做下游血壓回歸

假設每個 h5 檔案已包含 sliding window 訊號 (已正規化) 以及以下屬性：
  participant_age, participant_gender, participant_height, participant_weight,
  measurement_sbp, measurement_dbp
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from tqdm import tqdm
import shap
from sklearn.feature_selection import mutual_info_regression
from torch.amp import autocast, GradScaler

#############################################
# 參數設定
#############################################
WINDOW_LENGTH = 4096
H5_FOLDER = 'output_h5_folder'  # 請根據實際路徑調整
NUM_EPOCHS = 200
BATCH_SIZE = 128
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################################
# Dataset：BPRegressionDataset
#############################################
class BPRegressionDataset(Dataset):
    def __init__(self, h5_folder, input_mode="ppg", cache_size=1000, blacklist_file='h5_nan_blacklist.txt'):
        """
        優化資料集：讀取時過濾掉所有缺失值（NaN）的資料
        cache_size: 快取大小
        blacklist_file: 存儲包含 NaN 的檔案名的黑名單文件
        """
        self.h5_folder = h5_folder
        self.input_mode = input_mode.lower()
        self.samples = []  # 存放 (file, window_index)
        all_files = [os.path.join(h5_folder, f) for f in os.listdir(h5_folder) if f.endswith('.h5')]
        self.cache = {}  # 使用完整的索引作為快取鍵
        self.cache_size = cache_size

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
            raise ValueError("沒有找到有效的 h5 檔案！")
        
        self.file_indices = {}
        filtered_count = 0
        new_blacklist_entries = set()
        
        # 依序處理每個檔案
        for f in self.files:
            try:
                with h5py.File(f, 'r') as hf:
                    # 基本檢查：確認文件包含必要的數據集
                    if 'PPG' not in hf or 'Tonometry' not in hf:
                        print(f"檔案 {os.path.basename(f)} 缺少必要的數據集，加入黑名單")
                        new_blacklist_entries.add(f)
                        filtered_count += 1
                        continue
                    
                    # 讀取並檢查屬性是否存在且不為 nan
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
                            filtered_count += 1
                            continue
                        
                        # 檢查數值屬性是否為 NaN
                        if (np.isnan(float(attr_age)) or np.isnan(float(attr_height)) or
                            np.isnan(float(attr_weight)) or np.isnan(float(attr_sbp)) or
                            np.isnan(float(attr_dbp))):
                            print(f"檔案 {os.path.basename(f)} 屬性有 NaN 值，加入黑名單")
                            new_blacklist_entries.add(f)
                            filtered_count += 1
                            continue
                    except (ValueError, TypeError) as e:
                        print(f"檔案 {os.path.basename(f)} 屬性轉換錯誤: {e}，加入黑名單")
                        new_blacklist_entries.add(f)
                        filtered_count += 1
                        continue
                    
                    # 正常轉換屬性
                    age = float(attr_age) / 100.0
                    gender = 1.0 if str(attr_gender).strip().upper() == "M" else 0.0
                    height = float(attr_height) / 200.0
                    weight = float(attr_weight) / 150.0
                    sbp = float(attr_sbp) / 200.0
                    dbp = float(attr_dbp) / 200.0
                    
                    # 驗證血壓值的合理性
                    if sbp <= 0 or dbp <= 0 or sbp <= dbp:
                        print(f"檔案 {os.path.basename(f)} 血壓值不合理: SBP={attr_sbp}, DBP={attr_dbp}，加入黑名單")
                        new_blacklist_entries.add(f)
                        filtered_count += 1
                        continue

                    # 保存檔案層級 metadata
                    num_windows = hf['PPG'].shape[0]
                    self.file_indices[f] = {
                        'num_windows': num_windows,
                        'metadata': {
                            'personal_info': np.array([age, gender, height, weight], dtype=np.float32),
                            'target': np.array([sbp, dbp], dtype=np.float32)
                        }
                    }
                    
                    # 檢查窗口中的 NaN
                    has_nan_windows = False
                    for i in range(min(10, num_windows)):  # 只檢查前10個窗口以提高效率
                        if self.input_mode == "ppg":
                            signal = hf['PPG'][i]
                        elif self.input_mode == "tonometry":
                            signal = hf['Tonometry'][i]
                        elif self.input_mode == "fusion":
                            ppg = hf['PPG'][i]
                            tono = hf['Tonometry'][i]
                            if np.isnan(ppg).any() or np.isnan(tono).any():
                                has_nan_windows = True
                                break
                            continue
                        else:
                            raise ValueError("Unknown input_mode")
                        
                        if np.isnan(signal).any():
                            has_nan_windows = True
                            break
                    
                    if has_nan_windows:
                        print(f"檔案 {os.path.basename(f)} 信號數據包含 NaN，加入黑名單")
                        new_blacklist_entries.add(f)
                        filtered_count += 1
                        continue
                    
                    # 將所有窗口添加到樣本中
                    for i in range(num_windows):
                        self.samples.append((f, i))
            
            except (IOError, OSError, ValueError, KeyError, AttributeError) as e:
                print(f"處理檔案 {os.path.basename(f)} 時出錯: {str(e)}，加入黑名單")
                new_blacklist_entries.add(f)
                filtered_count += 1
        
        # 更新黑名單
        if new_blacklist_entries:
            self.blacklist.update(new_blacklist_entries)
            # 定期保存黑名單，防止中斷
            with open(blacklist_file, 'w') as f:
                for item in self.blacklist:
                    f.write(f"{item}\n")
            print(f"添加了 {len(new_blacklist_entries)} 個新的黑名單項目")
        
        print(f"過濾後總樣本數: {len(self.samples)}")
        print(f"總共過濾掉 {filtered_count} 個檔案，黑名單總數: {len(self.blacklist)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        try:
            f, win_idx = self.samples[idx]
            metadata = self.file_indices[f]['metadata']
            
            with h5py.File(f, 'r') as hf:
                try:
                    ppg = hf['PPG'][win_idx]
                    tono = hf['Tonometry'][win_idx]
                    
                    if self.input_mode == "ppg":
                        signal = np.expand_dims(ppg, axis=0)
                    elif self.input_mode == "tonometry":
                        signal = np.expand_dims(tono, axis=0)
                    elif self.input_mode == "fusion":
                        signal = np.stack([ppg, tono], axis=0)
                    else:
                        raise ValueError("Unknown input_mode")
                    
                    # 正規化信號
                    mean_val = np.mean(signal, axis=1, keepdims=True)
                    std_val = np.std(signal, axis=1, keepdims=True)
                    std_val[std_val < 1e-8] = 1e-8
                    signal = (signal - mean_val) / std_val
                    signal = signal.astype(np.float32)
                    
                    result = (
                        torch.tensor(signal),
                        torch.tensor(metadata['personal_info']),
                        torch.tensor(metadata['target'])
                    )
                    
                    # 快取結果
                    if len(self.cache) < self.cache_size:
                        self.cache[idx] = result
                    
                    return result
                
                except Exception as e:
                    print(f"讀取樣本 {f}:{win_idx} 時出錯: {str(e)}")
                    return self.__getitem__((idx + 1) % len(self))
        
        except Exception as e:
            print(f"獲取樣本 {idx} 時發生錯誤: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

#############################################
# Model 1：End-to-End 回歸模型
#############################################
class SignalEncoder(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super(SignalEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(base_channels, base_channels*2, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_channels*8),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

class PersonalInfoEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32):
        super(PersonalInfoEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class End2EndRegressionModel(nn.Module):
    def __init__(self, signal_in_channels, personal_input_dim=4, base_channels=32, rep_dim=128):
        super(End2EndRegressionModel, self).__init__()
        self.signal_encoder = SignalEncoder(signal_in_channels, base_channels)
        self.projection_head = nn.Linear(base_channels*8, rep_dim)
        self.personal_encoder = PersonalInfoEncoder(input_dim=personal_input_dim, hidden_dim=32)
        self.regression_head = nn.Sequential(
            nn.Linear(rep_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, signal, personal_info):
        feat = self.signal_encoder(signal)
        rep = self.projection_head(feat)
        rep_norm = F.normalize(rep, p=2, dim=1)
        personal_feat = self.personal_encoder(personal_info)
        combined = torch.cat([rep_norm, personal_feat], dim=1)
        pred = self.regression_head(combined)
        return pred

#############################################
# Model 2：基於對比表示學習的模型
#############################################
class ContrastiveBPModel(nn.Module):
    def __init__(self, signal_in_channels, personal_input_dim=4, base_channels=32, rep_dim=128):
        super(ContrastiveBPModel, self).__init__()
        self.signal_encoder = SignalEncoder(signal_in_channels, base_channels)
        self.projection_head = nn.Linear(base_channels*8, rep_dim)
        self.personal_encoder = PersonalInfoEncoder(input_dim=personal_input_dim, hidden_dim=32)
        self.regression_head = nn.Sequential(
            nn.Linear(rep_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, signal, personal_info):
        feat = self.signal_encoder(signal)
        rep = self.projection_head(feat)
        rep_norm = F.normalize(rep, p=2, dim=1)
        personal_feat = self.personal_encoder(personal_info)
        combined = torch.cat([rep_norm, personal_feat], dim=1)
        pred = self.regression_head(combined)
        return rep_norm, pred


#############################################
# Model 3：基於分支對比表示學習的模型
#############################################
class BranchContrastiveBPModel(nn.Module):
    def __init__(self, signal_in_channels, personal_input_dim=4, base_channels=32, rep_dim=128):
        super(BranchContrastiveBPModel, self).__init__()
        self.signal_encoder = SignalEncoder(signal_in_channels, base_channels)
        
        # 分別為SBP和DBP生成表示
        half_dim = rep_dim // 2
        self.sbp_projection = nn.Linear(base_channels*8, half_dim)
        self.dbp_projection = nn.Linear(base_channels*8, half_dim)
        
        self.personal_encoder = PersonalInfoEncoder(input_dim=personal_input_dim, hidden_dim=32)
        
        # 分別為SBP和DBP建立回歸預測頭
        self.sbp_regression = nn.Sequential(
            nn.Linear(half_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.dbp_regression = nn.Sequential(
            nn.Linear(half_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, signal, personal_info):
        feat = self.signal_encoder(signal)
        
        # 生成SBP和DBP的表示
        sbp_rep = self.sbp_projection(feat)
        dbp_rep = self.dbp_projection(feat)
        
        # 正規化表示
        sbp_rep_norm = F.normalize(sbp_rep, p=2, dim=1)
        dbp_rep_norm = F.normalize(dbp_rep, p=2, dim=1)
        
        # 個人特徵編碼
        personal_feat = self.personal_encoder(personal_info)
        
        # 分別預測SBP和DBP
        sbp_combined = torch.cat([sbp_rep_norm, personal_feat], dim=1)
        dbp_combined = torch.cat([dbp_rep_norm, personal_feat], dim=1)
        
        sbp_pred = self.sbp_regression(sbp_combined)
        dbp_pred = self.dbp_regression(dbp_combined)
        
        # 連接SBP和DBP預測
        pred = torch.cat([sbp_pred, dbp_pred], dim=1)
        
        return (sbp_rep_norm, dbp_rep_norm), pred


#############################################
# Contrastive Loss 函數
#############################################
def contrastive_loss(representations, targets, tau=0.1):
    # tau 為浮點數
    B = representations.size(0)
    sbp = targets[:, 0].unsqueeze(1)
    dbp = targets[:, 1].unsqueeze(1)
    diff_sbp = torch.abs(sbp - sbp.t())
    diff_dbp = torch.abs(dbp - dbp.t())
    bp_diff = (diff_sbp + diff_dbp) / 2.0
    target_sim = torch.exp(-bp_diff / tau)
    sim_matrix = representations @ representations.t()  # cosine similarity (已正規化) 
    loss = F.mse_loss(sim_matrix, target_sim)
    return loss

#############################################
# Training 與 Validation 函數 (End-to-End)
#############################################
def train_epoch_end2end(model, dataloader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    pbar = tqdm(dataloader, desc="End2End Training", miniters=max(1, len(dataloader)//100))
    
    for signal, personal, target in pbar:
        signal = signal.to(device, non_blocking=True)
        personal = personal.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None and scaler.is_enabled():
            with autocast(device_type="cuda"):
                pred = model(signal, personal)
                loss = criterion(pred, target)
        else:
            pred = model(signal, personal)
            loss = criterion(pred, target)
        
        if torch.isnan(loss):
            print("檢測到 loss 為 nan")
            print("signal min/max:", signal.min().item(), signal.max().item())
            print("pred:", pred)
            print("target:", target)
            raise ValueError("Loss 為 nan，終止訓練")
        
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        total_loss += loss.item() * signal.size(0)
    return total_loss / len(dataloader.dataset)

def validate_epoch_end2end(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for signal, personal, target in tqdm(dataloader, desc="End2End Validation"):
            signal = signal.to(device)
            personal = personal.to(device)
            target = target.to(device)
            pred = model(signal, personal)
            loss = criterion(pred, target)
            total_loss += loss.item() * signal.size(0)
    return total_loss / len(dataloader.dataset)

#############################################
# Training 與 Validation 函數 (Contrastive)
#############################################
def train_epoch_contrastive(model, dataloader, optimizer, device, lambda_reg=0.5, tau=0.1):
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_contr_loss = 0.0
    mse_loss_fn = nn.MSELoss()
    
    for signal, personal, target in tqdm(dataloader, desc="Contrastive Training"):
        signal = signal.to(device)
        personal = personal.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        
        rep, pred = model(signal, personal)
        
        reg_loss = mse_loss_fn(pred, target)
        contr_loss = contrastive_loss(rep, target, tau)
        
        # 確保 lambda_reg 為數值
        if isinstance(lambda_reg, GradScaler):
            raise ValueError("lambda_reg 應該是一個數值，而不是 GradScaler 實例")
        
        loss = lambda_reg * reg_loss + (1 - lambda_reg) * contr_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * signal.size(0)
        total_reg_loss += reg_loss.item() * signal.size(0)
        total_contr_loss += contr_loss.item() * signal.size(0)
    return total_loss / len(dataloader.dataset), total_reg_loss / len(dataloader.dataset), total_contr_loss / len(dataloader.dataset)

def validate_epoch_contrastive(model, dataloader, device, lambda_reg=0.5, tau=0.1):
    model.eval()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_contr_loss = 0.0
    mse_loss_fn = nn.MSELoss()
    with torch.no_grad():
        for signal, personal, target in tqdm(dataloader, desc="Contrastive Validation"):
            signal = signal.to(device)
            personal = personal.to(device)
            target = target.to(device)
            rep, pred = model(signal, personal)
            reg_loss = mse_loss_fn(pred, target)
            contr_loss = contrastive_loss(rep, target, tau)
            loss = lambda_reg * reg_loss + (1 - lambda_reg) * contr_loss
            total_loss += loss.item() * signal.size(0)
            total_reg_loss += reg_loss.item() * signal.size(0)
            total_contr_loss += contr_loss.item() * signal.size(0)
    return total_loss / len(dataloader.dataset), total_reg_loss / len(dataloader.dataset), total_contr_loss / len(dataloader.dataset)

#############################################
# SHAP 與 Mutual Information 分析函數
#############################################
def compute_shap(model, dataloader, device, num_samples=5, model_type="end2end"):
    background_signals, background_personal = [], []
    samples_signals, samples_personal = [], []
    for i, (signal, personal, target) in enumerate(dataloader):
        if i < num_samples:
            background_signals.append(signal)
            background_personal.append(personal)
        else:
            samples_signals.append(signal)
            samples_personal.append(personal)
        if i >= num_samples * 2:
            break
    background_signals = torch.cat(background_signals, dim=0).to(device)
    background_personal = torch.cat(background_personal, dim=0).to(device)
    samples_signals = torch.cat(samples_signals, dim=0).to(device)
    samples_personal = torch.cat(samples_personal, dim=0).to(device)
    
    def concat_input(signal, personal):
        batch, C, L = signal.shape
        return torch.cat([signal.view(batch, -1), personal], dim=1)
    
    background = concat_input(background_signals, background_personal)
    samples = concat_input(samples_signals, samples_personal)
    
    if model_type == "end2end":
        class WrapperModel(nn.Module):
            def __init__(self, base_model):
                super(WrapperModel, self).__init__()
                self.base_model = base_model
                self.signal_in_channels = base_model.signal_encoder.net[0].in_channels
                self.window_length = WINDOW_LENGTH
            def forward(self, x):
                batch = x.shape[0]
                C = self.signal_in_channels
                signal_flat = x[:, :C * self.window_length]
                personal = x[:, C * self.window_length:]
                signal = signal_flat.view(batch, C, self.window_length)
                return self.base_model(signal, personal)
        wrapper = WrapperModel(model).to(device)
    else:
        class WrapperModel(nn.Module):
            def __init__(self, base_model):
                super(WrapperModel, self).__init__()
                self.base_model = base_model
                self.signal_in_channels = base_model.signal_encoder.net[0].in_channels
                self.window_length = WINDOW_LENGTH
            def forward(self, x):
                batch = x.shape[0]
                C = self.signal_in_channels
                signal_flat = x[:, :C * self.window_length]
                personal = x[:, C * self.window_length:]
                signal = signal_flat.view(batch, C, self.window_length)
                _, out = self.base_model(signal, personal)
                return out
        wrapper = WrapperModel(model).to(device)
    
    explainer = shap.DeepExplainer(wrapper, background)
    shap_values = explainer.shap_values(samples)
    return shap_values

def compute_mutual_info(model, dataloader, device, model_type="end2end"):
    model.eval()
    features, targets = [], []
    with torch.no_grad():
        for signal, personal, target in dataloader:
            signal = signal.to(device)
            personal = personal.to(device)
            if model_type == "end2end":
                feat = model.signal_encoder(signal)
            else:
                feat = model.signal_encoder(signal)
            feat_mean = feat.mean(dim=0).cpu().numpy()
            features.append(feat_mean)
            targets.append(target[0].cpu().numpy())
    features = np.vstack(features)
    targets = np.vstack(targets)
    mi = []
    for i in range(targets.shape[1]):
        mi_val = mutual_info_regression(features, targets[:, i])
        mi.append(mi_val)
    return mi

#############################################
# 主訓練流程
#############################################
def main():
    input_modes = ["ppg"]#["ppg", "tonometry", "fusion"]
    methods = {
        # "End2End": (End2EndRegressionModel, train_epoch_end2end, validate_epoch_end2end, "end2end"),
        "Contrastive": (ContrastiveBPModel, train_epoch_contrastive, validate_epoch_contrastive, "contrastive")
    }
    
    num_workers = min(16, os.cpu_count())
    pin_memory = torch.cuda.is_available()
    
    use_subset = True
    subset_size = 50000
    
    # 暫時可以嘗試停用 AMP 以排除混合精度引起的問題
    use_amp = torch.cuda.is_available()  # 如有疑慮，可改成 False
    
    results = {}
    for method_name, (ModelClass, train_fn, val_fn, model_type) in methods.items():
        print(f"\n=== 使用 {method_name} 方法進行訓練 ===")
        results[method_name] = {}
        
        for mode in input_modes:
            print(f"\n--- 訓練模型：輸入組合 {mode.upper()} ---")
            
            # 加載完整數據集
            full_dataset = BPRegressionDataset(h5_folder=H5_FOLDER, input_mode=mode, cache_size=5000)
            print(f"完整資料集大小: {len(full_dataset)}")
            
            # 使用子集（如果需要）
            if use_subset and len(full_dataset) > subset_size:
                # 使用固定種子以確保可重複性
                torch.manual_seed(42)
                indices = torch.randperm(len(full_dataset))[:subset_size]
                subset_dataset = Subset(full_dataset, indices)
                print(f"使用子集進行訓練, 大小: {len(subset_dataset)}")
            else:
                subset_dataset = full_dataset
                
            # 按照 7:2:1 切分為訓練集、驗證集和測試集
            dataset_size = len(subset_dataset)
            train_size = int(dataset_size * 0.7)
            val_size = int(dataset_size * 0.2)
            test_size = dataset_size - train_size - val_size
            
            # 使用 random_split 函數切分數據集
            train_dataset, val_dataset, test_dataset = random_split(
                subset_dataset, 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)  # 確保可重複性
            )
            
            print(f"訓練集大小: {len(train_dataset)}")
            print(f"驗證集大小: {len(val_dataset)}")
            print(f"測試集大小: {len(test_dataset)}")
            
            # 創建數據加載器
            train_loader = DataLoader(
                train_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=4 if num_workers > 0 else None
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False,  # 驗證集不需要打亂
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=4 if num_workers > 0 else None
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False,  # 測試集不需要打亂
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=4 if num_workers > 0 else None
            )
            
            print(f"訓練批次數量: {len(train_loader)}")
            print(f"驗證批次數量: {len(val_loader)}")
            print(f"測試批次數量: {len(test_loader)}")
            
            torch.backends.cudnn.benchmark = True
            
            in_channels = 1 if mode in ["ppg", "tonometry"] else 2
            model = ModelClass(signal_in_channels=in_channels, personal_input_dim=4, base_channels=32).to(DEVICE)
            model = model.float()
                
            print(f"模型參數數量: {sum(p.numel() for p in model.parameters())}")
            
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            
            scaler = GradScaler(enabled=use_amp)
            
            # 使用驗證集評估模型
            best_val_loss = float('inf')
            for epoch in range(NUM_EPOCHS):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 利用 lower() 確保比較一致，對比方法不傳入 scaler
                reg_loss = 0.0
                contr_loss = 0.0
                val_reg_loss = 0.0
                val_contr_loss = 0.0
                if model_type.lower() == "contrastive":
                    train_loss, reg_loss, contr_loss = train_fn(model, train_loader, optimizer, DEVICE)
                else:
                    train_loss = train_fn(model, train_loader, optimizer, DEVICE, scaler)
                
                if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == NUM_EPOCHS - 1:
                    if model_type.lower() == "contrastive":
                        val_loss, val_reg_loss, val_contr_loss = val_fn(model, val_loader, DEVICE)
                    else:
                        val_loss = val_fn(model, val_loader, DEVICE)
                    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss: {train_loss:.4f} | Reg Loss: {reg_loss:.4f} | Contr Loss: {contr_loss:.4f}")
                    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Val Loss: {val_loss:.4f} | Reg Loss: {val_reg_loss:.4f} | Contr Loss: {val_contr_loss:.4f}")
                    scheduler.step(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
                        torch.save(model_cpu, f'best_compare_model_{method_name}_{mode}_200.pth')
                        print(f"保存最佳模型: {f'best_compare_model_{method_name}_{mode}_200.pth'}")
                else:
                    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss: {train_loss:.4f}")
            
            results[method_name][mode] = best_val_loss
            print(f"{mode.upper()} 模型最佳驗證損失: {best_val_loss:.4f}")
            
            # SHAP 與 Mutual Information 分析（如需要）
            try:
                small_loader = DataLoader(
                    Subset(subset_dataset, torch.randperm(len(subset_dataset))[:min(500, len(subset_dataset))]),
                    batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory
                )
                shap_vals = compute_shap(model, small_loader, DEVICE, model_type=model_type)
                print(f"{mode.upper()} 模型 SHAP 值: mean={np.mean(shap_vals[0]):.4f}, std={np.std(shap_vals[0]):.4f}")
                
                mi = compute_mutual_info(model, small_loader, DEVICE, model_type=model_type)
                print(f"{mode.upper()} 模型 Mutual Information: {mi}")
            except Exception as e:
                print(f"{mode.upper()} 模型分析時出錯: {e}")
            
            # 最終在測試集上評估
            print("\n--- 在測試集上評估最佳模型 ---")
            # 加載最佳模型
            best_model = ModelClass(signal_in_channels=in_channels, personal_input_dim=4, base_channels=32).to(DEVICE)
            best_model.load_state_dict(torch.load(f'best_compare_model_{method_name}_{mode}_200.pth'))
            
            # 在測試集上評估
            if model_type.lower() == "contrastive":
                test_loss, test_reg_loss, test_contr_loss = val_fn(best_model, test_loader, DEVICE)
                print(f"測試損失: {test_loss:.6f} | Reg Loss: {test_reg_loss:.6f} | Contr Loss: {test_contr_loss:.6f}")
                
                # 計算SBP和DBP的平均絕對誤差
                sbp_mae, dbp_mae = evaluate_bp_error(best_model, test_loader, DEVICE)
                print(f"SBP 平均絕對誤差: {sbp_mae:.2f} mmHg")
                print(f"DBP 平均絕對誤差: {dbp_mae:.2f} mmHg")
            else:
                test_loss = val_fn(best_model, test_loader, DEVICE)
                print(f"測試損失: {test_loss:.6f}")
                
                # 計算SBP和DBP的平均絕對誤差
                sbp_mae, dbp_mae = evaluate_bp_error(best_model, test_loader, DEVICE)
                print(f"SBP 平均絕對誤差: {sbp_mae:.2f} mmHg")
                print(f"DBP 平均絕對誤差: {dbp_mae:.2f} mmHg")
            
            print("\n----------------------------\n")
            
# 添加評估實際血壓誤差的函數
def evaluate_bp_error(model, dataloader, device):
    model.eval()
    sbp_errors = []
    dbp_errors = []
    
    with torch.no_grad():
        for signal, personal, target in tqdm(dataloader, desc="評估血壓誤差"):
            signal = signal.to(device)
            personal = personal.to(device)
            target = target.to(device)
            
            # 根據模型類型獲取預測值
            if isinstance(model, ContrastiveBPModel):
                _, pred = model(signal, personal)
            else:
                pred = model(signal, personal)
            
            # 將標準化的預測值轉換回原始血壓值
            pred_sbp = pred[:, 0] * 200.0  # SBP 預測值
            pred_dbp = pred[:, 1] * 200.0  # DBP 預測值
            true_sbp = target[:, 0] * 200.0  # SBP 真實值
            true_dbp = target[:, 1] * 200.0  # DBP 真實值
            
            # 計算絕對誤差
            sbp_error = torch.abs(pred_sbp - true_sbp)
            dbp_error = torch.abs(pred_dbp - true_dbp)
            
            sbp_errors.extend(sbp_error.cpu().numpy())
            dbp_errors.extend(dbp_error.cpu().numpy())
    
    # 計算平均絕對誤差
    sbp_mae = np.mean(sbp_errors)
    dbp_mae = np.mean(dbp_errors)
    
    return sbp_mae, dbp_mae

if __name__ == '__main__':
    main()
