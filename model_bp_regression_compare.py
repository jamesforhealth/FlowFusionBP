#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比較 PPG、Tonometry、EKG 與 Fusion (PPG+Tonometry+EKG) 三種方案對血壓預測的效能與穿戴訊號對於輸出預測解釋性的影響

採用兩種方法：
1. End-to-End 回歸模型：直接將一維訊號與個人資訊融合後進行血壓回歸
2. 基於對比表示學習的模型：先對信號進行對比學習，使嵌入空間反映血壓值分布遠近，再融合個人資訊做下游血壓回歸

假設每個 h5 檔案已包含 sliding window 訊號 (已正規化) 以及以下屬性：
  participant_age, participant_gender, participant_height, participant_weight,
  measurement_sbp, measurement_dbp
  
且新增包含 "EKG" 資料（若有）的資料集
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
# 若未來使用 PulseDB 125Hz，可能需要重新調整 WINDOW_LENGTH（例如根據秒數設定）
WINDOW_LENGTH = 4096  
H5_FOLDER = 'output_h5_folder'  # 請根據實際路徑調整
NUM_EPOCHS = 50
BATCH_SIZE = 1024#256#128
LR = 5e-4#1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################################
# Dataset：BPRegressionDataset
#############################################
import pickle

INDEX_FILENAME = "h5_index.pkl"  # 索引文件名

def scan_file(f, req_keys, num_window_check=10):
    """
    掃描單一文件，檢查必要的數據集、屬性與部分窗口數據是否有問題。
    若文件有效，返回一個字典 { 'file': f, 'num_windows': num_windows, 'metadata': { ... } }
    否則返回 None。
    """
    try:
        with h5py.File(f, 'r') as hf:
            # 檢查必須的數據集是否存在
            for key in req_keys:
                if key not in hf:
                    # 缺少必要數據集
                    return None
            # 檢查屬性是否存在
            attrs = {}
            for k in ['participant_age', 'participant_gender', 'participant_height', 
                      'participant_weight', 'measurement_sbp', 'measurement_dbp']:
                val = hf.attrs.get(k, None)
                if val is None:
                    return None
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    return None
                # 檢查數值是否 nan
                if np.isnan(val):
                    return None
                attrs[k] = val
            
            # 驗證血壓合理性
            sbp, dbp = attrs['measurement_sbp'], attrs['measurement_dbp']
            if sbp <= 0 or dbp <= 0 or sbp <= dbp:
                return None
            
            # 正規化屬性
            metadata = {
                'personal_info': np.array([
                    attrs['participant_age'] / 100.0,
                    1.0 if str(hf.attrs.get('participant_gender')).strip().upper() == "M" else 0.0,
                    attrs['participant_height'] / 200.0,
                    attrs['participant_weight'] / 150.0,
                ], dtype=np.float32),
                'target': np.array([sbp / 200.0, dbp / 200.0], dtype=np.float32)
            }
            num_windows = hf['PPG'].shape[0]
            # 檢查前 num_window_check 個窗口是否含有 NaN（根據 input_mode 決定要讀取的數據集）
            for i in range(min(num_window_check, num_windows)):
                if "fusion_ekg" in req_keys:
                    ppg = hf['PPG'][i]
                    tono = hf['Tonometry'][i]
                    ekg = hf['EKG'][i]
                    # 检查是否有NaN值或极端值
                    if (np.isnan(ppg).any() or np.isnan(tono).any() or np.isnan(ekg).any() or
                        np.abs(ppg).max() > 10000 or np.abs(tono).max() > 10000 or np.abs(ekg).max() > 10000):
                        return None
                elif req_keys == ["EKG"]:
                    sig = hf['EKG'][i]
                    if np.isnan(sig).any():
                        return None
                elif req_keys == ["PPG"]:
                    sig = hf['PPG'][i]
                    if np.isnan(sig).any():
                        return None
                elif req_keys == ["Tonometry"]:
                    sig = hf['Tonometry'][i]
                    if np.isnan(sig).any():
                        return None
                elif req_keys == ["PPG", "Tonometry"]:
                    ppg = hf['PPG'][i]
                    tono = hf['Tonometry'][i]
                    if np.isnan(ppg).any() or np.isnan(tono).any():
                        return None
                elif req_keys == ["EKG", "PPG"]:
                    ekg = hf['EKG'][i]
                    ppg = hf['PPG'][i]
                    if (np.isnan(ekg).any() or np.isnan(ppg).any() or
                        np.abs(ekg).max() > 10000 or np.abs(ppg).max() > 10000):
                        return None
            return {'file': f, 'num_windows': num_windows, 'metadata': metadata}
    except Exception as e:
        return None

class BPRegressionDataset(Dataset):
    def __init__(self, h5_folder, input_mode="ppg", cache_size=1000, 
                 blacklist_file='h5_nan_blacklist.txt', num_window_check=None):
        """
        優化資料集：利用預先生成的索引文件快速讀取元數據，
        並支援 "ppg"、"tonometry"、"fusion"、"fusion_ekg"、"ekg" 模式。
        """
        self.h5_folder = h5_folder
        self.input_mode = input_mode.lower()
        self.cache = {}
        self.cache_size = cache_size

        # 讀取黑名單（已排除有問題的文件）
        self.blacklist = set()
        if os.path.exists(blacklist_file):
            with open(blacklist_file, 'r') as f:
                self.blacklist = set(line.strip() for line in f)
            print(f"已從 {blacklist_file} 讀取 {len(self.blacklist)} 個黑名單文件")

        # 所有文件（排除黑名單中的）
        all_files = [os.path.join(h5_folder, f) for f in os.listdir(h5_folder) if f.endswith('.h5')]
        self.files = [f for f in all_files if f not in self.blacklist]
        print(f"總文件數: {len(all_files)}, 過濾後: {len(self.files)}")
        if len(self.files) == 0:
            raise ValueError("沒有找到有效的 h5 檔案！")
        
        # 根據模式決定必須的數據集
        def required_keys(mode):
            if mode == "ppg":
                return ["PPG"]
            elif mode == "tonometry":
                return ["Tonometry"]
            elif mode == "fusion":
                return ["PPG", "Tonometry"]
            elif mode == "fusion_ekg":
                return ["PPG", "Tonometry", "EKG"]
            elif mode == "ekg":
                return ["EKG"]
            elif mode == "ekg_ppg":
                return ["EKG", "PPG"]
            else:
                raise ValueError("Unknown input_mode")
        self.req_keys = required_keys(self.input_mode)

        # 決定檢查窗口數量
        if num_window_check is None:
            if self.input_mode in ["fusion_ekg", "ekg"]:
                self.num_window_check = 1
            else:
                self.num_window_check = 10
        else:
            self.num_window_check = num_window_check

        # 優先嘗試讀取索引文件
        index_path = os.path.join(self.h5_folder, INDEX_FILENAME)
        if os.path.exists(index_path):
            print("載入預先生成的索引文件...")
            with open(index_path, 'rb') as f:
                self.file_indices = pickle.load(f)
        else:
            print("索引文件不存在，開始掃描所有文件...")
            self.file_indices = {}
            # 使用 tqdm 顯示進度
            from concurrent.futures import ProcessPoolExecutor, as_completed
            futures = []
            with ProcessPoolExecutor() as executor:
                for f in self.files:
                    futures.append(executor.submit(scan_file, f, self.req_keys, self.num_window_check))
                for future in tqdm(as_completed(futures), total=len(futures), desc="生成索引"):
                    result = future.result()
                    if result is not None:
                        fpath = result['file']
                        self.file_indices[fpath] = result
            # 保存索引文件，下次直接讀取
            with open(index_path, 'wb') as f:
                pickle.dump(self.file_indices, f)
            print("索引文件生成完成！")

        # 建立 samples：僅採用在索引中有記錄的文件
        self.samples = []
        for f, info in self.file_indices.items():
            # f 必須在 self.files（且不在黑名單中）
            num_windows = info['num_windows']
            for i in range(num_windows):
                self.samples.append((f, i))
        print(f"總樣本數: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        # 最多尝试遍历数据集一遍
        max_attempts = min(len(self.samples), 1000)  # 设置一个合理的上限
        attempts = 0
        
        current_idx = idx
        while attempts < max_attempts:
            attempts += 1
            try:
                f, win_idx = self.samples[current_idx]
                metadata = self.file_indices[f]['metadata']
                with h5py.File(f, 'r') as hf:
                    try:
                        ppg = hf['PPG'][win_idx] if "PPG" in hf else None
                        tono = hf['Tonometry'][win_idx] if "Tonometry" in hf else None
                        ekg = hf['EKG'][win_idx] if "EKG" in hf else None
                        
                        # 确保根据输入模式检查必要的数据是否存在
                        if self.input_mode == "ppg" and ppg is None:
                            raise ValueError("PPG数据不存在")
                        elif self.input_mode == "tonometry" and tono is None:
                            raise ValueError("Tonometry数据不存在")
                        elif self.input_mode == "ekg" and ekg is None:
                            raise ValueError("EKG数据不存在")
                        elif self.input_mode == "ekg_ppg" and (ekg is None or ppg is None):
                            raise ValueError("EKG或PPG数据不存在")
                        
                        # 原有的处理逻辑...
                        if self.input_mode == "ppg":
                            signal = np.expand_dims(ppg, axis=0)
                        elif self.input_mode == "tonometry":
                            signal = np.expand_dims(tono, axis=0)
                        elif self.input_mode == "fusion":
                            signal = np.stack([ppg, tono], axis=0)
                        elif self.input_mode == "fusion_ekg":
                            signal = np.stack([ppg, tono, ekg], axis=0)
                        elif self.input_mode == "ekg":
                            signal = np.expand_dims(ekg, axis=0)
                        elif self.input_mode == "ekg_ppg":
                            signal = np.stack([ekg, ppg], axis=0)
                        else:
                            raise ValueError("Unknown input_mode")
                        
                        # 对信号进行标准化处理，防止极端值
                        for c in range(signal.shape[0]):
                            # 使用稳健的归一化方法
                            chan = signal[c]
                            # 方法1：基于均值和标准差的标准化
                            mean_val = np.mean(chan)
                            std_val = np.std(chan)
                            if std_val < 1e-8:
                                std_val = 1e-8
                            signal[c] = (chan - mean_val) / std_val
                            
                            # 额外的安全检查：将超出范围的值裁剪到有效范围
                            signal[c] = np.clip(signal[c], -10.0, 10.0)
                        
                        # 转换为float32类型
                        signal = signal.astype(np.float32)
                        
                        # 成功找到有效样本，返回结果
                        result = (
                            torch.tensor(signal, dtype=torch.float32),
                            torch.tensor(metadata['personal_info'], dtype=torch.float32),
                            torch.tensor(metadata['target'], dtype=torch.float32)
                        )
                        if len(self.cache) < self.cache_size:
                            self.cache[idx] = result
                        return result
                        
                    except Exception as e:
                        print(f"读取样本 {f}:{win_idx} 时出错: {str(e)}")
                        current_idx = (current_idx + 1) % len(self.samples)
                        continue
            except Exception as e:
                print(f"获取样本 {current_idx} 时发生错误: {str(e)}")
                current_idx = (current_idx + 1) % len(self.samples)
                continue
        
        # 如果尝试了足够多次仍未找到有效样本，则返回一个默认值或抛出异常
        raise RuntimeError(f"尝试了{max_attempts}次后仍无法获取有效样本")

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
    B = representations.size(0)
    sbp = targets[:, 0].unsqueeze(1)
    dbp = targets[:, 1].unsqueeze(1)
    diff_sbp = torch.abs(sbp - sbp.t())
    diff_dbp = torch.abs(dbp - dbp.t())
    bp_diff = (diff_sbp + diff_dbp) / 2.0
    target_sim = torch.exp(-bp_diff / tau)
    sim_matrix = representations @ representations.t()
    loss = F.mse_loss(sim_matrix, target_sim)
    return loss

def branch_contrastive_loss(representations, targets, tau=0.1):
    """
    計算分支對比損失
    
    Args:
        representations: 元組 (sbp_rep, dbp_rep)，其中每個元素是形狀為[B, D]的張量
        targets: 形狀為[B, 2]的張量，包含SBP和DBP值
        tau: 相似度標度因子
    
    Returns:
        sbp對比損失和dbp對比損失的總和
    """
    sbp_rep, dbp_rep = representations
    B = sbp_rep.size(0)
    
    # 提取SBP和DBP目標
    sbp = targets[:, 0].unsqueeze(1)
    dbp = targets[:, 1].unsqueeze(1)
    
    # 計算SBP的對比損失
    diff_sbp = torch.abs(sbp - sbp.t())
    sbp_target_sim = torch.exp(-diff_sbp / tau)
    sbp_sim_matrix = sbp_rep @ sbp_rep.t()
    sbp_loss = F.mse_loss(sbp_sim_matrix, sbp_target_sim)
    
    # 計算DBP的對比損失
    diff_dbp = torch.abs(dbp - dbp.t())
    dbp_target_sim = torch.exp(-diff_dbp / tau)
    dbp_sim_matrix = dbp_rep @ dbp_rep.t()
    dbp_loss = F.mse_loss(dbp_sim_matrix, dbp_target_sim)
    
    return sbp_loss + dbp_loss
#############################################
# Training 與 Validation 函數 (End-to-End)
#############################################
def train_epoch_end2end(model, dataloader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    pbar = tqdm(dataloader, desc="End2End Training", miniters=max(1, len(dataloader)//100))
    
    for signal, personal, target in pbar:
        # 确保数据类型一致性
        signal = signal.to(device, dtype=torch.float32, non_blocking=True)
        personal = personal.to(device, dtype=torch.float32, non_blocking=True)
        target = target.to(device, dtype=torch.float32, non_blocking=True)
        
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
def train_epoch_contrastive(model, dataloader, optimizer, device, scaler=None, lambda_reg=0.5, tau=0.1):
    """
    训练一个epoch的对比学习模型
    
    参数:
        scaler: GradScaler对象，用于混合精度训练
        lambda_reg: 回归损失的权重系数
        tau: 对比损失的温度参数
    """
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_contr_loss = 0.0
    mse_loss_fn = nn.MSELoss()
    pbar = tqdm(dataloader, desc="Contrastive Training", miniters=max(1, len(dataloader)//100))
    
    for signal, personal, target in pbar:
        # 检查信号数据是否存在极端值
        if torch.isnan(signal).any() or torch.isinf(signal).any() or \
           signal.abs().max() > 1000:  # 设置一个合理的阈值
            print(f"跳过包含异常值的批次，信号范围: {signal.min().item()} - {signal.max().item()}")
            continue
            
        signal = signal.to(device, dtype=torch.float32, non_blocking=True)
        personal = personal.to(device, dtype=torch.float32, non_blocking=True)
        target = target.to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None and scaler.is_enabled():
            with autocast(device_type="cuda"):
                rep, pred = model(signal, personal)
                reg_loss = mse_loss_fn(pred, target)
                contr_loss = contrastive_loss(rep, target, tau)
                loss = lambda_reg * reg_loss + (1 - lambda_reg) * contr_loss
        else:
            rep, pred = model(signal, personal)
            reg_loss = mse_loss_fn(pred, target)
            contr_loss = contrastive_loss(rep, target, tau)
            loss = lambda_reg * reg_loss + (1 - lambda_reg) * contr_loss
        
        if torch.isnan(loss):
            print("检测到 loss 为 nan")
            print("signal min/max:", signal.min().item(), signal.max().item())
            print("pred:", pred)
            print("target:", target)
            raise ValueError("Loss 为 nan，终止训练")
            
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

def train_epoch_contrastive_branch(model, dataloader, optimizer, device, scaler=None, lambda_reg=0.5, tau=0.1):
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_contr_loss = 0.0
    mse_loss_fn = nn.MSELoss()
    pbar = tqdm(dataloader, desc="分支對比學習訓練", miniters=max(1, len(dataloader)//100))
    
    for signal, personal, target in pbar:
        # 检查信号数据是否存在极端值
        if torch.isnan(signal).any() or torch.isinf(signal).any() or \
           signal.abs().max() > 1000:  # 设置一个合理的阈值
            print(f"跳过包含异常值的批次，信号范围: {signal.min().item()} - {signal.max().item()}")
            continue
            
        signal = signal.to(device, dtype=torch.float32, non_blocking=True)
        personal = personal.to(device, dtype=torch.float32, non_blocking=True)
        target = target.to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None and scaler.is_enabled():
            with autocast(device_type="cuda"):
                reps, pred = model(signal, personal)
                reg_loss = mse_loss_fn(pred, target)
                contr_loss = branch_contrastive_loss(reps, target, tau)
                loss = lambda_reg * reg_loss + (1 - lambda_reg) * contr_loss
        else:
            reps, pred = model(signal, personal)
            reg_loss = mse_loss_fn(pred, target)
            contr_loss = branch_contrastive_loss(reps, target, tau)
            loss = lambda_reg * reg_loss + (1 - lambda_reg) * contr_loss
        
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
        total_reg_loss += reg_loss.item() * signal.size(0)
        total_contr_loss += contr_loss.item() * signal.size(0)
    
    return total_loss / len(dataloader.dataset), total_reg_loss / len(dataloader.dataset), total_contr_loss / len(dataloader.dataset)

def validate_epoch_contrastive_branch(model, dataloader, device, lambda_reg=0.5, tau=0.1):
    model.eval()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_contr_loss = 0.0
    mse_loss_fn = nn.MSELoss()
    
    with torch.no_grad():
        for signal, personal, target in tqdm(dataloader, desc="分支對比學習驗證"):
            signal = signal.to(device)
            personal = personal.to(device)
            target = target.to(device)
            
            reps, pred = model(signal, personal)
            
            reg_loss = mse_loss_fn(pred, target)
            contr_loss = branch_contrastive_loss(reps, target, tau)
            
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
    # 這邊以 Fusion + EKG 為例（融合 PPG、Tonometry 與 EKG）
    input_modes = ["fusion_ekg"]  # 如果 PulseDB，只能用 "ppg"
    methods = {
        # "End2End": (End2EndRegressionModel, train_epoch_end2end, validate_epoch_end2end, "end2end"),
        # "Contrastive": (ContrastiveBPModel, train_epoch_contrastive, validate_epoch_contrastive, "contrastive"),
        "BranchContrastive": (BranchContrastiveBPModel, train_epoch_contrastive_branch, validate_epoch_contrastive_branch, "branch_contrastive")
    }
    
    num_workers = min(16, os.cpu_count())
    pin_memory = torch.cuda.is_available()
    
    use_subset = True
    subset_size = 1000000  # 根據實際資料數量調整
    
    # AMP 可根据需要启用或停用
    use_amp = False  # 改为False，禁用混合精度训练
    
    results = {}
    for method_name, (ModelClass, train_fn, val_fn, model_type) in methods.items():
        print(f"\n=== 使用 {method_name} 方法進行訓練 ===")
        results[method_name] = {}
        
        for mode in input_modes:
            print(f"\n--- 訓練模型：輸入組合 {mode.upper()} ---")
            
            full_dataset = BPRegressionDataset(h5_folder=H5_FOLDER, input_mode=mode, cache_size=5000)
            print(f"完整資料集大小: {len(full_dataset)}")
            
            if use_subset and len(full_dataset) > subset_size:
                torch.manual_seed(42)
                indices = torch.randperm(len(full_dataset))[:subset_size]
                subset_dataset = Subset(full_dataset, indices)
                print(f"使用子集進行訓練, 大小: {len(subset_dataset)}")
            else:
                subset_dataset = full_dataset
                
            dataset_size = len(subset_dataset)
            train_size = int(dataset_size * 0.7)
            val_size = int(dataset_size * 0.2)
            test_size = dataset_size - train_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(
                subset_dataset, 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            print(f"訓練集大小: {len(train_dataset)}")
            print(f"驗證集大小: {len(val_dataset)}")
            print(f"測試集大小: {len(test_dataset)}")
            
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
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=4 if num_workers > 0 else None
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=4 if num_workers > 0 else None
            )
            
            print(f"訓練批次數量: {len(train_loader)}")
            print(f"驗證批次數量: {len(val_loader)}")
            print(f"測試批次數量: {len(test_loader)}")
            
            torch.backends.cudnn.benchmark = True
            in_channels = 1 if mode in ["ppg", "tonometry", "ekg"] else (3 if mode=="fusion_ekg" else 2)
            model = ModelClass(signal_in_channels=in_channels, personal_input_dim=4, base_channels=32).to(DEVICE)
            model = model.float()  # 确保模型参数为 float32 类型
                
            print(f"模型參數數量: {sum(p.numel() for p in model.parameters())}")
            
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            scaler = GradScaler(enabled=use_amp)
            
            best_val_loss = float('inf')
            
            model_path = f'best_compare_model_{method_name}_{mode}_200.pth'
            if os.path.exists(model_path):
                print(f"加载预训练模型: {model_path}")
                model.load_state_dict(torch.load(model_path))
            else:
                print(f"预训练模型不存在: {model_path}")
            
            for epoch in range(NUM_EPOCHS):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 初始化损失变量
                train_reg_loss = 0.0
                train_contr_loss = 0.0
                val_reg_loss = 0.0
                val_contr_loss = 0.0
                
                # 根据模型类型处理返回值
                if model_type.lower() in ["contrastive", "branch_contrastive"]:
                    train_loss, train_reg_loss, train_contr_loss = train_fn(model, train_loader, optimizer, DEVICE, scaler)
                else:
                    train_loss = train_fn(model, train_loader, optimizer, DEVICE, scaler)
                
                if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == NUM_EPOCHS - 1:
                    if model_type.lower() in ["contrastive", "branch_contrastive"]:
                        val_loss, val_reg_loss, val_contr_loss = val_fn(model, val_loader, DEVICE)
                        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss: {train_loss:.6f} | Reg Loss: {train_reg_loss:.6f} | Contr Loss: {train_contr_loss:.6f}")
                        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Val Loss: {val_loss:.6f} | Reg Loss: {val_reg_loss:.6f} | Contr Loss: {val_contr_loss:.6f}")
                    else:
                        val_loss = val_fn(model, val_loader, DEVICE)
                        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss: {train_loss:.6f}")
                        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Val Loss: {val_loss:.6f}")
                    scheduler.step(val_loss)
                    
                else:
                    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss: {train_loss:.6f}")
            
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
                    torch.save(model_cpu, model_path)
                    print(f"保存最佳模型: {f'best_compare_model_{method_name}_{mode}_200.pth'}")

            results[method_name][mode] = best_val_loss
            print(f"{mode.upper()} 模型最佳驗證損失: {best_val_loss:.6f}")
            
            try:
                
                small_loader = DataLoader(
                    Subset(subset_dataset, torch.randperm(len(subset_dataset))[:min(500, len(subset_dataset))]),
                    batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory
                )
                shap_vals = compute_shap(model, small_loader, DEVICE, model_type=model_type)
                print(f"{mode.upper()} 模型 SHAP 值: mean={np.mean(shap_vals[0]):.6f}, std={np.std(shap_vals[0]):.6f}")
                
                mi = compute_mutual_info(model, small_loader, DEVICE, model_type=model_type)
                print(f"{mode.upper()} 模型 Mutual Information: {mi}")
            except Exception as e:
                print(f"{mode.upper()} 模型分析時出錯: {e}")
            
            print("\n--- 在測試集上評估最佳模型 ---")
            best_model = ModelClass(signal_in_channels=in_channels, personal_input_dim=4, base_channels=32).to(DEVICE)
            best_model.load_state_dict(torch.load(f'best_compare_model_{method_name}_{mode}_200.pth'))
            
            if model_type.lower() in ["contrastive", "branch_contrastive"]:
                test_loss, test_reg_loss, test_contr_loss = val_fn(best_model, test_loader, DEVICE)
                print(f"測試損失: {test_loss:.6f} | Reg Loss: {test_reg_loss:.6f} | Contr Loss: {test_contr_loss:.6f}")
                sbp_mae, dbp_mae = evaluate_bp_error(best_model, test_loader, DEVICE)
                print(f"SBP 平均絕對誤差: {sbp_mae:.2f} mmHg")
                print(f"DBP 平均絕對誤差: {dbp_mae:.2f} mmHg")
            else:
                test_loss = val_fn(best_model, test_loader, DEVICE)
                print(f"測試損失: {test_loss:.6f}")
                sbp_mae, dbp_mae = evaluate_bp_error(best_model, test_loader, DEVICE)
                print(f"SBP 平均絕對誤差: {sbp_mae:.2f} mmHg")
                print(f"DBP 平均絕對誤差: {dbp_mae:.2f} mmHg")
            
            print("\n----------------------------\n")
            
def evaluate_bp_error(model, dataloader, device):
    """
    评估血压预测模型在测试集上的平均绝对误差(MAE)
    
    此函数计算模型预测的收缩压(SBP)和舒张压(DBP)与真实值之间的平均绝对误差。
    支持多种模型架构，包括端到端回归模型和对比学习模型。
    
    参数:
        model (nn.Module): 训练好的血压预测模型
        dataloader (DataLoader): 包含测试数据的DataLoader
        device (torch.device): 执行计算的设备(CPU或CUDA)
        
    返回:
        tuple: (sbp_mae, dbp_mae) - 收缩压和舒张压的平均绝对误差(单位:mmHg)
    """
    model.eval()  # 将模型设置为评估模式
    sbp_errors = []  # 存储所有SBP预测误差
    dbp_errors = []  # 存储所有DBP预测误差
    
    with torch.no_grad():  # 禁用梯度计算以加速推理
        for signal, personal, target in tqdm(dataloader, desc="評估血壓誤差"):
            # 将数据转移到指定设备
            signal = signal.to(device)     # 生理信号数据
            personal = personal.to(device) # 个人信息特征
            target = target.to(device)     # 真实血压值(经过归一化)
            
            # 获取模型预测
            output = model(signal, personal)
            
            # 处理不同类型的模型输出
            if isinstance(output, tuple):
                # 对比学习模型返回(representation, prediction)
                _, pred = output  # 只需要预测部分
            else:
                # 端到端模型直接返回prediction
                pred = output
            
            # 将归一化的预测值转换回实际血压值(mmHg)
            pred_sbp = pred[:, 0] * 200.0  # 收缩压预测值
            pred_dbp = pred[:, 1] * 200.0  # 舒张压预测值
            true_sbp = target[:, 0] * 200.0  # 真实收缩压
            true_dbp = target[:, 1] * 200.0  # 真实舒张压
            
            # 计算绝对误差
            sbp_error = torch.abs(pred_sbp - true_sbp)
            dbp_error = torch.abs(pred_dbp - true_dbp)
            
            # 收集所有批次的误差
            sbp_errors.extend(sbp_error.cpu().numpy())
            dbp_errors.extend(dbp_error.cpu().numpy())
    
    # 计算平均绝对误差
    sbp_mae = np.mean(sbp_errors)
    dbp_mae = np.mean(dbp_errors)
    return sbp_mae, dbp_mae

def analyze_signal_importance(model_path, model_type="End2EndRegressionModel", input_mode="fusion_ekg"):
    """分析不同信号通道的重要性"""
    # 加载模型
    in_channels = 3  # fusion_ekg模式下有3个通道
    model = End2EndRegressionModel(signal_in_channels=in_channels, personal_input_dim=4, base_channels=32).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 准备小数据集
    dataset = BPRegressionDataset(h5_folder=H5_FOLDER, input_mode=input_mode, cache_size=2000)
    indices = torch.randperm(len(dataset))[:500]  # 取500个样本
    small_dataset = Subset(dataset, indices)
    dataloader = DataLoader(small_dataset, batch_size=32, shuffle=False)
    
    # 分别计算每个通道的SHAP值
    shap_values_by_channel = {}
    channel_names = ["ECG", "PPG", "Tonometry"]
    
    # 收集每个通道的输入
    inputs_by_channel = {name: [] for name in channel_names}
    targets = []
    
    with torch.no_grad():
        for batch_idx, (signal, personal, target) in enumerate(dataloader):
            if batch_idx >= 10:  # 限制批次数
                break
            signal = signal.to(DEVICE)
            for i, name in enumerate(channel_names):
                inputs_by_channel[name].append(signal[:, i:i+1, :].cpu().numpy())
            targets.append(target.numpy())
    
    # 计算SHAP值
    for i, name in enumerate(channel_names):
        explainer = shap.DeepExplainer(model, [np.concatenate(inputs_by_channel[name]), np.concatenate(targets)])
        shap_values = explainer.shap_values(np.concatenate(inputs_by_channel[name]))
        shap_values_by_channel[name] = np.abs(shap_values).mean()
    
    # 归一化SHAP值总和为100%
    total_importance = sum(shap_values_by_channel.values())
    relative_importance = {k: (v/total_importance)*100 for k, v in shap_values_by_channel.items()}
    
    return relative_importance

if __name__ == '__main__':
    main()
