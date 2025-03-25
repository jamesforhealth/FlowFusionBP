#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 measurements_auscultatory 與 measurements_oscillometric 兩個資料夾中的原始波型資料，
依據 sliding window 切分後，連同參與者、量測資訊與特徵資訊一起存成 h5 檔案。

重點修改：
  1. 嘗試先以 header=None 讀取 tsv 檔案，若第一列資料無法轉為 float，則改用 header=0 讀取，
     以避免把欄位名稱誤讀入導致全為 NaN。
  2. 呼叫與 GUI app 相同的濾波函式（filter_ekg, filter_ppg, filter_tonometry），使得轉檔後訊號尺度一致。
  3. 從 features.tsv 取得特徵資訊（baseline_sbp、baseline_dbp、delta_sbp、delta_dbp），並存入 h5 屬性。
  4. 參與者性別直接以原始字串儲存（例如 "M" 或 "F"）。
"""

import os
import glob
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from scipy import signal

# ======================
# 參數設定
# ======================
QUALITY_THRESHOLD = 0.65
WINDOW_LENGTH = 4096    # 每個 sliding window 的點數
WINDOW_STEP = 64       # 視窗步幅

# 資料夾與檔案路徑（請依實際情況調整）
AUSCULTATORY_DIR = 'measurements_auscultatory'
OSCILLOMETRIC_DIR = 'measurements_oscillometric'
OUTPUT_DIR = 'output_h5_folder'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Metadata 檔案
PARTICIPANTS_TSV = 'participants.tsv'
FEATURES_TSV = 'features.tsv'  # 若無此檔案可忽略
AUSCULTATORY_META_TSV = 'measurements_auscultatory.tsv'
OSCILLOMETRIC_META_TSV = 'measurements_oscillometric.tsv'

# ======================
# 讀取 Metadata 檔案
# ======================
print("讀取 metadata 檔案...")
participants_df = pd.read_csv(PARTICIPANTS_TSV, sep='\t', dtype=str)
if os.path.exists(FEATURES_TSV):
    features_df = pd.read_csv(FEATURES_TSV, sep='\t', dtype=str)
else:
    features_df = None

ausc_meta_df = pd.read_csv(AUSCULTATORY_META_TSV, sep='\t', dtype=str)
osc_meta_df = pd.read_csv(OSCILLOMETRIC_META_TSV, sep='\t', dtype=str)

# ----------------------
# Helper 函式：根據 pid, phase, measurement id 取得量測 metadata
# ----------------------
def get_measurement_metadata(pid, phase, meas_id, method='auscultatory'):
    meas_id_for_compare = meas_id.replace('_',' ')
    df = ausc_meta_df if method == 'auscultatory' else osc_meta_df
    meta_rows = df[(df['pid'] == pid) & 
                   (df['phase'] == phase) & 
                   (df['measurement'].str.replace("_", " ") == meas_id_for_compare)]
    if not meta_rows.empty:
        return meta_rows.iloc[0]
    else:
        return None

# ----------------------
# (選用) 濾波函式，與 GUI 內使用的相同
# ----------------------
def filter_ekg(x, fs):
    sos_dc = signal.iirfilter(N=2, Wn=0.1/(fs/2), btype='highpass', ftype='butter', output='sos')
    y = signal.sosfiltfilt(sos_dc, x)
    wp = 40.0 / (fs/2.0)
    ws = 45.0 / (fs/2.0)
    sos_lp = signal.iirdesign(wp=wp, ws=ws, gpass=0.1, gstop=40, ftype='ellip', output='sos')
    y = signal.sosfiltfilt(sos_lp, y)
    w0 = 60.0 / (fs/2.0)
    bw = w0 / 3.0
    w1 = [w0 - bw/2.0, w0 + bw/2.0]
    sos_notch = signal.iirfilter(N=6, rp=0.1, Wn=w1, btype='bandstop', ftype='cheby1', output='sos')
    y = signal.sosfiltfilt(sos_notch, y)
    return y

def filter_ppg(x, fs):
    sos_hp = signal.butter(N=4, Wn=0.25/(fs/2.0), btype='highpass', output='sos')
    y = signal.sosfiltfilt(sos_hp, x)
    wp = 10.0 / (fs/2.0)
    ws = 12.0 / (fs/2.0)
    sos_lp = signal.iirdesign(wp=wp, ws=ws, gpass=1, gstop=60, ftype='ellip', output='sos')
    y = signal.sosfiltfilt(sos_lp, y)
    return y

def filter_tonometry(x, fs):
    ws = 0.2/(fs/2.0)
    wp = 0.3/(fs/2.0)
    sos_hp = signal.iirdesign(wp=wp, ws=ws, gpass=1, gstop=60, ftype='ellip', output='sos')
    y = signal.sosfiltfilt(sos_hp, x)
    wp2 = 22.0 / (fs/2.0)
    ws2 = 26.0 / (fs/2.0)
    sos_lp = signal.iirdesign(wp=wp2, ws=ws2, gpass=0.1, gstop=40, ftype='ellip', output='sos')
    y = signal.sosfiltfilt(sos_lp, y)
    return y

# ----------------------
# 處理資料夾
# ----------------------
measurement_dirs = [
    (AUSCULTATORY_DIR, 'auscultatory'),
    (OSCILLOMETRIC_DIR, 'oscillometric')
]

for m_dir, method in measurement_dirs:
    if not os.path.exists(m_dir):
        print(f"資料夾 {m_dir} 不存在，跳過")
        continue

    subdirs = sorted([d for d in os.listdir(m_dir) if os.path.isdir(os.path.join(m_dir, d))])
    for subdir in tqdm(subdirs, desc=f"處理 {m_dir} 子資料夾"):
        subject_folder = os.path.join(m_dir, subdir)
        default_pid = subdir
        tsv_files = glob.glob(os.path.join(subject_folder, '*.tsv'))
        
        for file in tsv_files:
            filename = os.path.basename(file)
            h5_filename = os.path.splitext(filename)[0] + '.h5'
            h5_filepath = os.path.join(OUTPUT_DIR, h5_filename)
            if os.path.exists(h5_filepath):
                print(f"h5 檔案 {h5_filename} 已存在，跳過")
                continue

            parts = filename.split('.')
            if parts[-1].lower() != 'tsv':
                continue

            if len(parts) == 3:
                pid = default_pid
                phase = parts[0]
                meas_id = parts[1]
            elif len(parts) >= 4:
                pid = parts[0]
                phase = parts[1]
                meas_id = parts[2]
            else:
                print(f"檔名格式不符，略過: {filename}")
                continue

            measurement = meas_id
            measurement_for_compare = measurement.replace("_", " ")

            # -----------------------------
            # 讀取 TSV 檔案
            # -----------------------------
            try:
                data = pd.read_csv(file, sep='\t', header=None)
                try:
                    float(data.iloc[0, 0])
                except (ValueError, TypeError):
                    data = pd.read_csv(file, sep='\t', header=0)
            except Exception as e:
                print(f"讀取 {filename} 時發生錯誤: {e}，跳過")
                continue

            if data.shape[1] == 7:
                ekg_signal = pd.to_numeric(data.iloc[:, 1], errors='coerce').values
                ppg_signal = pd.to_numeric(data.iloc[:, 2], errors='coerce').values
                tono_signal = pd.to_numeric(data.iloc[:, 3], errors='coerce').values
            elif data.shape[1] == 6:
                ekg_signal = pd.to_numeric(data.iloc[:, 0], errors='coerce').values
                ppg_signal = pd.to_numeric(data.iloc[:, 1], errors='coerce').values
                tono_signal = pd.to_numeric(data.iloc[:, 2], errors='coerce').values
            else:
                print(f"{filename} 欄位數 ({data.shape[1]}) 不符合預期，略過。")
                continue

            min_length = min(len(ekg_signal), len(ppg_signal), len(tono_signal))
            if min_length < WINDOW_LENGTH:
                print(f"{filename} 信號長度 ({min_length}) 少於視窗長度，略過。")
                continue

            ekg_signal = ekg_signal[:min_length]
            ppg_signal = ppg_signal[:min_length]
            tono_signal = tono_signal[:min_length]

            # 估算 fs，若無時間資訊則假設等間隔
            fs = None
            if min_length > 1:
                if data.shape[1] == 7:
                    x_data = pd.to_numeric(data.iloc[:, 0], errors='coerce').values
                    if np.all(~np.isnan(x_data)):
                        dt = np.mean(np.diff(x_data))
                        if dt > 0:
                            fs = 1.0 / dt
                else:
                    fs = 500
            if fs is not None:
                ekg_signal = filter_ekg(ekg_signal, fs)
                ppg_signal = filter_ppg(ppg_signal, fs)
                tono_signal = filter_tonometry(tono_signal, fs)

            ekg_windows = []
            ppg_windows = []
            tono_windows = []
            for start in range(0, min_length - WINDOW_LENGTH + 1, WINDOW_STEP):
                ekg_windows.append(ekg_signal[start:start+WINDOW_LENGTH])
                ppg_windows.append(ppg_signal[start:start+WINDOW_LENGTH])
                tono_windows.append(tono_signal[start:start+WINDOW_LENGTH])
            if len(ekg_windows) == 0:
                print(f"{filename} 無法切出視窗，略過。")
                continue

            ekg_windows = np.array(ekg_windows)
            ppg_windows = np.array(ppg_windows)
            tono_windows = np.array(tono_windows)

            if np.isnan(ekg_windows[0]).all() or np.isnan(ppg_windows[0]).all() or np.isnan(tono_windows[0]).all():
                print(f"警告：{filename} 第一個視窗有問題 (全 NaN)，請檢查檔案內容")
            
            # -----------------------------
            # 存成 h5 檔案
            # -----------------------------
            try:
                with h5py.File(h5_filepath, 'w') as hf:
                    hf.create_dataset('EKG', data=ekg_windows)
                    hf.create_dataset('PPG', data=ppg_windows)
                    hf.create_dataset('Tonometry', data=tono_windows)
                    
                    # 存入受試者資訊 (gender 直接存原始字串，如 "M" 或 "F")
                    part_row = participants_df[participants_df['pid'] == pid].iloc[0]
                    participant_info = {
                        "age": part_row.get("age", ""),
                        #gender編碼成0或1
                        "gender": 0 if part_row.get("gender", "") == "M" else 1,
                        "height": part_row.get("height", ""),
                        "weight": part_row.get("weight", "")
                    }
                    for key, value in participant_info.items():
                        hf.attrs[f"participant_{key}"] = value
                    
                    # 存入量測資訊
                    current_meta_df = ausc_meta_df if method=='auscultatory' else osc_meta_df
                    rows = current_meta_df[
                        (current_meta_df['pid'] == pid) &
                        (current_meta_df['phase'] == phase) &
                        (current_meta_df['measurement'] == measurement_for_compare)
                    ]
                    meas_dict = {}
                    if not rows.empty:
                        meas_dict = rows.to_dict('records')[0]
                    measurement_info = {
                        "sbp": meas_dict.get("sbp", ""),
                        "dbp": meas_dict.get("dbp", "")
                    }
                    for key, value in measurement_info.items():
                        hf.attrs[f"measurement_{key}"] = value
                    
                    # 存入特徵資訊 (從 features.tsv 取得)
                    if features_df is not None:
                        feat_rows = features_df[
                            (features_df['pid'] == pid) &
                            (features_df['phase'] == phase) &
                            (features_df['measurement'] == measurement_for_compare)
                        ]
                        if not feat_rows.empty:
                            feat_dict = feat_rows.to_dict('records')[0]
                            hf.attrs["feature_baseline_sbp"] = feat_dict.get("baseline_sbp", "")
                            hf.attrs["feature_baseline_dbp"] = feat_dict.get("baseline_dbp", "")
                            hf.attrs["feature_delta_sbp"] = feat_dict.get("delta_sbp", "")
                            hf.attrs["feature_delta_dbp"] = feat_dict.get("delta_dbp", "")
                    # print(f"EKG.shape: {ekg_windows.shape}")
                    # print(f"PPG.shape: {ppg_windows.shape}")
                    # print(f"Tonometry.shape: {tono_windows.shape}")
                    # for attr_key in hf.attrs:
                    #     print(f"  {attr_key}: {hf.attrs[attr_key]}")
                    # input("Press Enter to continue...")
            except Exception as e:
                print(f"存檔 {h5_filename} 發生錯誤: {e}，跳過此檔案")
                continue

            print(f"已存 {h5_filename}（來源: {filename}，共 {ekg_windows.shape[0]} 個視窗）")

print("全部轉檔完成！")
