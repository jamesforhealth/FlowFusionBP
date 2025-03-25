#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檢查 h5 檔案內容

本程式會遍歷指定資料夾內所有 h5 檔案，
對每個檔案依序檢查以下內容：
  1. 各個資料集（"EKG", "PPG", "Tonometry"）是否存在，並檢查其中是否包含 NaN，
     同時列印其 shape、平均值與標準差。
  2. 重要屬性（例如 participant_age、participant_gender、participant_height、participant_weight、
     measurement_sbp、measurement_dbp、feature_delta_sbp、feature_delta_dbp）是否存在及可轉換為 float，
     並檢查是否為 NaN。
  
檢查結果將在終端機中印出，方便你確認數據是否有問題。
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
def check_h5_file(file_path):
    report = []
    # report.append(f"檔案: {file_path}")
    try:
        with h5py.File(file_path, 'r') as hf:
            # 檢查主要資料集
            datasets = ['EKG', 'PPG', 'Tonometry']
            for ds in datasets:
                if ds in hf:
                    data = hf[ds][:]
                    if np.any(np.isnan(data)):
                        report.append(f"  [警告] 波型資料 {ds} 中含有 NaN 值！")
                        input(f"[警告] 波型資料 {ds} 中含有 NaN 值 Press Enter to continue...")
                    # else:
                    #     mean_val = np.mean(data)
                    #     std_val = np.std(data)
                    #     report.append(f"  {ds} 資料檢查通過 (shape: {data.shape}, mean: {mean_val:.4f}, std: {std_val:.4f})")
                else:
                    report.append(f"  [警告] 找不到資料集 {ds}。")
                    input(f"[警告] 找不到資料集 {ds} Press Enter to continue...")
            
            # 檢查重要屬性
            attr_names = [
                'participant_age',
                'participant_gender',
                'participant_height',
                'participant_weight',
                'measurement_sbp',
                'measurement_dbp',
                # 'feature_delta_sbp',
                # 'feature_delta_dbp'
            ]
            for attr in attr_names:
                if attr in hf.attrs:
                    attr_val = hf.attrs[attr]
                    try:
                        attr_val_float = float(attr_val)
                        if np.isnan(attr_val_float):
                            report.append(f"  [警告] 屬性 {attr} 為 NaN！")
                            input(f"[警告] 屬性 {attr} 為 NaN Press Enter to continue...")
                        # else:
                        #     report.append(f"  屬性 {attr} 檢查通過：{attr_val_float}")
                    except Exception as e:
                        report.append(f"  [錯誤] 屬性 {attr} 轉換為 float 失敗：{attr_val} (錯誤: {e})")
                        input(f"[錯誤] 屬性 {attr} 轉換為 float 失敗 Press Enter to continue...")
                else:
                    report.append(f"  [警告] 屬性 {attr} 不存在。")
                    input(f"[警告] 屬性 {attr} 不存在 Press Enter to continue...")
    except Exception as e:
        report.append(f"  [錯誤] 無法打開檔案: {e}")
    return "\n".join(report)

def main():
    h5_folder = 'output_h5_folder'
    if not os.path.exists(h5_folder):
        print(f"資料夾 {h5_folder} 不存在。")
        return
    files = [os.path.join(h5_folder, f) for f in os.listdir(h5_folder) if f.endswith('.h5')]
    if not files:
        print("未找到任何 h5 檔案。")
        return
    for file in tqdm(files):
        report = check_h5_file(file)
        # print(report)
        # print("-" * 50)

if __name__ == '__main__':
    main()
