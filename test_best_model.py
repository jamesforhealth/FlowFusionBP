import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import os
import argparse
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 导入数据集类
from model_bp_regression_compare import BPRegressionDataset

# 设定常量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H5_FOLDER = "output_h5_folder"  # 请替换为您的数据夹路径
BATCH_SIZE = 128

def test_model(model_path, model_type="ContrastiveBPModel", input_mode="fusion_ekg", test_size=50000):
    """
    测试指定的血压预测模型
    
    参数:
        model_path: 模型权重文件路径
        model_type: 模型类型名称，可选值: "ContrastiveBPModel", "End2EndRegressionModel", "BranchContrastiveBPModel"
        input_mode: 输入模式，如 "fusion_ekg", "ppg", "ekg_ppg" 等
        test_size: 测试集大小，默认使用后50000个样本
    """
    print(f"使用设备: {DEVICE}")
    print(f"加载模型: {model_path}")
    print(f"模型类型: {model_type}")
    print(f"输入模式: {input_mode}")
    
    # 加载数据集
    full_dataset = BPRegressionDataset(h5_folder=H5_FOLDER, input_mode=input_mode, cache_size=2000)
    print(f"完整数据集大小: {len(full_dataset)}")
    
    # 使用指定数量样本作为测试集
    dataset_size = len(full_dataset)
    if dataset_size > test_size:
        start_idx = max(0, dataset_size - test_size)
        indices = list(range(start_idx, dataset_size))
        test_dataset = Subset(full_dataset, indices)
    else:
        test_dataset = full_dataset
        
    print(f"测试集大小: {len(test_dataset)}")
    
    # 建立数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    # 动态导入并实例化模型
    try:
        # 导入相应的模型类
        from model_bp_regression_compare import (
            ContrastiveBPModel, 
            End2EndRegressionModel, 
            BranchContrastiveBPModel
        )
        
        # 获取输入通道数
        if input_mode == "ppg" or input_mode == "tonometry" or input_mode == "ekg":
            signal_in_channels = 1
        elif input_mode == "fusion":
            signal_in_channels = 2
        elif input_mode == "fusion_ekg" or input_mode == "ekg_ppg":
            signal_in_channels = 3 if input_mode == "fusion_ekg" else 2
        else:
            raise ValueError(f"不支持的输入模式: {input_mode}")
        
        # 根据模型类型创建模型
        ModelClass = {
            "ContrastiveBPModel": ContrastiveBPModel,
            "End2EndRegressionModel": End2EndRegressionModel,
            "BranchContrastiveBPModel": BranchContrastiveBPModel
        }.get(model_type)
        
        if ModelClass is None:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model = ModelClass(signal_in_channels=signal_in_channels, personal_input_dim=4, base_channels=32).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
    # 预测与评估
    all_sbp_pred = []
    all_dbp_pred = []
    all_sbp_true = []
    all_dbp_true = []
    
    with torch.no_grad():
        for signal, personal, target in tqdm(test_loader, desc="评估血压预测"):
            signal = signal.to(DEVICE, dtype=torch.float32)
            personal = personal.to(DEVICE, dtype=torch.float32)
            
            # 根据不同模型类型处理输出
            if isinstance(model, ContrastiveBPModel) or isinstance(model, BranchContrastiveBPModel):
                _, pred = model(signal, personal)
            else:
                pred = model(signal, personal)
            
            # 收集预测和真实血压值
            all_sbp_pred.extend(pred[:, 0].cpu().numpy())
            all_dbp_pred.extend(pred[:, 1].cpu().numpy())
            all_sbp_true.extend(target[:, 0].numpy())
            all_dbp_true.extend(target[:, 1].numpy())
    
    # 转换为numpy数组
    all_sbp_pred = np.array(all_sbp_pred)
    all_dbp_pred = np.array(all_dbp_pred)
    all_sbp_true = np.array(all_sbp_true)
    all_dbp_true = np.array(all_dbp_true)
    
    # 如果数据已正规化，请将其转换回实际血压值（假设已正规化到0-1区间）
    sbp_min, sbp_max = 60, 200  # 示例值，请根据您的数据集调整
    dbp_min, dbp_max = 30, 150  # 示例值，请根据您的数据集调整
    
    sbp_pred_mmHg = all_sbp_pred * (sbp_max - sbp_min) + sbp_min
    dbp_pred_mmHg = all_dbp_pred * (dbp_max - dbp_min) + dbp_min
    sbp_true_mmHg = all_sbp_true * (sbp_max - sbp_min) + sbp_min
    dbp_true_mmHg = all_dbp_true * (dbp_max - dbp_min) + dbp_min
    
    # 计算评估指标
    sbp_mae = mean_absolute_error(sbp_true_mmHg, sbp_pred_mmHg)
    dbp_mae = mean_absolute_error(dbp_true_mmHg, dbp_pred_mmHg)
    sbp_rmse = np.sqrt(mean_squared_error(sbp_true_mmHg, sbp_pred_mmHg))
    dbp_rmse = np.sqrt(mean_squared_error(dbp_true_mmHg, dbp_pred_mmHg))
    sbp_r2 = r2_score(sbp_true_mmHg, sbp_pred_mmHg)
    dbp_r2 = r2_score(dbp_true_mmHg, dbp_pred_mmHg)
    
    results_dir = f"test_results_{model_type}_{input_mode}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n===== 血压预测评估结果 =====")
    print(f"SBP 平均绝对误差 (MAE): {sbp_mae:.2f} mmHg")
    print(f"DBP 平均绝对误差 (MAE): {dbp_mae:.2f} mmHg")
    print(f"SBP 均方根误差 (RMSE): {sbp_rmse:.2f} mmHg")
    print(f"DBP 均方根误差 (RMSE): {dbp_rmse:.2f} mmHg")
    print(f"SBP 决定系数 (R²): {sbp_r2:.4f}")
    print(f"DBP 决定系数 (R²): {dbp_r2:.4f}")
    
    # 计算在不同误差范围内的百分比
    sbp_errors = np.abs(sbp_pred_mmHg - sbp_true_mmHg)
    dbp_errors = np.abs(dbp_pred_mmHg - dbp_true_mmHg)
    
    print("\n===== 误差分布 =====")
    for threshold in [5, 10, 15]:
        sbp_within = np.mean(sbp_errors <= threshold) * 100
        dbp_within = np.mean(dbp_errors <= threshold) * 100
        print(f"SBP 误差在 {threshold} mmHg 内的比例: {sbp_within:.2f}%")
        print(f"DBP 误差在 {threshold} mmHg 内的比例: {dbp_within:.2f}%")
    
    # 绘制散点图
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(sbp_true_mmHg, sbp_pred_mmHg, alpha=0.3, s=10)
    min_val = min(sbp_true_mmHg.min(), sbp_pred_mmHg.min())
    max_val = max(sbp_true_mmHg.max(), sbp_pred_mmHg.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('实际 SBP (mmHg)')
    plt.ylabel('预测 SBP (mmHg)')
    plt.title(f'SBP 预测 (MAE: {sbp_mae:.2f} mmHg, R²: {sbp_r2:.4f})')
    
    plt.subplot(1, 2, 2)
    plt.scatter(dbp_true_mmHg, dbp_pred_mmHg, alpha=0.3, s=10)
    min_val = min(dbp_true_mmHg.min(), dbp_pred_mmHg.min())
    max_val = max(dbp_true_mmHg.max(), dbp_pred_mmHg.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('实际 DBP (mmHg)')
    plt.ylabel('预测 DBP (mmHg)')
    plt.title(f'DBP 预测 (MAE: {dbp_mae:.2f} mmHg, R²: {dbp_r2:.4f})')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/bp_prediction_results.png', dpi=300)
    print(f"\n结果散点图已保存为 '{results_dir}/bp_prediction_results.png'")
    
    # 绘制误差分布直方图
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(sbp_errors, bins=30, alpha=0.7)
    plt.axvline(x=sbp_mae, color='r', linestyle='--', label=f'MAE: {sbp_mae:.2f}')
    plt.xlabel('绝对误差 (mmHg)')
    plt.ylabel('样本数')
    plt.title('SBP 预测误差分布')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(dbp_errors, bins=30, alpha=0.7)
    plt.axvline(x=dbp_mae, color='r', linestyle='--', label=f'MAE: {dbp_mae:.2f}')
    plt.xlabel('绝对误差 (mmHg)')
    plt.ylabel('样本数')
    plt.title('DBP 预测误差分布')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/bp_error_distribution.png', dpi=300)
    print(f"误差分布直方图已保存为 '{results_dir}/bp_error_distribution.png'")
    
    # 找出预测最好和最差的样本
    best_sbp_idx = np.argmin(sbp_errors)
    worst_sbp_idx = np.argmax(sbp_errors)
    best_dbp_idx = np.argmin(dbp_errors)
    worst_dbp_idx = np.argmax(dbp_errors)
    
    print("\n===== 最佳与最差预测样本 =====")
    print(f"最佳 SBP 预测: 实际 {sbp_true_mmHg[best_sbp_idx]:.1f}, 预测 {sbp_pred_mmHg[best_sbp_idx]:.1f}, 误差 {sbp_errors[best_sbp_idx]:.1f} mmHg")
    print(f"最差 SBP 预测: 实际 {sbp_true_mmHg[worst_sbp_idx]:.1f}, 预测 {sbp_pred_mmHg[worst_sbp_idx]:.1f}, 误差 {sbp_errors[worst_sbp_idx]:.1f} mmHg")
    print(f"最佳 DBP 预测: 实际 {dbp_true_mmHg[best_dbp_idx]:.1f}, 预测 {dbp_pred_mmHg[best_dbp_idx]:.1f}, 误差 {dbp_errors[best_dbp_idx]:.1f} mmHg")
    print(f"最差 DBP 预测: 实际 {dbp_true_mmHg[worst_dbp_idx]:.1f}, 预测 {dbp_pred_mmHg[worst_dbp_idx]:.1f}, 误差 {dbp_errors[worst_dbp_idx]:.1f} mmHg")
    
    # 保存结果到文本文件
    with open(f'{results_dir}/results_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"模型: {model_type}\n")
        f.write(f"输入模式: {input_mode}\n")
        f.write(f"模型路径: {model_path}\n\n")
        f.write("===== 血压预测评估结果 =====\n")
        f.write(f"SBP 平均绝对误差 (MAE): {sbp_mae:.2f} mmHg\n")
        f.write(f"DBP 平均绝对误差 (MAE): {dbp_mae:.2f} mmHg\n")
        f.write(f"SBP 均方根误差 (RMSE): {sbp_rmse:.2f} mmHg\n")
        f.write(f"DBP 均方根误差 (RMSE): {dbp_rmse:.2f} mmHg\n")
        f.write(f"SBP 决定系数 (R²): {sbp_r2:.4f}\n")
        f.write(f"DBP 决定系数 (R²): {dbp_r2:.4f}\n\n")
        
        f.write("===== 误差分布 =====\n")
        for threshold in [5, 10, 15]:
            sbp_within = np.mean(sbp_errors <= threshold) * 100
            dbp_within = np.mean(dbp_errors <= threshold) * 100
            f.write(f"SBP 误差在 {threshold} mmHg 内的比例: {sbp_within:.2f}%\n")
            f.write(f"DBP 误差在 {threshold} mmHg 内的比例: {dbp_within:.2f}%\n")
    
    print(f"结果摘要已保存至 '{results_dir}/results_summary.txt'")
    
    return {
        'sbp_mae': sbp_mae,
        'dbp_mae': dbp_mae,
        'sbp_rmse': sbp_rmse,
        'dbp_rmse': dbp_rmse,
        'sbp_r2': sbp_r2,
        'dbp_r2': dbp_r2
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试血压预测模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--model_type', type=str, default='ContrastiveBPModel', 
                        choices=['ContrastiveBPModel', 'End2EndRegressionModel', 'BranchContrastiveBPModel'],
                        help='模型类型')
    parser.add_argument('--input_mode', type=str, default='fusion_ekg', 
                        choices=['ppg', 'tonometry', 'fusion', 'fusion_ekg', 'ekg', 'ekg_ppg'],
                        help='输入信号模式')
    parser.add_argument('--test_size', type=int, default=100000, help='测试样本数量')
    
    args = parser.parse_args()
    test_model(args.model_path, args.model_type, args.input_mode, args.test_size) 