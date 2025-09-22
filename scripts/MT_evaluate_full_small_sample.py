import os
os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch.multiprocessing as mp
import sys
import datetime
import logging
from torch.cuda.amp import autocast
import joblib

# 设置多处理方法
mp.set_start_method('spawn', force=True)

# 导入自定义模块
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from scripts.MTI_Net import setup_logger, set_seed, calculate_metrics, format_metrics_table, MultiTaskTMI, CSVDataset

def load_pretrained_model(model_path, device, input_size, num_tasks):
    """加载预训练模型"""
    # 创建模型实例 - 确保参数与训练时一致
    model = MultiTaskTMI(
        input_size=input_size,
        hidden_dim=128,
        num_layers=1,
        num_heads=1,
        num_tasks=num_tasks,
        drop=0.15        
    ).to(device)
    
    try:
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        
        # 处理不同的保存格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 打印模型键名以便调试
        print("模型权重键名:")
        for key in list(state_dict.keys())[:10]:  # 只显示前10个
            print(f"  {key}")
        
        # 尝试加载权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"意外的键: {unexpected_keys}")
        
        print(f"成功加载预训练模型: {model_path}")
        return model
    except Exception as e:
        print(f"加载预训练模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_full_dataset(model, dataloader, task_names, device, logger, target_scaler=None):
    """评估模型在完整数据集上的性能"""
    model.eval()
    all_preds = []
    all_targets = []
    
    logger.info("开始评估模型性能...")
    
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(dataloader):
            features = features.to(device)
            targets = targets.to(device)
            
            with autocast():
                outputs = model(features)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"已处理 {batch_idx + 1} 个批次")
    
    # 合并所有预测和目标
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    logger.info(f"评估完成，共处理 {len(all_preds)} 个样本")
    
    # 计算标准化数据的指标
    r2_normalized, rmse_normalized, rpd_normalized = calculate_metrics(all_targets, all_preds)
    
    # 如果有标准化器，计算原始数据的指标
    if target_scaler is not None:
        # 反标准化
        all_preds_original = target_scaler.inverse_transform(all_preds)
        all_targets_original = target_scaler.inverse_transform(all_targets)
        
        # 计算原始数据的指标
        r2_original, rmse_original, rpd_original = calculate_metrics(all_targets_original, all_preds_original)
        
        return {
            'predictions_normalized': all_preds,
            'targets_normalized': all_targets,
            'predictions_original': all_preds_original,
            'targets_original': all_targets_original,
            'metrics_normalized': {
                'r2': r2_normalized,
                'rmse': rmse_normalized,
                'rpd': rpd_normalized
            },
            'metrics_original': {
                'r2': r2_original,
                'rmse': rmse_original,
                'rpd': rpd_original
            }
        }
    else:
        return {
            'predictions_normalized': all_preds,
            'targets_normalized': all_targets,
            'metrics_normalized': {
                'r2': r2_normalized,
                'rmse': rmse_normalized,
                'rpd': rpd_normalized
            }
        }

def save_detailed_results(results, task_names, data, results_dir):
    """保存详细的评估结果"""
    # 保存预测结果
    if 'predictions_original' in results:
        # 使用原始数据
        pred_df = pd.DataFrame(results['predictions_original'], 
                              columns=[f'pred_{col}' for col in task_names])
        target_df = pd.DataFrame(results['targets_original'], 
                                columns=[f'true_{col}' for col in task_names])
        
        # 添加样本信息
        if len(data) == len(pred_df):
            # 添加前三列（通常是样本标识信息）
            info_df = data.iloc[:, :3].reset_index(drop=True)
            result_df = pd.concat([info_df, target_df, pred_df], axis=1)
        else:
            result_df = pd.concat([target_df, pred_df], axis=1)
        
        # 计算误差
        for col in task_names:
            result_df[f'error_{col}'] = result_df[f'pred_{col}'] - result_df[f'true_{col}']
            result_df[f'abs_error_{col}'] = np.abs(result_df[f'error_{col}'])
            result_df[f'rel_error_{col}'] = result_df[f'error_{col}'] / result_df[f'true_{col}'] * 100
        
        result_file = os.path.join(results_dir, 'full_dataset_predictions.csv')
        result_df.to_csv(result_file, index=False)
        print(f"保存详细预测结果到: {result_file}")
        
        # 保存指标汇总
        metrics = results['metrics_original']
    else:
        metrics = results['metrics_normalized']
    
    # 创建指标汇总DataFrame
    metrics_df = pd.DataFrame({
        'Parameter': task_names,
        'R2': metrics['r2'],
        'RMSE': metrics['rmse'],
        'RPD': metrics['rpd']
    })
    
    # 添加平均值行
    avg_row = pd.DataFrame({
        'Parameter': ['Average'],
        'R2': [np.mean(metrics['r2'])],
        'RMSE': [np.mean(metrics['rmse'])],
        'RPD': [np.mean(metrics['rpd'])]
    })
    
    metrics_df = pd.concat([metrics_df, avg_row], ignore_index=True)
    
    metrics_file = os.path.join(results_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    print(f"保存评估指标到: {metrics_file}")

def analyze_parameter_performance(results, task_names, logger):
    """分析各参数的性能表现"""
    if 'metrics_original' in results:
        metrics = results['metrics_original']
    else:
        metrics = results['metrics_normalized']
    
    r2_values = metrics['r2']
    rmse_values = metrics['rmse']
    rpd_values = metrics['rpd']
    
    logger.info("\n========= 参数性能分析 =========")
    
    # 按R²排序
    r2_ranking = sorted(enumerate(r2_values), key=lambda x: x[1], reverse=True)
    logger.info("\n按R²排序的参数性能:")
    for rank, (idx, r2_val) in enumerate(r2_ranking, 1):
        logger.info(f"{rank:2d}. {task_names[idx]:<10} R²={r2_val:.4f}")
    
    # 按RPD排序
    rpd_ranking = sorted(enumerate(rpd_values), key=lambda x: x[1], reverse=True)
    logger.info("\n按RPD排序的参数性能:")
    for rank, (idx, rpd_val) in enumerate(rpd_ranking, 1):
        logger.info(f"{rank:2d}. {task_names[idx]:<10} RPD={rpd_val:.4f}")
    
    # 性能分类
    excellent = [task_names[i] for i, r2 in enumerate(r2_values) if r2 >= 0.8]
    good = [task_names[i] for i, r2 in enumerate(r2_values) if 0.6 <= r2 < 0.8]
    fair = [task_names[i] for i, r2 in enumerate(r2_values) if 0.4 <= r2 < 0.6]
    poor = [task_names[i] for i, r2 in enumerate(r2_values) if r2 < 0.4]
    
    logger.info("\n按性能等级分类:")
    logger.info(f"优秀 (R²≥0.8): {excellent}")
    logger.info(f"良好 (0.6≤R²<0.8): {good}")
    logger.info(f"一般 (0.4≤R²<0.6): {fair}")
    logger.info(f"较差 (R²<0.4): {poor}")

def main():
    # 配置参数
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        # 'pretrained_model_path': 'results/MT csv_multitask_vit_13param/1290/best_model.pt',
        # 'small_sample_data_path': 'data/processed/subsets/129_MT_all_samples_mean_reflectance_raw.csv',
        # 'task_names': ['RUE', 'Pn', 'Gs', 'Ci', 'Tr', 'Ci-Ca', 'WUE', 'iWUE', 'Pmax', 'Rd', 'Ic', 'SPAD', 'LAW'],  # 13个参数
        # 'results_dir': os.path.join('results', 'MT_full_small_sample_evaluation'),
        'pretrained_model_path': 'results/Rice csv_multitask_vit_8param/1141/best_model.pt',
        'small_sample_data_path': 'data/processed/Rice subsets/164_rice_all_samples_mean_reflectance_raw.csv',
        'task_names': ['SPAD','Pn', 'LNC', 'Chl-a', 'Chl-b','LAW', 'Cx', 'Chl'],  # 8个参数
        'results_dir': os.path.join('results', 'rice_full_small_sample_evaluation'),
        'feature_cols': list(range(3, 276)),  # 特征列索引
        'batch_size': 32,
        'seed': 42,
        'save_predictions': True,  # 是否保存预测结果
        'normalize_data': True,    # 是否标准化数据
    }
    
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config['results_dir'] = os.path.join(config['results_dir'], timestamp)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # 设置日志和随机种子
    logger = setup_logger(config)
    set_seed(config['seed'])
    
    logger.info("==================== 小样本数据集完整评估 ====================")
    logger.info(f"预训练模型路径: {config['pretrained_model_path']}")
    logger.info(f"小样本数据路径: {config['small_sample_data_path']}")
    logger.info(f"评估参数: {config['task_names']}")
    
    # 检查预训练模型是否存在
    if not os.path.exists(config['pretrained_model_path']):
        logger.error(f"预训练模型文件不存在: {config['pretrained_model_path']}")
        return
    
    # 加载小样本数据
    logger.info("加载小样本数据...")
    try:
        df = pd.read_csv(config['small_sample_data_path'])
        logger.info(f"成功加载数据，共 {len(df)} 条记录")
        
        # 检查目标列是否存在
        missing_cols = [col for col in config['task_names'] if col not in df.columns]
        if missing_cols:
            logger.error(f"以下目标列不存在: {missing_cols}")
            return
        
        # 检查特征列是否存在
        if max(config['feature_cols']) >= df.shape[1]:
            logger.error(f"特征列索引超出范围: {max(config['feature_cols'])} >= {df.shape[1]}")
            return
        
        logger.info("数据基本统计信息:")
        for col in config['task_names']:
            logger.info(f"{col}: 均值={df[col].mean():.4f}, 标准差={df[col].std():.4f}, 范围=[{df[col].min():.4f}, {df[col].max():.4f}]")
            
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        return
    
    # 创建数据集（使用全部数据，不划分）
    # 重要：使用与训练时相同的标准化方式
    dataset = CSVDataset(
        data=df,
        feature_cols=config['feature_cols'],
        target_cols=config['task_names'],
        normalize=config['normalize_data'],
        device=None
    )
    
    # 检查是否需要加载训练时的标准化器
    scaler_path = os.path.join(os.path.dirname(config['pretrained_model_path']), 'scalers.pkl')
    if os.path.exists(scaler_path):
        logger.info(f"发现训练时的标准化器文件: {scaler_path}")
        try:
            saved_scalers = joblib.load(scaler_path)
            # 使用训练时的标准化器
            if config['normalize_data']:
                dataset.feature_scaler = saved_scalers['feature_scaler']
                dataset.target_scaler = saved_scalers['target_scaler']
                # 重新应用标准化
                dataset.features = dataset.feature_scaler.transform(dataset.features)
                dataset.targets = dataset.target_scaler.transform(dataset.targets)
                logger.info("成功加载并应用训练时的标准化器")
        except Exception as e:
            logger.warning(f"加载训练时标准化器失败: {e}")
            logger.warning("将使用当前数据计算的标准化器")
    else:
        logger.warning(f"未找到训练时的标准化器文件: {scaler_path}")
        logger.warning("将使用当前数据计算的标准化器")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,  # 不打乱，保持原始顺序
        num_workers=0
    )
    
    logger.info(f"数据集大小: {len(dataset)} 个样本")
    logger.info(f"特征维度: {len(config['feature_cols'])}")
    logger.info(f"目标任务数: {len(config['task_names'])}")
    
    # 加载预训练模型
    logger.info("加载预训练模型...")
    model = load_pretrained_model(
        model_path=config['pretrained_model_path'],
        device=config['device'],
        input_size=len(config['feature_cols']),
        num_tasks=len(config['task_names'])
    )
    
    if model is None:
        logger.error("模型加载失败，终止评估")
        return
    
    # 确保模型处于评估模式
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # 评估模型
    logger.info("开始在完整小样本数据集上评估模型...")
    target_scaler = dataset.target_scaler if config['normalize_data'] else None
    
    results = evaluate_full_dataset(
        model=model,
        dataloader=dataloader,
        task_names=config['task_names'],
        device=config['device'],
        logger=logger,
        target_scaler=target_scaler
    )
    
    # 输出评估结果
    if 'metrics_original' in results:
        metrics = results['metrics_original']
        logger.info("\n========= 完整数据集评估结果（原始数据） =========")
    else:
        metrics = results['metrics_normalized']
        logger.info("\n========= 完整数据集评估结果（标准化数据） =========")
    
    # 使用格式化表格显示结果
    logger.info(format_metrics_table(config['task_names'], metrics['r2'], metrics['rmse'], metrics['rpd']))
    
    # 输出汇总统计
    logger.info(f"\n平均性能: R²={np.mean(metrics['r2']):.4f}, RMSE={np.mean(metrics['rmse']):.4f}, RPD={np.mean(metrics['rpd']):.4f}")
    
    # 详细的参数性能分析
    analyze_parameter_performance(results, config['task_names'], logger)
    
    # 保存结果
    if config['save_predictions']:
        logger.info("保存评估结果...")
        save_detailed_results(results, config['task_names'], df, config['results_dir'])
        
        # 保存标准化器（如果使用了标准化）
        if config['normalize_data']:
            scaler_file = os.path.join(config['results_dir'], 'scalers.pkl')
            joblib.dump({
                'feature_scaler': dataset.feature_scaler,
                'target_scaler': dataset.target_scaler
            }, scaler_file)
            logger.info(f"保存标准化器到: {scaler_file}")
    
    # 最终总结
    logger.info("\n==================== 评估总结 ====================")
    logger.info(f"数据集: {len(df)} 个样本")
    logger.info(f"特征数: {len(config['feature_cols'])}")
    logger.info(f"目标参数数: {len(config['task_names'])}")
    logger.info(f"平均R²: {np.mean(metrics['r2']):.4f}")
    
    # 识别表现最好和最差的参数
    best_param_idx = np.argmax(metrics['r2'])
    worst_param_idx = np.argmin(metrics['r2'])
    logger.info(f"表现最好的参数: {config['task_names'][best_param_idx]} (R²={metrics['r2'][best_param_idx]:.4f})")
    logger.info(f"表现最差的参数: {config['task_names'][worst_param_idx]} (R²={metrics['r2'][worst_param_idx]:.4f})")
    
    logger.info(f"所有结果已保存到: {config['results_dir']}")
    logger.info("==================== 评估完成 ====================")

if __name__ == "__main__":
    main()
