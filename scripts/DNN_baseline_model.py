import os
import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error  # 添加缺失的指标计算函数导入
from sklearn.model_selection import train_test_split  # 添加数据集拆分函数
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import datetime
import json
import torch.nn as nn

# 导入自定义模块
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) #e:\pycharm_program\MT_hyperspectral_inversion
sys.path.append(root_dir)
from utils.plot_setting import setfig

from mt_hyperspectral.data.dataset import HSIDataset
from mt_hyperspectral.models.DNN_ensemble import create_ensemble_model
from mt_hyperspectral.utils.metrics import evaluate_single_model
from mt_hyperspectral.training.trainer import safe_format
from mt_hyperspectral.utils.visualization import create_prediction_plot_with_r2, plot_evaluation_summary, create_prediction_plot_from_loader, create_feature_comparison_plot
from mt_hyperspectral.utils.io import save_results, save_model
from mt_hyperspectral.utils.baseset import setup_logger, set_seed, count_parameters

def load_and_evaluate_ensemble(config):
    """
    加载预训练的集成模型并进行评估
    """
    # 设置日志和随机种子
    logger = setup_logger(config)
    set_seed(config['seed'])
    
    # 加载原始数据
    logger.info(f"从 {config['data_path']} 加载数据...")
    try:
        df = pd.read_csv(config['data_path'])
        logger.info(f"成功加载数据，共 {len(df)} 条记录")
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        return
    
    # 从模型配置文件加载信息
    model_config_path = os.path.join(config['model_dir'], 'model_config.json')
    if not os.path.exists(model_config_path):
        logger.error(f"找不到模型配置文件: {model_config_path}")
        return
        
    try:
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        
        # 获取特征列和目标列
        feature_names = model_config.get('feature_names', [])
        target_cols = model_config.get('target_variables', [])
        pca_components = model_config.get('pca_components', -1)
        
        if not feature_names:
            # 如果模型配置中没有特征名称，则使用默认的特征列(4-276)
            feature_cols = list(range(3, 276))  # 从0开始索引，第4-276列
            logger.info(f"使用默认特征列: 第4-276列，共{len(feature_cols)}个特征")
        else:
            feature_cols = [df.columns.get_loc(name) for name in feature_names if name in df.columns]
            logger.info(f"从模型配置加载特征列，共{len(feature_cols)}个特征")
        
        logger.info(f"目标变量: {', '.join(target_cols)}")
        logger.info(f"PCA组件数: {'原始特征' if pca_components == -1 else pca_components}")
    except Exception as e:
        logger.error(f"解析模型配置文件失败: {str(e)}")
        return
    
    # 提取特征和目标变量
    X = df[feature_names].values if feature_names else df.iloc[:, feature_cols].values
    y = df[target_cols].values
    
    logger.info(f"特征数量: {X.shape[1]}, 目标变量数量: {y.shape[1]}")
    logger.info(f"样本数量: {X.shape[0]}")
    
    # 加载归一化参数
    normalization_path = os.path.join(config['model_dir'], 'normalization_params.pkl')
    if not os.path.exists(normalization_path):
        logger.error(f"找不到归一化参数文件: {normalization_path}")
        return
        
    try:
        with open(normalization_path, 'rb') as f:
            normalization_info = pickle.load(f)
        
        X_mean = normalization_info['x_mean']
        X_std = normalization_info['x_std']
        y_mean = normalization_info['y_mean']
        y_std = normalization_info['y_std']
        
        logger.info("成功加载归一化参数")
    except Exception as e:
        logger.error(f"加载归一化参数失败: {str(e)}")
        return
    
    # 如果使用PCA，加载PCA转换器
    if pca_components > 0:
        pca_path = os.path.join(config['model_dir'], 'pca_transformer.pkl')
        if not os.path.exists(pca_path):
            logger.error(f"找不到PCA转换器文件: {pca_path}")
            return
            
        try:
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
            
            # 应用PCA转换
            X = pca.transform(X)
            logger.info(f"应用PCA转换，特征维度: {X.shape[1]}")
        except Exception as e:
            logger.error(f"加载或应用PCA转换器失败: {str(e)}")
            return
    
    # 划分数据集为训练集、验证集和测试集
    logger.info("划分数据集为训练集、验证集和测试集...")
    
    # 检查是否有训练时的索引可用
    if 'train_indices' in config and 'val_indices' in config and 'test_indices' in config:
        # 使用训练时的索引
        logger.info("使用训练时的数据索引进行划分")
        train_indices = config['train_indices']
        val_indices = config['val_indices']
        test_indices = config['test_indices']
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
    else:
        # 使用随机划分
        logger.info("使用随机划分数据集")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=config['seed'])
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=config['seed'])
    
    logger.info(f"数据集划分: 训练集 {X_train.shape[0]} 样本 ({X_train.shape[0]/X.shape[0]:.1%}), "
          f"验证集 {X_val.shape[0]} 样本 ({X_val.shape[0]/X.shape[0]:.1%}), "
          f"测试集 {X_test.shape[0]} 样本 ({X_test.shape[0]/X.shape[0]:.1%})")
    
    # 创建每个数据集的 HSIDataset 对象
    train_dataset = HSIDataset(X_train, y_train, X_mean, X_std, y_mean, y_std)
    val_dataset = HSIDataset(X_val, y_val, X_mean, X_std, y_mean, y_std)
    test_dataset = HSIDataset(X_test, y_test, X_mean, X_std, y_mean, y_std)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 加载模型
    device = config['device']
    models = []
    
    for i, target in enumerate(target_cols):
        # 首先尝试加载完整模型文件
        full_model_path = os.path.join(config['model_dir'], f"{target}_full_model.pth")
        model_path = os.path.join(config['model_dir'], f"{target}_model.pth")
        
        if os.path.exists(full_model_path):
            try:
                # 直接加载整个模型
                model = torch.load(full_model_path, map_location=device)
                logger.info(f"成功加载完整模型: {target}")
                model.eval()
                models.append(model)
                continue
            except Exception as e:
                logger.warning(f"加载完整模型失败: {str(e)}，尝试其他方法")
        
        # 如果无法加载完整模型，尝试使用create_ensemble_model创建结构相同的模型
        if not os.path.exists(model_path):
            logger.error(f"找不到模型文件: {model_path}")
            continue
            
        try:
            # 尝试读取架构文件以获取更精确的模型结构
            arch_file = os.path.join(config['model_dir'], f"{target}_architecture.json")
            if os.path.exists(arch_file):
                with open(arch_file, 'r') as f:
                    arch = json.load(f)
                    
                logger.info(f"使用架构文件创建模型: {target}")
                
                # 根据架构创建模型
                input_dim = arch.get('input_dim', X.shape[1])
                
                # 使用与训练时相同的方式创建模型
                # 这里使用create_ensemble_model来确保架构与训练时一致
                temp_models = create_ensemble_model(input_dim, 1, device)
                model = temp_models[0]  # 获取单个模型
            else:
                # 如果没有架构文件，直接使用与训练时相同的方法创建模型
                logger.info(f"使用标准方法创建模型: {target}")
                temp_models = create_ensemble_model(X.shape[1], 1, device)
                model = temp_models[0]  # 获取单个模型
                
            # 加载权重
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)
            logger.info(f"成功加载模型权重: {target}")
        except Exception as e:
            logger.error(f"加载模型 {target} 失败: {str(e)}")
            continue
    
    if not models:
        logger.error("没有成功加载任何模型")
        return
    
    # 评估模型
    logger.info("开始评估模型...")
    
    # 对每个数据集分别进行预测
    train_preds = []
    train_targets = []
    val_preds = []
    val_targets = []
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        # 训练集评估
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            batch_preds = []
            for model in models:
                pred = model(X_batch)
                batch_preds.append(pred.cpu().numpy())
            train_preds.append(np.hstack(batch_preds))
            train_targets.append(y_batch.numpy())
        
        # 验证集评估
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            batch_preds = []
            for model in models:
                pred = model(X_batch)
                batch_preds.append(pred.cpu().numpy())
            val_preds.append(np.hstack(batch_preds))
            val_targets.append(y_batch.numpy())
        
        # 测试集评估
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            batch_preds = []
            for model in models:
                pred = model(X_batch)
                batch_preds.append(pred.cpu().numpy())
            test_preds.append(np.hstack(batch_preds))
            test_targets.append(y_batch.numpy())
    
    # 合并批次结果
    train_preds = np.vstack(train_preds)
    train_targets = np.vstack(train_targets)
    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)
    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    
    # 计算并打印评估指标，使用与训练脚本相同的格式
    feature_name = "原始特征" if pca_components == -1 else f"{pca_components}个PCA组件"
    
    logger.info(f"{'='*20} 详细评估指标 {'='*20}")
    logger.info(f"组件: {feature_name}")
    logger.info(f"{'参数':<10}{'数据集':<12}{'R²':<10}{'RMSE':<10}{'MAE':<10}{'解释方差':<12}")
    logger.info(f"{'-'*60}")
    
    # 初始化存储评估指标的字典
    metrics = {
        'train_r2': [], 'train_rmse': [], 'train_mae': [], 'train_ev': [],
        'val_r2': [], 'val_rmse': [], 'val_mae': [], 'val_ev': [],
        'test_r2': [], 'test_rmse': [], 'test_mae': [], 'test_ev': []
    }
    
    # 初始化详细指标列表，用于保存到CSV
    detailed_metrics = []
    
    # 计算每个参数的指标
    for i, target in enumerate(target_cols):
        # 训练集指标
        train_r2 = r2_score(train_targets[:, i], train_preds[:, i])
        train_rmse = np.sqrt(mean_squared_error(train_targets[:, i], train_preds[:, i]))
        train_mae = np.mean(np.abs(train_targets[:, i] - train_preds[:, i]))
        train_ev = 1 - np.var(train_targets[:, i] - train_preds[:, i]) / np.var(train_targets[:, i])
        
        # 验证集指标
        val_r2 = r2_score(val_targets[:, i], val_preds[:, i])
        val_rmse = np.sqrt(mean_squared_error(val_targets[:, i], val_preds[:, i]))
        val_mae = np.mean(np.abs(val_targets[:, i] - val_preds[:, i]))
        val_ev = 1 - np.var(val_targets[:, i] - val_preds[:, i]) / np.var(val_targets[:, i])
        
        # 测试集指标
        test_r2 = r2_score(test_targets[:, i], test_preds[:, i])
        test_rmse = np.sqrt(mean_squared_error(test_targets[:, i], test_preds[:, i]))
        test_mae = np.mean(np.abs(test_targets[:, i] - test_preds[:, i]))
        test_ev = 1 - np.var(test_targets[:, i] - test_preds[:, i]) / np.var(test_targets[:, i])
        
        # 存储指标
        metrics['train_r2'].append(train_r2)
        metrics['train_rmse'].append(train_rmse)
        metrics['train_mae'].append(train_mae)
        metrics['train_ev'].append(train_ev)
        
        metrics['val_r2'].append(val_r2)
        metrics['val_rmse'].append(val_rmse)
        metrics['val_mae'].append(val_mae)
        metrics['val_ev'].append(val_ev)
        
        metrics['test_r2'].append(test_r2)
        metrics['test_rmse'].append(test_rmse)
        metrics['test_mae'].append(test_mae)
        metrics['test_ev'].append(test_ev)
        
        # 打印结果
        logger.info(f"{target:<10}{'训练集':<12}{train_r2:<10.4f}{train_rmse:<10.4f}{train_mae:<10.4f}{train_ev:<12.4f}")
        logger.info(f"{'':<10}{'验证集':<12}{val_r2:<10.4f}{val_rmse:<10.4f}{val_mae:<10.4f}{val_ev:<12.4f}")
        logger.info(f"{'':<10}{'测试集':<12}{test_r2:<10.4f}{test_rmse:<10.4f}{test_mae:<10.4f}{test_ev:<12.4f}")
        logger.info(f"{'-'*60}")
        
        # 添加到详细指标列表
        detailed_metrics.append({
            'Parameter': target,
            'Dataset': 'Train',
            'R2': train_r2,
            'RMSE': train_rmse,
            'MAE': train_mae,
            'Explained_Variance': train_ev
        })
        detailed_metrics.append({
            'Parameter': target,
            'Dataset': 'Validation',
            'R2': val_r2,
            'RMSE': val_rmse,
            'MAE': val_mae,
            'Explained_Variance': val_ev
        })
        detailed_metrics.append({
            'Parameter': target,
            'Dataset': 'Test',
            'R2': test_r2,
            'RMSE': test_rmse,
            'MAE': test_mae,
            'Explained_Variance': test_ev
        })
    
    # 保存详细评估指标到CSV文件
    metrics_dir = os.path.join(config['results_dir'], 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    if pca_components == -1:
        metrics_file = os.path.join(metrics_dir, 'original_features_detailed_metrics.csv')
    else:
        metrics_file = os.path.join(metrics_dir, f'pca_{pca_components}_detailed_metrics.csv')
    
    pd.DataFrame(detailed_metrics).to_csv(metrics_file, index=False)
    logger.info(f"详细评估指标已保存到: {metrics_file}")
    
    # 计算整体平均指标
    avg_train_r2 = np.mean(metrics['train_r2'])
    avg_val_r2 = np.mean(metrics['val_r2'])
    avg_test_r2 = np.mean(metrics['test_r2'])
    
    avg_train_rmse = np.mean(metrics['train_rmse'])
    avg_val_rmse = np.mean(metrics['val_rmse'])
    avg_test_rmse = np.mean(metrics['test_rmse'])
    
    logger.info(f"整体平均指标:")
    logger.info(f"R² - 训练集: {avg_train_r2:.4f}, 验证集: {avg_val_r2:.4f}, 测试集: {avg_test_r2:.4f}")
    logger.info(f"RMSE - 训练集: {avg_train_rmse:.4f}, 验证集: {avg_val_rmse:.4f}, 测试集: {avg_test_rmse:.4f}")
    
    # 保存评估结果
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # 创建全局预测结果汇总 - 合并训练、验证和测试集结果
    all_preds = np.vstack([train_preds, val_preds, test_preds])
    all_targets = np.vstack([train_targets, val_targets, test_targets])
    
    # 使用上面已计算的指标，不再需要重新计算
    r2_scores = metrics['test_r2']  # 使用测试集的R²值
    rmse_values = metrics['test_rmse']  # 使用测试集的RMSE值
    mae_values = metrics['test_mae']  # 使用测试集的MAE值
    ev_values = metrics['test_ev']  # 使用测试集的解释方差
    
    # 计算平均指标
    mean_r2 = avg_test_r2
    mean_rmse = avg_test_rmse
    mean_mae = np.mean(metrics['test_mae'])
    mean_ev = np.mean(metrics['test_ev'])
    
    # 保存预测结果 - 为每个数据集单独创建DataFrame
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    # 反归一化以获取原始单位的预测值
    denorm_train_preds = train_preds * y_std + y_mean
    denorm_train_targets = train_targets * y_std + y_mean
    denorm_val_preds = val_preds * y_std + y_mean
    denorm_val_targets = val_targets * y_std + y_mean
    denorm_test_preds = test_preds * y_std + y_mean
    denorm_test_targets = test_targets * y_std + y_mean
    
    # 添加每个任务的真实值和预测值 - 分别添加到对应的DataFrame
    for i, target in enumerate(target_cols):
        if i >= train_preds.shape[1]:
            continue
            
        # 训练集结果
        train_df[f"{target}_true"] = denorm_train_targets[:, i]
        train_df[f"{target}_pred"] = denorm_train_preds[:, i]
        train_df[f"{target}_error"] = denorm_train_preds[:, i] - denorm_train_targets[:, i]
        
        # 验证集结果
        val_df[f"{target}_true"] = denorm_val_targets[:, i]
        val_df[f"{target}_pred"] = denorm_val_preds[:, i]
        val_df[f"{target}_error"] = denorm_val_preds[:, i] - denorm_val_targets[:, i]
        
        # 测试集结果
        test_df[f"{target}_true"] = denorm_test_targets[:, i]
        test_df[f"{target}_pred"] = denorm_test_preds[:, i]
        test_df[f"{target}_error"] = denorm_test_preds[:, i] - denorm_test_targets[:, i]
    
    # 分别保存三个数据集的结果
    train_df.to_csv(os.path.join(config['results_dir'], 'train_predictions.csv'), index=False)
    val_df.to_csv(os.path.join(config['results_dir'], 'val_predictions.csv'), index=False)
    test_df.to_csv(os.path.join(config['results_dir'], 'test_predictions.csv'), index=False)
    logger.info(f"评估结果已保存到 {config['results_dir']}")
    
    # 创建可视化 - 使用测试集数据
    logger.info("生成评估可视化...")
    
    # 创建结果列表，用于汇总绘图
    results_for_summary = []
    
    # 为每个目标变量创建散点图
    for i, target in enumerate(target_cols):
        if i >= test_preds.shape[1]:
            continue
            
        # 创建结果字典，用于plot_evaluation_summary
        result_dict = {
            'target': target,
            'train_preds': train_preds[:, i:i+1],
            'train_targets': train_targets[:, i:i+1],
            'val_preds': val_preds[:, i:i+1],
            'val_targets': val_targets[:, i:i+1],
            'test_preds': test_preds[:, i:i+1],
            'test_targets': test_targets[:, i:i+1],
            'train_r2': metrics['train_r2'][i],
            'val_r2': metrics['val_r2'][i],
            'test_r2': metrics['test_r2'][i],
            'train_rmse': metrics['train_rmse'][i],
            'val_rmse': metrics['val_rmse'][i],
            'test_rmse': metrics['test_rmse'][i]
        }
        results_for_summary.append(result_dict)
        
        try:
            # 使用与ensemble_evaluate_model copy.py相同的绘图方法
            model = models[i] if isinstance(models[i], nn.Module) else models[i]
            
            # 第一种方法：使用标准可视化函数，添加try-except捕获CUDA错误
            try:
                create_prediction_plot_with_r2(
                    model, i, test_dataset, None, 
                    target, device, config['results_dir'],
                    metrics['test_r2'][i], None, 
                    format='PDF', plot_train=False
                )
                logger.info(f"成功创建{target}的预测图")
            except RuntimeError as cuda_err:
                if 'CUDA' in str(cuda_err):
                    logger.warning(f"标准方法CUDA错误: {str(cuda_err)}")
                    # 如果是CUDA错误，尝试临时切换到CPU处理
                    logger.info(f"尝试切换到CPU处理可视化...")
                    temp_model = model.cpu()
                    temp_device = torch.device('cpu')
                    try:
                        create_prediction_plot_with_r2(
                            temp_model, i, test_dataset, None, 
                            target, temp_device, config['results_dir'],
                            metrics['test_r2'][i], None, 
                            format='PDF', plot_train=False
                        )
                        logger.info(f"使用CPU成功创建{target}的预测图")
                        # 处理完成后将模型移回GPU
                        model = model.to(device)
                    except Exception as cpu_e:
                        logger.warning(f"CPU处理也失败: {str(cpu_e)}")
                        model = model.to(device)  # 确保模型被移回GPU
                        # 尝试备选方法
                        raise
            except Exception as e:
                logger.warning(f"警告: 尝试标准方法创建 {target} 的预测图时出错: {str(e)}")
                
                # 第二种完全独立的方法：使用DataLoader
                try:
                    logger.info(f"尝试备选方法为{target}创建预测图")
                    
                    # 确定标准化参数
                    if isinstance(y_mean, (np.ndarray, list)) and len(y_mean) > i:
                        y_mean_val = y_mean[i]
                        y_std_val = y_std[i]
                    else:
                        y_mean_val = y_mean
                        y_std_val = y_std
                        
                    # 调用visualization.py中的函数创建预测图
                    plot_path = create_prediction_plot_from_loader(
                        model=model,
                        idx=i,
                        test_loader=test_loader,
                        target=target,
                        device=device,
                        output_dir=config['results_dir'],
                        test_r2=metrics['test_r2'][i],
                        y_mean=y_mean_val,
                        y_std=y_std_val,
                        format='PDF'
                    )
                    logger.info(f"已使用备选方法成功保存{target}的预测图到 {plot_path}")
                except Exception as e2:
                    logger.warning(f"备选预测图方法也失败: {str(e2)}")
                    logger.warning(f"跳过{target}的可视化，继续评估")
        except Exception as e:
            logger.warning(f"为{target}创建预测图时出错: {str(e)}")
            # 退回到原始绘图方法
            try:
                setfig(column=1, x=3, y=3)
                plt.figure(figsize=(5, 5))
                plt.scatter(denorm_test_targets[:, i], denorm_test_preds[:, i], alpha=0.5)
                
                # 添加对角线
                min_val = min(denorm_test_targets[:, i].min(), denorm_test_preds[:, i].min())
                max_val = max(denorm_test_targets[:, i].max(), denorm_test_preds[:, i].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                plt.title(f'{target} (R² = {r2_scores[i]:.4f}, RMSE = {rmse_values[i]:.4f})')
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                
                # 保存为PDF格式
                plt.savefig(
                    os.path.join(config['results_dir'], f'{target}_scatter.pdf'),
                    format='pdf', 
                    bbox_inches='tight', 
                    transparent=True
                )
                plt.close()
            except Exception as fallback_e:
                logger.error(f"无法创建{target}的预测图，即使使用备选方法: {str(fallback_e)}")

    # 使用汇总函数创建所有评估图表
    try:
        plot_evaluation_summary(results_for_summary, config['results_dir'], format='PDF', include_train=True)
        logger.info(f"成功创建评估汇总图表")
    except Exception as e:
        logger.warning(f"创建评估汇总图表时出错: {str(e)}")
        
        # 退回到创建R²条形图
        try:
            setfig(column=1, x=5, y=3)
            plt.figure(figsize=(10, 6))
            bars = plt.bar(target_cols[:len(r2_scores)], r2_scores)
            
            # 为条形图添加颜色
            for i, bar in enumerate(bars):
                r2 = r2_scores[i]
                if r2 > 0.5:
                    bar.set_color('green')
                elif r2 > 0.3:
                    bar.set_color('lightgreen')
                elif r2 > 0:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
            plt.title(f'R² Scores for Each Target Variable')
            plt.ylabel('R² Score')
            plt.xticks(rotation=45, ha='right')
            
            # 添加值标签
            for i, v in enumerate(r2_scores):
                plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=8)
            
            plt.savefig(
                os.path.join(config['results_dir'], 'r2_scores.pdf'),
                format='pdf',
                bbox_inches='tight',
                transparent=True
            )
            plt.close()
        except Exception as fallback_e:
            logger.error(f"无法创建R²条形图: {str(fallback_e)}")
    
    # 保存评估配置
    config['evaluation_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config['target_variables'] = target_cols
    config['pca_components'] = pca_components
    config['average_metrics'] = {
        'mean_r2': float(mean_r2),
        'mean_rmse': float(mean_rmse),
        'mean_mae': float(mean_mae),
        'mean_ev': float(mean_ev)
    }
    
    with open(os.path.join(config['results_dir'], 'evaluation_config.json'), 'w') as f:
        # 处理不可序列化的对象
        config_copy = {k: v for k, v in config.items() if k != 'device'}
        json.dump(config_copy, f, indent=4)
    
    # 修改返回值以包含更多信息
    return {
        'train_r2': metrics['train_r2'],
        'val_r2': metrics['val_r2'],
        'test_r2': metrics['test_r2'],
        'train_rmse': metrics['train_rmse'],
        'val_rmse': metrics['val_rmse'],
        'test_rmse': metrics['test_rmse'],
        'mean_train_r2': avg_train_r2,
        'mean_val_r2': avg_val_r2,
        'mean_test_r2': avg_test_r2,
        'mean_train_rmse': avg_train_rmse,
        'mean_val_rmse': avg_val_rmse,
        'mean_test_rmse': avg_test_rmse,
        'train_predictions': train_preds,
        'val_predictions': val_preds,
        'test_predictions': test_preds,
        'train_targets': train_targets,
        'val_targets': val_targets,
        'test_targets': test_targets
    }

def main():
    # 配置参数
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'data_path': 'data/Tomato_subsets/dataset_all_per_class.csv',
        'model_dir': 'results/MDNN/MT/models/original_features',
        # 'data_path': 'data/Rice_subsets/rice dataset_all_per_class.csv',#水稻
        # 'model_dir': 'results/MDNN/Rice/models/original_features',
        'batch_size': 32,
        'seed': 42,
        'results_dir': 'results/ensemble_evaluation_allp',
    }
    
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config['results_dir'] = os.path.join(config['results_dir'], timestamp)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # 设置logger
    logger = setup_logger(config)
    logger.info("开始评估模型...")
    logger.info(f"使用设备: {config['device']}")
    
    # 尝试加载模型训练时的数据索引
    data_indices_path = os.path.join(config['model_dir'], 'data_indices.npz')
    if os.path.exists(data_indices_path):
        logger.info(f"正在加载训练时使用的数据集索引: {data_indices_path}")
        try:
            data_indices = np.load(data_indices_path)
            config['train_indices'] = data_indices['train_indices']
            config['val_indices'] = data_indices['val_indices']
            config['test_indices'] = data_indices['test_indices']
            logger.info(f"成功加载数据索引，训练集: {len(config['train_indices'])} 样本, "
                    f"验证集: {len(config['val_indices'])} 样本, "
                    f"测试集: {len(config['test_indices'])} 样本")
        except Exception as e:
            logger.warning(f"加载数据索引失败: {str(e)}，将使用默认的数据划分方法")
    else:
        logger.info("未找到训练时的数据索引文件，将使用默认的数据划分方法")
    
    # 加载并评估模型
    results = load_and_evaluate_ensemble(config)
    
    # 更新打印的结果信息
    if results:
        print("评估完成!")
        print(f"平均R² - 训练集: {results['mean_train_r2']:.4f}, 验证集: {results['mean_val_r2']:.4f}, 测试集: {results['mean_test_r2']:.4f}")
        print(f"平均RMSE - 训练集: {results['mean_train_rmse']:.4f}, 验证集: {results['mean_val_rmse']:.4f}, 测试集: {results['mean_test_rmse']:.4f}")
        print(f"详细结果已保存到: {config['results_dir']}")
    else:
        print("评估失败，请查看日志了解详情。")

if __name__ == "__main__":
    main()