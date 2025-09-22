import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

def evaluate_model(model, data_loader, device, y_mean, y_std):
    """评估函数，计算模型在给定数据集上的性能指标"""
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 收集结果
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
    
    # 合并批次结果
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    # 反标准化
    all_targets = all_targets * y_std + y_mean
    all_predictions = all_predictions * y_std + y_mean
    
    # 计算指标
    r2_scores = r2_score(all_targets, all_predictions, multioutput='raw_values')
    mse = mean_squared_error(all_targets, all_predictions, multioutput='raw_values')
    rmse = np.sqrt(mse)
    
    return all_targets, all_predictions, r2_scores, rmse

def evaluate_single_model(model, target_idx, train_loader, val_loader, test_loader, target_name, device, results_dir=None):
    """评估单个模型在训练集、验证集和测试集上的性能
    
    参数:
        model: 要评估的模型
        target_idx: 目标变量的索引
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        target_name: 目标变量名称
        device: 设备（CPU或GPU）
        results_dir: 结果保存目录
        
    返回:
        包含评估结果的字典
    """
    import numpy as np
    import torch
    from sklearn.metrics import r2_score, mean_squared_error
    
    model.eval()
    
    # 收集训练集预测
    train_preds = []
    train_targets = []
    with torch.no_grad():
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            target = y[:, target_idx].unsqueeze(1)
            output = model(X)
            train_preds.append(output.cpu().numpy())
            train_targets.append(target.cpu().numpy())
    
    train_preds = np.vstack(train_preds)
    train_targets = np.vstack(train_targets)
    
    # 收集验证集预测
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            target = y[:, target_idx].unsqueeze(1)
            output = model(X)
            val_preds.append(output.cpu().numpy())
            val_targets.append(target.cpu().numpy())
    
    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)
    
    # 收集测试集预测
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            target = y[:, target_idx].unsqueeze(1)
            output = model(X)
            test_preds.append(output.cpu().numpy())
            test_targets.append(target.cpu().numpy())
    
    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    
    # 计算评估指标
    train_r2 = r2_score(train_targets, train_preds)
    train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
    
    val_r2 = r2_score(val_targets, val_preds)
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    
    test_r2 = r2_score(test_targets, test_preds)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    
    print(f"\n{target_name} 评估结果:")
    print(f"训练集: R² = {train_r2:.4f}, RMSE = {train_rmse:.4f}")
    print(f"验证集: R² = {val_r2:.4f}, RMSE = {val_rmse:.4f}")
    print(f"测试集: R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}")
    
    # 保存评估结果
    if results_dir is not None:
        import pandas as pd
        import os
        
        results_df = pd.DataFrame({
            'Dataset': ['Train', 'Validation', 'Test'],
            'R2': [train_r2, val_r2, test_r2],
            'RMSE': [train_rmse, val_rmse, test_rmse]
        })
        
        os.makedirs(results_dir, exist_ok=True)
        results_df.to_csv(f"{results_dir}/{target_name}_evaluation.csv", index=False)
        print(f"评估结果已保存至: {results_dir}/{target_name}_evaluation.csv")
    
    # 确保返回预测值和真实值
    return {
        'target': target_name,
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'val_r2': val_r2,
        'val_rmse': val_rmse,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'train_preds': train_preds,     # 确保包含这些
        'train_targets': train_targets,
        'val_preds': val_preds,
        'val_targets': val_targets,
        'test_preds': test_preds,
        'test_targets': test_targets
    }
