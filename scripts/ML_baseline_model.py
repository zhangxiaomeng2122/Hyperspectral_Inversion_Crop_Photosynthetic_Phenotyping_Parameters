import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import datetime
import pickle
import json
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

# 导入自定义模块
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from utils.plot_setting import setfig
from mt_hyperspectral.utils.baseset import setup_logger, set_seed
from mt_hyperspectral.models.ML_baseline import (
    PLSRMultiOutputModel, SVRMultiOutputModel, RFMultiOutputModel, 
    XGBMultiOutputModel, MLPMultiOutputModel, MPLSRMultiOutputModel, find_optimal_components
)

def train_plsr_model(config, logger=None):
    """
    训练PLSR多目标输出模型
    """
    # 记录开始时间
    start_time = time.time()
    
    # 加载数据
    logger.info("开始加载数据...")
    if not os.path.exists(config['data_path']):
        logger.error(f"错误: 找不到文件 {config['data_path']}")
        return None
    
    df = pd.read_csv(config['data_path'])
    logger.info(f"成功加载数据集，共 {len(df)} 条记录")
    
    # 提取特征和目标变量
    id_cols = df.columns[:3].tolist()
    logger.info(f"样本标识列（前3列）: {id_cols}")
    
    feature_cols = df.columns[3:276].tolist()
    logger.info(f"特征列数量: {len(feature_cols)}")
    
    # 根据作物类型设置目标变量
    if config['crop_type'] == 'tomato':
        target_cols = ['RUE', 'Pn', 'Gs', 'Ci', 'Tr', 'Ci-Ca', 'WUE', 'iWUE', 'α', 'Pmax', 'θ', 'Rd', 'SPAD', 'Ic', 'LAW']
    else:  # rice
        target_cols = ['SPAD', 'Pn', 'LNC', 'Chl-a', 'Chl-b', 'LAW', 'Cx', 'Chl']
    
    # 检查目标变量列是否在数据集中存在
    available_target_cols = [col for col in target_cols if col in df.columns]
    if len(available_target_cols) < len(target_cols):
        missing_cols = [col for col in target_cols if col not in available_target_cols]
        logger.warning(f"以下目标变量列在数据集中不存在: {missing_cols}")
        logger.info(f"将使用可用的目标变量列: {available_target_cols}")
        target_cols = available_target_cols
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    logger.info(f"特征数量: {X.shape[1]}, 目标变量数量: {y.shape[1]}")
    logger.info(f"样本数量: {X.shape[0]}")
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logger.info(f"数据集划分: 训练集 {X_train.shape[0]} 样本 ({X_train.shape[0]/X.shape[0]:.1%}), "
                f"验证集 {X_val.shape[0]} 样本 ({X_val.shape[0]/X.shape[0]:.1%}), "
                f"测试集 {X_test.shape[0]} 样本 ({X_test.shape[0]/X.shape[0]:.1%})")
    
    # 找到最佳PLSR组件数
    if config.get('optimize_components', True):
        n_components, comp_details = find_optimal_components(
            X_train, y_train, target_cols, 
            max_components=config.get('max_components', 30),
            cv=config.get('cv_folds', 5),
            logger=logger
        )
    else:
        n_components = config.get('n_components', 15)
        logger.info(f"使用预设的PLSR组件数: {n_components}")
    
    # 创建并训练PLSR模型
    logger.info(f"开始训练PLSR模型，使用 {n_components} 个组件...")
    model = PLSRMultiOutputModel(n_components=n_components, max_iter=1000)
    model.fit(X_train, y_train, target_cols)
    
    # 评估模型
    train_results = model.evaluate(X_train, y_train, target_cols)
    val_results = model.evaluate(X_val, y_val, target_cols)
    test_results = model.evaluate(X_test, y_test, target_cols)
    
    # 计算平均指标
    avg_train_r2 = np.mean(train_results['r2_scores'])
    avg_val_r2 = np.mean(val_results['r2_scores'])
    avg_test_r2 = np.mean(test_results['r2_scores'])
    
    avg_train_rmse = np.mean(train_results['rmse_values'])
    avg_val_rmse = np.mean(val_results['rmse_values'])
    avg_test_rmse = np.mean(test_results['rmse_values'])
    
    avg_train_rpd = np.mean(train_results['rpd_values'])
    avg_val_rpd = np.mean(val_results['rpd_values'])
    avg_test_rpd = np.mean(test_results['rpd_values'])
    
    logger.info(f"\nAverage R²: Train={avg_train_r2:.4f}, Validation={avg_val_r2:.4f}, Test={avg_test_r2:.4f}")
    
    # 显示三个数据集性能对比
    logger.info(f"\n========= 三个数据集性能对比 =========")
    logger.info(f"{'数据集':<10}{'平均R²':<12}{'平均RMSE':<12}{'平均RPD':<12}")
    logger.info(f"训练集      {avg_train_r2:.4f}     {avg_train_rmse:.4f}     {avg_train_rpd:.4f}")
    logger.info(f"验证集      {avg_val_r2:.4f}     {avg_val_rmse:.4f}     {avg_val_rpd:.4f}")
    logger.info(f"测试集      {avg_test_r2:.4f}     {avg_test_rmse:.4f}     {avg_test_rpd:.4f}")
    logger.info(f"========================================")
    
    # 显示详细任务评估指标
    logger.info(f"\n========= 详细任务评估指标 =========")
    logger.info(f"{'参数':<10}{'数据集':<13}{'R²':<10}{'RMSE':<10}{'MAE':<10}{'解释方差':<14}")
    logger.info(f"------------------------------------------------------------")
    
    for i, target in enumerate(target_cols):
        # 获取当前参数的训练集指标
        train_r2 = train_results['r2_scores'][i]
        train_rmse = train_results['rmse_values'][i]
        train_mae = train_results['mae_values'][i]
        
        # 计算解释方差（explained variance）
        train_true = train_results['targets'][:, i]
        train_pred = train_results['predictions'][:, i]
        train_ev = 1 - np.var(train_true - train_pred) / np.var(train_true)
        
        # 获取当前参数的验证集指标
        val_r2 = val_results['r2_scores'][i]
        val_rmse = val_results['rmse_values'][i]
        val_mae = val_results['mae_values'][i]
        
        val_true = val_results['targets'][:, i]
        val_pred = val_results['predictions'][:, i]
        val_ev = 1 - np.var(val_true - val_pred) / np.var(val_true)
        
        # 获取当前参数的测试集指标
        test_r2 = test_results['r2_scores'][i]
        test_rmse = test_results['rmse_values'][i]
        test_mae = test_results['mae_values'][i]
        
        test_true = test_results['targets'][:, i]
        test_pred = test_results['predictions'][:, i]
        test_ev = 1 - np.var(test_true - test_pred) / np.var(test_true)
        
        # 显示指标
        logger.info(f"{target:<10}{'训练集':<13}{train_r2:.4f}    {train_rmse:.4f}    {train_mae:.4f}    {train_ev:.4f}      ")
        logger.info(f"{'':<10}{'验证集':<13}{val_r2:.4f}    {val_rmse:.4f}    {val_mae:.4f}    {val_ev:.4f}      ")
        logger.info(f"{'':<10}{'测试集':<13}{test_r2:.4f}    {test_rmse:.4f}    {test_mae:.4f}    {test_ev:.4f}      ")
        logger.info(f"------------------------------------------------------------")
    
    # 显示各参数测试集性能
    logger.info(f"\n=================== 各参数测试集性能 ===================")
    logger.info(f"{'参数':<12}{'R²':<12}{'RMSE':<12}{'RPD':<12}")
    logger.info(f"-------------------------------------------------")
    
    for i, target in enumerate(target_cols):
        test_r2 = test_results['r2_scores'][i]
        test_rmse = test_results['rmse_values'][i]
        test_rpd = test_results['rpd_values'][i]
        logger.info(f"{target:<12}{test_r2:.4f}     {test_rmse:.4f}     {test_rpd:.4f}")
    
    logger.info(f"-------------------------------------------------")
    logger.info(f"{'平均':<12}{avg_test_r2:.4f}     {avg_test_rmse:.4f}     {avg_test_rpd:.4f}")
    logger.info(f"=================================================")
    logger.info(f"训练完成! 最佳验证集R²: {avg_val_r2:.4f}, 测试集R²: {avg_test_r2:.4f}")
    
    # 保存模型和结果
    save_results_and_model(config, model, target_cols, train_results, val_results, test_results, n_components, logger)
    
    # 计算并记录训练时间
    end_time = time.time()
    training_duration = end_time - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"\nPLSR模型训练完成，耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    return model, train_results, val_results, test_results

def train_svr_model(config, logger=None):
    """
    训练SVR多目标输出模型
    """
    # 记录开始时间
    start_time = time.time()
    
    # 加载数据
    logger.info("开始加载数据...")
    if not os.path.exists(config['data_path']):
        logger.error(f"错误: 找不到文件 {config['data_path']}")
        return None
    
    df = pd.read_csv(config['data_path'])
    logger.info(f"成功加载数据集，共 {len(df)} 条记录")
    
    # 提取特征和目标变量
    id_cols = df.columns[:3].tolist()
    logger.info(f"样本标识列（前3列）: {id_cols}")
    
    feature_cols = df.columns[3:276].tolist()
    logger.info(f"特征列数量: {len(feature_cols)}")
    
    # 根据作物类型设置目标变量
    if config['crop_type'] == 'tomato':
        target_cols = ['RUE', 'Pn', 'Gs', 'Ci', 'Tr', 'Ci-Ca', 'WUE', 'iWUE', 'α', 'Pmax', 'θ', 'Rd', 'SPAD', 'Ic', 'LAW']
    else:  # rice
        target_cols = ['SPAD', 'Pn', 'LNC', 'Chl-a', 'Chl-b', 'LAW', 'Cx', 'Chl']
    
    # 检查目标变量列是否在数据集中存在
    available_target_cols = [col for col in target_cols if col in df.columns]
    if len(available_target_cols) < len(target_cols):
        missing_cols = [col for col in target_cols if col not in available_target_cols]
        logger.warning(f"以下目标变量列在数据集中不存在: {missing_cols}")
        logger.info(f"将使用可用的目标变量列: {available_target_cols}")
        target_cols = available_target_cols
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    logger.info(f"特征数量: {X.shape[1]}, 目标变量数量: {y.shape[1]}")
    logger.info(f"样本数量: {X.shape[0]}")
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logger.info(f"数据集划分: 训练集 {X_train.shape[0]} 样本 ({X_train.shape[0]/X.shape[0]:.1%}), "
                f"验证集 {X_val.shape[0]} 样本 ({X_val.shape[0]/X.shape[0]:.1%}), "
                f"测试集 {X_test.shape[0]} 样本 ({X_test.shape[0]/X.shape[0]:.1%})")
    
    # 创建SVR模型
    logger.info("开始训练SVR模型...")
    
    # 读取SVR参数
    svr_params = config.get('svr_params', {})
    kernel = svr_params.get('kernel', 'rbf')
    C = svr_params.get('C', 10.0)
    epsilon = svr_params.get('epsilon', 0.1)
    gamma = svr_params.get('gamma', 'scale')
    
    # 创建并训练SVR模型
    model = SVRMultiOutputModel(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
    model.fit(X_train, y_train, target_cols)
    
    # 评估模型
    train_results = model.evaluate(X_train, y_train, target_cols)
    val_results = model.evaluate(X_val, y_val, target_cols)
    test_results = model.evaluate(X_test, y_test, target_cols)
    
    # 计算平均指标
    avg_train_r2 = np.mean(train_results['r2_scores'])
    avg_val_r2 = np.mean(val_results['r2_scores'])
    avg_test_r2 = np.mean(test_results['r2_scores'])
    
    avg_train_rmse = np.mean(train_results['rmse_values'])
    avg_val_rmse = np.mean(val_results['rmse_values'])
    avg_test_rmse = np.mean(test_results['rmse_values'])
    
    avg_train_rpd = np.mean(train_results['rpd_values'])
    avg_val_rpd = np.mean(val_results['rpd_values'])
    avg_test_rpd = np.mean(test_results['rpd_values'])
    
    logger.info(f"\nAverage R²: Train={avg_train_r2:.4f}, Validation={avg_val_r2:.4f}, Test={avg_test_r2:.4f}")
    
    # 显示三个数据集性能对比
    logger.info(f"\n========= 三个数据集性能对比 =========")
    logger.info(f"{'数据集':<10}{'平均R²':<12}{'平均RMSE':<12}{'平均RPD':<12}")
    logger.info(f"训练集      {avg_train_r2:.4f}     {avg_train_rmse:.4f}     {avg_train_rpd:.4f}")
    logger.info(f"验证集      {avg_val_r2:.4f}     {avg_val_rmse:.4f}     {avg_val_rpd:.4f}")
    logger.info(f"测试集      {avg_test_r2:.4f}     {avg_test_rmse:.4f}     {avg_test_rpd:.4f}")
    logger.info(f"========================================")
    
    # 显示详细任务评估指标
    logger.info(f"\n========= 详细任务评估指标 =========")
    logger.info(f"{'参数':<10}{'数据集':<13}{'R²':<10}{'RMSE':<10}{'MAE':<10}{'解释方差':<14}")
    logger.info(f"------------------------------------------------------------")
    
    for i, target in enumerate(target_cols):
        # 获取当前参数的训练集指标
        train_r2 = train_results['r2_scores'][i]
        train_rmse = train_results['rmse_values'][i]
        train_mae = train_results['mae_values'][i]
        
        # 计算解释方差（explained variance）
        train_true = train_results['targets'][:, i]
        train_pred = train_results['predictions'][:, i]
        train_ev = 1 - np.var(train_true - train_pred) / np.var(train_true)
        
        # 获取当前参数的验证集指标
        val_r2 = val_results['r2_scores'][i]
        val_rmse = val_results['rmse_values'][i]
        val_mae = val_results['mae_values'][i]
        
        val_true = val_results['targets'][:, i]
        val_pred = val_results['predictions'][:, i]
        val_ev = 1 - np.var(val_true - val_pred) / np.var(val_true)
        
        # 获取当前参数的测试集指标
        test_r2 = test_results['r2_scores'][i]
        test_rmse = test_results['rmse_values'][i]
        test_mae = test_results['mae_values'][i]
        
        test_true = test_results['targets'][:, i]
        test_pred = test_results['predictions'][:, i]
        test_ev = 1 - np.var(test_true - test_pred) / np.var(test_true)
        
        # 显示指标
        logger.info(f"{target:<10}{'训练集':<13}{train_r2:.4f}    {train_rmse:.4f}    {train_mae:.4f}    {train_ev:.4f}      ")
        logger.info(f"{'':<10}{'验证集':<13}{val_r2:.4f}    {val_rmse:.4f}    {val_mae:.4f}    {val_ev:.4f}      ")
        logger.info(f"{'':<10}{'测试集':<13}{test_r2:.4f}    {test_rmse:.4f}    {test_mae:.4f}    {test_ev:.4f}      ")
        logger.info(f"------------------------------------------------------------")
    
    # 显示各参数测试集性能
    logger.info(f"\n=================== 各参数测试集性能 ===================")
    logger.info(f"{'参数':<12}{'R²':<12}{'RMSE':<12}{'RPD':<12}")
    logger.info(f"-------------------------------------------------")
    
    for i, target in enumerate(target_cols):
        test_r2 = test_results['r2_scores'][i]
        test_rmse = test_results['rmse_values'][i]
        test_rpd = test_results['rpd_values'][i]
        logger.info(f"{target:<12}{test_r2:.4f}     {test_rmse:.4f}     {test_rpd:.4f}")
    
    logger.info(f"-------------------------------------------------")
    logger.info(f"{'平均':<12}{avg_test_r2:.4f}     {avg_test_rmse:.4f}     {avg_test_rpd:.4f}")
    logger.info(f"=================================================")
    logger.info(f"训练完成! 最佳验证集R²: {avg_val_r2:.4f}, 测试集R²: {avg_test_r2:.4f}")
    
    # 保存模型和结果
    # 注意：SVR模型不需要n_components参数，应该传入None
    save_results_and_model(config, model, target_cols, train_results, val_results, test_results, None, logger)
    
    # 计算并记录训练时间
    end_time = time.time()
    training_duration = end_time - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"\nSVR模型训练完成，耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    return model, train_results, val_results, test_results

def train_rf_model(config, logger=None):
    """
    训练随机森林多目标输出模型
    """
    # 记录开始时间
    start_time = time.time()
    
    # 加载数据
    logger.info("开始加载数据...")
    if not os.path.exists(config['data_path']):
        logger.error(f"错误: 找不到文件 {config['data_path']}")
        return None
    
    df = pd.read_csv(config['data_path'])
    logger.info(f"成功加载数据集，共 {len(df)} 条记录")
    
    # 提取特征和目标变量
    id_cols = df.columns[:3].tolist()
    logger.info(f"样本标识列（前3列）: {id_cols}")
    
    feature_cols = df.columns[3:276].tolist()
    logger.info(f"特征列数量: {len(feature_cols)}")
    
    # 根据作物类型设置目标变量
    if config['crop_type'] == 'tomato':
        target_cols = ['RUE', 'Pn', 'Gs', 'Ci', 'Tr', 'Ci-Ca', 'WUE', 'iWUE', 'α', 'Pmax', 'θ', 'Rd', 'SPAD', 'Ic', 'LAW']
    else:  # rice
        target_cols = ['SPAD', 'Pn', 'LNC', 'Chl-a', 'Chl-b', 'LAW', 'Cx', 'Chl']
    
    # 检查目标变量列是否在数据集中存在
    available_target_cols = [col for col in target_cols if col in df.columns]
    if len(available_target_cols) < len(target_cols):
        missing_cols = [col for col in target_cols if col not in available_target_cols]
        logger.warning(f"以下目标变量列在数据集中不存在: {missing_cols}")
        logger.info(f"将使用可用的目标变量列: {available_target_cols}")
        target_cols = available_target_cols
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    logger.info(f"特征数量: {X.shape[1]}, 目标变量数量: {y.shape[1]}")
    logger.info(f"样本数量: {X.shape[0]}")
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logger.info(f"数据集划分: 训练集 {X_train.shape[0]} 样本 ({X_train.shape[0]/X.shape[0]:.1%}), "
                f"验证集 {X_val.shape[0]} 样本 ({X_val.shape[0]/X.shape[0]:.1%}), "
                f"测试集 {X_test.shape[0]} 样本 ({X_test.shape[0]/X.shape[0]:.1%})")
    
    # 创建随机森林模型
    logger.info("开始训练随机森林模型...")
    
    # 读取随机森林参数
    rf_params = config.get('rf_params', {})
    n_estimators = rf_params.get('n_estimators', 100)
    max_depth = rf_params.get('max_depth', 20)
    min_samples_split = rf_params.get('min_samples_split', 5)
    min_samples_leaf = rf_params.get('min_samples_leaf', 2)
    
    # 创建并训练随机森林模型
    model = RFMultiOutputModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train, target_cols)
    
    # 评估模型
    train_results = model.evaluate(X_train, y_train, target_cols)
    val_results = model.evaluate(X_val, y_val, target_cols)
    test_results = model.evaluate(X_test, y_test, target_cols)
    
    # 计算平均指标
    avg_train_r2 = np.mean(train_results['r2_scores'])
    avg_val_r2 = np.mean(val_results['r2_scores'])
    avg_test_r2 = np.mean(test_results['r2_scores'])
    
    avg_train_rmse = np.mean(train_results['rmse_values'])
    avg_val_rmse = np.mean(val_results['rmse_values'])
    avg_test_rmse = np.mean(test_results['rmse_values'])
    
    avg_train_rpd = np.mean(train_results['rpd_values'])
    avg_val_rpd = np.mean(val_results['rpd_values'])
    avg_test_rpd = np.mean(test_results['rpd_values'])
    
    # 输出评估结果
    logger.info(f"\nAverage R²: Train={avg_train_r2:.4f}, Validation={avg_val_r2:.4f}, Test={avg_test_r2:.4f}")
    
    # 显示三个数据集性能对比
    logger.info(f"\n========= 三个数据集性能对比 =========")
    logger.info(f"{'数据集':<10}{'平均R²':<12}{'平均RMSE':<12}{'平均RPD':<12}")
    logger.info(f"训练集      {avg_train_r2:.4f}     {avg_train_rmse:.4f}     {avg_train_rpd:.4f}")
    logger.info(f"验证集      {avg_val_r2:.4f}     {avg_val_rmse:.4f}     {avg_val_rpd:.4f}")
    logger.info(f"测试集      {avg_test_r2:.4f}     {avg_test_rmse:.4f}     {avg_test_rpd:.4f}")
    logger.info(f"========================================")
    
    # 显示详细任务评估指标
    logger.info(f"\n========= 详细任务评估指标 =========")
    logger.info(f"{'参数':<10}{'数据集':<13}{'R²':<10}{'RMSE':<10}{'MAE':<10}{'解释方差':<14}")
    logger.info(f"------------------------------------------------------------")
    
    for i, target in enumerate(target_cols):
        # 获取当前参数的训练集指标
        train_r2 = train_results['r2_scores'][i]
        train_rmse = train_results['rmse_values'][i]
        train_mae = train_results['mae_values'][i]
        
        # 计算解释方差（explained variance）
        train_true = train_results['targets'][:, i]
        train_pred = train_results['predictions'][:, i]
        train_ev = 1 - np.var(train_true - train_pred) / np.var(train_true)
        
        # 获取当前参数的验证集指标
        val_r2 = val_results['r2_scores'][i]
        val_rmse = val_results['rmse_values'][i]
        val_mae = val_results['mae_values'][i]
        
        val_true = val_results['targets'][:, i]
        val_pred = val_results['predictions'][:, i]
        val_ev = 1 - np.var(val_true - val_pred) / np.var(val_true)
        
        # 获取当前参数的测试集指标
        test_r2 = test_results['r2_scores'][i]
        test_rmse = test_results['rmse_values'][i]
        test_mae = test_results['mae_values'][i]
        
        test_true = test_results['targets'][:, i]
        test_pred = test_results['predictions'][:, i]
        test_ev = 1 - np.var(test_true - test_pred) / np.var(test_true)
        
        # 显示指标
        logger.info(f"{target:<10}{'训练集':<13}{train_r2:.4f}    {train_rmse:.4f}    {train_mae:.4f}    {train_ev:.4f}      ")
        logger.info(f"{'':<10}{'验证集':<13}{val_r2:.4f}    {val_rmse:.4f}    {val_mae:.4f}    {val_ev:.4f}      ")
        logger.info(f"{'':<10}{'测试集':<13}{test_r2:.4f}    {test_rmse:.4f}    {test_mae:.4f}    {test_ev:.4f}      ")
        logger.info(f"------------------------------------------------------------")
    
    # 显示各参数测试集性能
    logger.info(f"\n=================== 各参数测试集性能 ===================")
    logger.info(f"{'参数':<12}{'R²':<12}{'RMSE':<12}{'RPD':<12}")
    logger.info(f"-------------------------------------------------")
    
    for i, target in enumerate(target_cols):
        test_r2 = test_results['r2_scores'][i]
        test_rmse = test_results['rmse_values'][i]
        test_rpd = test_results['rpd_values'][i]
        logger.info(f"{target:<12}{test_r2:.4f}     {test_rmse:.4f}     {test_rpd:.4f}")
    
    logger.info(f"-------------------------------------------------")
    logger.info(f"{'平均':<12}{avg_test_r2:.4f}     {avg_test_rmse:.4f}     {avg_test_rpd:.4f}")
    logger.info(f"=================================================")
    logger.info(f"训练完成! 最佳验证集R²: {avg_val_r2:.4f}, 测试集R²: {avg_test_r2:.4f}")
    
    # 保存模型和结果
    # 注意：随机森林模型不需要n_components参数，应该传入None
    save_results_and_model(config, model, target_cols, train_results, val_results, test_results, None, logger)
    
    # 计算并记录训练时间
    end_time = time.time()
    training_duration = end_time - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"\n随机森林模型训练完成，耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    return model, train_results, val_results, test_results

def train_xgb_model(config, logger=None):
    """
    训练XGBoost多目标输出模型
    """
    # 记录开始时间
    start_time = time.time()
    
    # 加载数据
    logger.info("开始加载数据...")
    if not os.path.exists(config['data_path']):
        logger.error(f"错误: 找不到文件 {config['data_path']}")
        return None
    
    df = pd.read_csv(config['data_path'])
    logger.info(f"成功加载数据集，共 {len(df)} 条记录")
    
    # 提取特征和目标变量
    id_cols = df.columns[:3].tolist()
    logger.info(f"样本标识列（前3列）: {id_cols}")
    
    feature_cols = df.columns[3:276].tolist()
    logger.info(f"特征列数量: {len(feature_cols)}")
    
    # 根据作物类型设置目标变量
    if config['crop_type'] == 'tomato':
        target_cols = ['RUE', 'Pn', 'Gs', 'Ci', 'Tr', 'Ci-Ca', 'WUE', 'iWUE', 'α', 'Pmax', 'θ', 'Rd', 'SPAD', 'Ic', 'LAW']
    else:  # rice
        target_cols = ['SPAD', 'Pn', 'LNC', 'Chl-a', 'Chl-b', 'LAW', 'Cx', 'Chl']
    
    # 检查目标变量列是否在数据集中存在
    available_target_cols = [col for col in target_cols if col in df.columns]
    if len(available_target_cols) < len(target_cols):
        missing_cols = [col for col in target_cols if col not in available_target_cols]
        logger.warning(f"以下目标变量列在数据集中不存在: {missing_cols}")
        logger.info(f"将使用可用的目标变量列: {available_target_cols}")
        target_cols = available_target_cols
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    logger.info(f"特征数量: {X.shape[1]}, 目标变量数量: {y.shape[1]}")
    logger.info(f"样本数量: {X.shape[0]}")
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logger.info(f"数据集划分: 训练集 {X_train.shape[0]} 样本 ({X_train.shape[0]/X.shape[0]:.1%}), "
                f"验证集 {X_val.shape[0]} 样本 ({X_val.shape[0]/X.shape[0]:.1%}), "
                f"测试集 {X_test.shape[0]} 样本 ({X_test.shape[0]/X.shape[0]:.1%})")
    
    # 创建XGBoost模型
    logger.info("开始训练XGBoost模型...")
    
    # 读取XGBoost参数
    xgb_params = config.get('xgb_params', {})
    max_depth = xgb_params.get('max_depth', 6)
    learning_rate = xgb_params.get('learning_rate', 0.1)
    n_estimators = xgb_params.get('n_estimators', 100)
    subsample = xgb_params.get('subsample', 0.8)
    colsample_bytree = xgb_params.get('colsample_bytree', 0.8)
    
    # 创建并训练XGBoost模型
    model = XGBMultiOutputModel(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42
    )
    model.fit(X_train, y_train, target_cols, X_val=X_val, y_val=y_val)
    
    # 评估模型
    train_results = model.evaluate(X_train, y_train, target_cols)
    val_results = model.evaluate(X_val, y_val, target_cols)
    test_results = model.evaluate(X_test, y_test, target_cols)
    
    # 计算平均指标
    avg_train_r2 = np.mean(train_results['r2_scores'])
    avg_val_r2 = np.mean(val_results['r2_scores'])
    avg_test_r2 = np.mean(test_results['r2_scores'])
    
    avg_train_rmse = np.mean(train_results['rmse_values'])
    avg_val_rmse = np.mean(val_results['rmse_values'])
    avg_test_rmse = np.mean(test_results['rmse_values'])
    
    avg_train_rpd = np.mean(train_results['rpd_values'])
    avg_val_rpd = np.mean(val_results['rpd_values'])
    avg_test_rpd = np.mean(test_results['rpd_values'])
    
    # 输出评估结果
    logger.info(f"\nAverage R²: Train={avg_train_r2:.4f}, Validation={avg_val_r2:.4f}, Test={avg_test_r2:.4f}")
    
    # 显示三个数据集性能对比
    logger.info(f"\n========= 三个数据集性能对比 =========")
    logger.info(f"{'数据集':<10}{'平均R²':<12}{'平均RMSE':<12}{'平均RPD':<12}")
    logger.info(f"训练集      {avg_train_r2:.4f}     {avg_train_rmse:.4f}     {avg_train_rpd:.4f}")
    logger.info(f"验证集      {avg_val_r2:.4f}     {avg_val_rmse:.4f}     {avg_val_rpd:.4f}")
    logger.info(f"测试集      {avg_test_r2:.4f}     {avg_test_rmse:.4f}     {avg_test_rpd:.4f}")
    logger.info(f"========================================")
    
    # 显示详细任务评估指标
    logger.info(f"\n========= 详细任务评估指标 =========")
    logger.info(f"{'参数':<10}{'数据集':<13}{'R²':<10}{'RMSE':<10}{'MAE':<10}{'解释方差':<14}")
    logger.info(f"------------------------------------------------------------")
    
    for i, target in enumerate(target_cols):
        # 获取当前参数的训练集指标
        train_r2 = train_results['r2_scores'][i]
        train_rmse = train_results['rmse_values'][i]
        train_mae = train_results['mae_values'][i]
        
        # 计算解释方差（explained variance）
        train_true = train_results['targets'][:, i]
        train_pred = train_results['predictions'][:, i]
        train_ev = 1 - np.var(train_true - train_pred) / np.var(train_true)
        
        # 获取当前参数的验证集指标
        val_r2 = val_results['r2_scores'][i]
        val_rmse = val_results['rmse_values'][i]
        val_mae = val_results['mae_values'][i]
        
        val_true = val_results['targets'][:, i]
        val_pred = val_results['predictions'][:, i]
        val_ev = 1 - np.var(val_true - val_pred) / np.var(val_true)
        
        # 获取当前参数的测试集指标
        test_r2 = test_results['r2_scores'][i]
        test_rmse = test_results['rmse_values'][i]
        test_mae = test_results['mae_values'][i]
        
        test_true = test_results['targets'][:, i]
        test_pred = test_results['predictions'][:, i]
        test_ev = 1 - np.var(test_true - test_pred) / np.var(test_true)
        
        # 显示指标
        logger.info(f"{target:<10}{'训练集':<13}{train_r2:.4f}    {train_rmse:.4f}    {train_mae:.4f}    {train_ev:.4f}      ")
        logger.info(f"{'':<10}{'验证集':<13}{val_r2:.4f}    {val_rmse:.4f}    {val_mae:.4f}    {val_ev:.4f}      ")
        logger.info(f"{'':<10}{'测试集':<13}{test_r2:.4f}    {test_rmse:.4f}    {test_mae:.4f}    {test_ev:.4f}      ")
        logger.info(f"------------------------------------------------------------")
    
    # 显示各参数测试集性能
    logger.info(f"\n=================== 各参数测试集性能 ===================")
    logger.info(f"{'参数':<12}{'R²':<12}{'RMSE':<12}{'RPD':<12}")
    logger.info(f"-------------------------------------------------")
    
    for i, target in enumerate(target_cols):
        test_r2 = test_results['r2_scores'][i]
        test_rmse = test_results['rmse_values'][i]
        test_rpd = test_results['rpd_values'][i]
        logger.info(f"{target:<12}{test_r2:.4f}     {test_rmse:.4f}     {test_rpd:.4f}")
    
    logger.info(f"-------------------------------------------------")
    logger.info(f"{'平均':<12}{avg_test_r2:.4f}     {avg_test_rmse:.4f}     {avg_test_rpd:.4f}")
    logger.info(f"=================================================")
    logger.info(f"训练完成! 最佳验证集R²: {avg_val_r2:.4f}, 测试集R²: {avg_test_r2:.4f}")
    
    # 保存模型和结果
    # 注意：XGBoost模型不需要n_components参数，应该传入None
    save_results_and_model(config, model, target_cols, train_results, val_results, test_results, None, logger)
    
    # 计算并记录训练时间
    end_time = time.time()
    training_duration = end_time - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"\nXGBoost模型训练完成，耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    return model, train_results, val_results, test_results

def train_mlp_model(config, logger=None):
    """
    训练多层感知机（MLP）多目标输出模型
    """
    # 记录开始时间
    start_time = time.time()
    
    # 加载数据
    logger.info("开始加载数据...")
    if not os.path.exists(config['data_path']):
        logger.error(f"错误: 找不到文件 {config['data_path']}")
        return None
    
    df = pd.read_csv(config['data_path'])
    logger.info(f"成功加载数据集，共 {len(df)} 条记录")
    
    # 提取特征和目标变量
    id_cols = df.columns[:3].tolist()
    logger.info(f"样本标识列（前3列）: {id_cols}")
    
    feature_cols = df.columns[3:276].tolist()
    logger.info(f"特征列数量: {len(feature_cols)}")
    
    # 根据作物类型设置目标变量
    if config['crop_type'] == 'tomato':
        target_cols = ['RUE', 'Pn', 'Gs', 'Ci', 'Tr', 'Ci-Ca', 'WUE', 'iWUE', 'α', 'Pmax', 'θ', 'Rd', 'SPAD', 'Ic', 'LAW']
    else:  # rice
        target_cols = ['SPAD', 'Pn', 'LNC', 'Chl-a', 'Chl-b', 'LAW', 'Cx', 'Chl']
    
    # 检查目标变量列是否在数据集中存在
    available_target_cols = [col for col in target_cols if col in df.columns]
    if len(available_target_cols) < len(target_cols):
        missing_cols = [col for col in target_cols if col not in available_target_cols]
        logger.warning(f"以下目标变量列在数据集中不存在: {missing_cols}")
        logger.info(f"将使用可用的目标变量列: {available_target_cols}")
        target_cols = available_target_cols
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    logger.info(f"特征数量: {X.shape[1]}, 目标变量数量: {y.shape[1]}")
    logger.info(f"样本数量: {X.shape[0]}")
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logger.info(f"数据集划分: 训练集 {X_train.shape[0]} 样本 ({X_train.shape[0]/X.shape[0]:.1%}), "
                f"验证集 {X_val.shape[0]} 样本 ({X_val.shape[0]/X.shape[0]:.1%}), "
                f"测试集 {X_test.shape[0]} 样本 ({X_test.shape[0]/X.shape[0]:.1%})")
    
    # 创建MLP模型
    logger.info("开始训练MLP模型...")
    
    # 读取MLP参数
    mlp_params = config.get('mlp_params', {})
    hidden_layer_sizes = mlp_params.get('hidden_layer_sizes', (100, 50))
    activation = mlp_params.get('activation', 'relu')
    solver = mlp_params.get('solver', 'adam')
    alpha = mlp_params.get('alpha', 0.0001)
    max_iter = mlp_params.get('max_iter', 500)
    
    # 创建并训练MLP模型
    model = MLPMultiOutputModel(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        max_iter=max_iter,
        random_state=42
    )
    model.fit(X_train, y_train, target_cols)
    
    # 评估模型
    train_results = model.evaluate(X_train, y_train, target_cols)
    val_results = model.evaluate(X_val, y_val, target_cols)
    test_results = model.evaluate(X_test, y_test, target_cols)
    
    # 计算平均指标
    avg_train_r2 = np.mean(train_results['r2_scores'])
    avg_val_r2 = np.mean(val_results['r2_scores'])
    avg_test_r2 = np.mean(test_results['r2_scores'])
    
    avg_train_rmse = np.mean(train_results['rmse_values'])
    avg_val_rmse = np.mean(val_results['rmse_values'])
    avg_test_rmse = np.mean(test_results['rmse_values'])
    
    avg_train_rpd = np.mean(train_results['rpd_values'])
    avg_val_rpd = np.mean(val_results['rpd_values'])
    avg_test_rpd = np.mean(test_results['rpd_values'])
    
    # 输出评估结果
    logger.info(f"\nAverage R²: Train={avg_train_r2:.4f}, Validation={avg_val_r2:.4f}, Test={avg_test_r2:.4f}")
    
    # 显示三个数据集性能对比
    logger.info(f"\n========= 三个数据集性能对比 =========")
    logger.info(f"{'数据集':<10}{'平均R²':<12}{'平均RMSE':<12}{'平均RPD':<12}")
    logger.info(f"训练集      {avg_train_r2:.4f}     {avg_train_rmse:.4f}     {avg_train_rpd:.4f}")
    logger.info(f"验证集      {avg_val_r2:.4f}     {avg_val_rmse:.4f}     {avg_val_rpd:.4f}")
    logger.info(f"测试集      {avg_test_r2:.4f}     {avg_test_rmse:.4f}     {avg_test_rpd:.4f}")
    logger.info(f"========================================")
    
    # 显示详细任务评估指标
    logger.info(f"\n========= 详细任务评估指标 =========")
    logger.info(f"{'参数':<10}{'数据集':<13}{'R²':<10}{'RMSE':<10}{'MAE':<10}{'解释方差':<14}")
    logger.info(f"------------------------------------------------------------")
    
    for i, target in enumerate(target_cols):
        # 获取当前参数的训练集指标
        train_r2 = train_results['r2_scores'][i]
        train_rmse = train_results['rmse_values'][i]
        train_mae = train_results['mae_values'][i]
        
        # 计算解释方差（explained variance）
        train_true = train_results['targets'][:, i]
        train_pred = train_results['predictions'][:, i]
        train_ev = 1 - np.var(train_true - train_pred) / np.var(train_true)
        
        # 获取当前参数的验证集指标
        val_r2 = val_results['r2_scores'][i]
        val_rmse = val_results['rmse_values'][i]
        val_mae = val_results['mae_values'][i]
        
        val_true = val_results['targets'][:, i]
        val_pred = val_results['predictions'][:, i]
        val_ev = 1 - np.var(val_true - val_pred) / np.var(val_true)
        
        # 获取当前参数的测试集指标
        test_r2 = test_results['r2_scores'][i]
        test_rmse = test_results['rmse_values'][i]
        test_mae = test_results['mae_values'][i]
        
        test_true = test_results['targets'][:, i]
        test_pred = test_results['predictions'][:, i]
        test_ev = 1 - np.var(test_true - test_pred) / np.var(test_true)
        
        # 显示指标
        logger.info(f"{target:<10}{'训练集':<13}{train_r2:.4f}    {train_rmse:.4f}    {train_mae:.4f}    {train_ev:.4f}      ")
        logger.info(f"{'':<10}{'验证集':<13}{val_r2:.4f}    {val_rmse:.4f}    {val_mae:.4f}    {val_ev:.4f}      ")
        logger.info(f"{'':<10}{'测试集':<13}{test_r2:.4f}    {test_rmse:.4f}    {test_mae:.4f}    {test_ev:.4f}      ")
        logger.info(f"------------------------------------------------------------")
    
    # 显示各参数测试集性能
    logger.info(f"\n=================== 各参数测试集性能 ===================")
    logger.info(f"{'参数':<12}{'R²':<12}{'RMSE':<12}{'RPD':<12}")
    logger.info(f"-------------------------------------------------")
    
    for i, target in enumerate(target_cols):
        test_r2 = test_results['r2_scores'][i]
        test_rmse = test_results['rmse_values'][i]
        test_rpd = test_results['rpd_values'][i]
        logger.info(f"{target:<12}{test_r2:.4f}     {test_rmse:.4f}     {test_rpd:.4f}")
    
    logger.info(f"-------------------------------------------------")
    logger.info(f"{'平均':<12}{avg_test_r2:.4f}     {avg_test_rmse:.4f}     {avg_test_rpd:.4f}")
    logger.info(f"=================================================")
    logger.info(f"训练完成! 最佳验证集R²: {avg_val_r2:.4f}, 测试集R²: {avg_test_r2:.4f}")
    
    # 保存模型和结果
    # 注意：MLP模型不需要n_components参数，应该传入None
    save_results_and_model(config, model, target_cols, train_results, val_results, test_results, None, logger)
    
    # 计算并记录训练时间
    end_time = time.time()
    training_duration = end_time - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"\nMLP模型训练完成，耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    return model, train_results, val_results, test_results

def train_mplsr_model(config, logger=None):
    """
    训练MPLSR多输出模型（单模型预测所有参数）
    """
    # 记录开始时间
    start_time = time.time()
    
    # 加载数据
    logger.info("开始加载数据...")
    if not os.path.exists(config['data_path']):
        logger.error(f"错误: 找不到文件 {config['data_path']}")
        return None
    
    df = pd.read_csv(config['data_path'])
    logger.info(f"成功加载数据集，共 {len(df)} 条记录")
    
    # 提取特征和目标变量
    id_cols = df.columns[:3].tolist()
    logger.info(f"样本标识列（前3列）: {id_cols}")
    
    feature_cols = df.columns[3:276].tolist()
    logger.info(f"特征列数量: {len(feature_cols)}")
    
    # 根据作物类型设置目标变量
    if config['crop_type'] == 'tomato':
        target_cols = ['RUE', 'Pn', 'Gs', 'Ci', 'Tr', 'Ci-Ca', 'WUE', 'iWUE', 'α', 'Pmax', 'θ', 'Rd', 'SPAD', 'Ic', 'LAW']
    else:  # rice
        target_cols = ['SPAD', 'Pn', 'LNC', 'Chl-a', 'Chl-b', 'LAW', 'Cx', 'Chl']
    
    # 检查目标变量列是否在数据集中存在
    available_target_cols = [col for col in target_cols if col in df.columns]
    if len(available_target_cols) < len(target_cols):
        missing_cols = [col for col in target_cols if col not in available_target_cols]
        logger.warning(f"以下目标变量列在数据集中不存在: {missing_cols}")
        logger.info(f"将使用可用的目标变量列: {available_target_cols}")
        target_cols = available_target_cols
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    logger.info(f"特征数量: {X.shape[1]}, 目标变量数量: {y.shape[1]}")
    logger.info(f"样本数量: {X.shape[0]}")
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logger.info(f"数据集划分: 训练集 {X_train.shape[0]} 样本 ({X_train.shape[0]/X.shape[0]:.1%}), "
                f"验证集 {X_val.shape[0]} 样本 ({X_val.shape[0]/X.shape[0]:.1%}), "
                f"测试集 {X_test.shape[0]} 样本 ({X_test.shape[0]/X.shape[0]:.1%})")
    
    # 找到最佳MPLSR组件数
    if config.get('optimize_components', True):
        # 对于多输出PLSR，使用不同的优化策略
        n_components = find_optimal_mplsr_components(
            X_train, y_train, target_cols,
            max_components=config.get('max_components', 30),
            cv=config.get('cv_folds', 5),
            logger=logger
        )
    else:
        n_components = config.get('n_components', 15)
        logger.info(f"使用预设的MPLSR组件数: {n_components}")
    
    # 创建并训练MPLSR模型
    logger.info(f"开始训练MPLSR模型，使用 {n_components} 个组件...")
    logger.info("注意: MPLSR使用单个模型同时预测所有目标变量")
    
    model = MPLSRMultiOutputModel(n_components=n_components, max_iter=1000)
    model.fit(X_train, y_train, target_cols)
    
    # 输出模型信息
    model_info = model.get_model_info()
    logger.info(f"MPLSR模型信息:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")
    
    # 评估模型
    train_results = model.evaluate(X_train, y_train, target_cols)
    val_results = model.evaluate(X_val, y_val, target_cols)
    test_results = model.evaluate(X_test, y_test, target_cols)
    
    # 计算平均指标
    avg_train_r2 = np.mean(train_results['r2_scores'])
    avg_val_r2 = np.mean(val_results['r2_scores'])
    avg_test_r2 = np.mean(test_results['r2_scores'])
    
    avg_train_rmse = np.mean(train_results['rmse_values'])
    avg_val_rmse = np.mean(val_results['rmse_values'])
    avg_test_rmse = np.mean(test_results['rmse_values'])
    
    avg_train_rpd = np.mean(train_results['rpd_values'])
    avg_val_rpd = np.mean(val_results['rpd_values'])
    avg_test_rpd = np.mean(test_results['rpd_values'])
    
    logger.info(f"\nAverage R²: Train={avg_train_r2:.4f}, Validation={avg_val_r2:.4f}, Test={avg_test_r2:.4f}")
    
    # 显示三个数据集性能对比
    logger.info(f"\n========= 三个数据集性能对比 =========")
    logger.info(f"{'数据集':<10}{'平均R²':<12}{'平均RMSE':<12}{'平均RPD':<12}")
    logger.info(f"训练集      {avg_train_r2:.4f}     {avg_train_rmse:.4f}     {avg_train_rpd:.4f}")
    logger.info(f"验证集      {avg_val_r2:.4f}     {avg_val_rmse:.4f}     {avg_val_rpd:.4f}")
    logger.info(f"测试集      {avg_test_r2:.4f}     {avg_test_rmse:.4f}     {avg_test_rpd:.4f}")
    logger.info(f"========================================")
    
    # 显示详细任务评估指标
    logger.info(f"\n========= 详细任务评估指标 =========")
    logger.info(f"{'参数':<10}{'数据集':<13}{'R²':<10}{'RMSE':<10}{'MAE':<10}{'解释方差':<14}")
    logger.info(f"------------------------------------------------------------")
    
    for i, target in enumerate(target_cols):
        # 获取当前参数的训练集指标
        train_r2 = train_results['r2_scores'][i]
        train_rmse = train_results['rmse_values'][i]
        train_mae = train_results['mae_values'][i]
        
        # 计算解释方差（explained variance）
        train_true = train_results['targets'][:, i]
        train_pred = train_results['predictions'][:, i]
        train_ev = 1 - np.var(train_true - train_pred) / np.var(train_true)
        
        # 获取当前参数的验证集指标
        val_r2 = val_results['r2_scores'][i]
        val_rmse = val_results['rmse_values'][i]
        val_mae = val_results['mae_values'][i]
        
        val_true = val_results['targets'][:, i]
        val_pred = val_results['predictions'][:, i]
        val_ev = 1 - np.var(val_true - val_pred) / np.var(val_true)
        
        # 获取当前参数的测试集指标
        test_r2 = test_results['r2_scores'][i]
        test_rmse = test_results['rmse_values'][i]
        test_mae = test_results['mae_values'][i]
        
        test_true = test_results['targets'][:, i]
        test_pred = test_results['predictions'][:, i]
        test_ev = 1 - np.var(test_true - test_pred) / np.var(test_true)
        
        # 显示指标
        logger.info(f"{target:<10}{'训练集':<13}{train_r2:.4f}    {train_rmse:.4f}    {train_mae:.4f}    {train_ev:.4f}      ")
        logger.info(f"{'':<10}{'验证集':<13}{val_r2:.4f}    {val_rmse:.4f}    {val_mae:.4f}    {val_ev:.4f}      ")
        logger.info(f"{'':<10}{'测试集':<13}{test_r2:.4f}    {test_rmse:.4f}    {test_mae:.4f}    {test_ev:.4f}      ")
        logger.info(f"------------------------------------------------------------")
    
    # 显示各参数测试集性能
    logger.info(f"\n=================== 各参数测试集性能 ===================")
    logger.info(f"{'参数':<12}{'R²':<12}{'RMSE':<12}{'RPD':<12}")
    logger.info(f"-------------------------------------------------")
    
    for i, target in enumerate(target_cols):
        test_r2 = test_results['r2_scores'][i]
        test_rmse = test_results['rmse_values'][i]
        test_rpd = test_results['rpd_values'][i]
        logger.info(f"{target:<12}{test_r2:.4f}     {test_rmse:.4f}     {test_rpd:.4f}")
    
    logger.info(f"-------------------------------------------------")
    logger.info(f"{'平均':<12}{avg_test_r2:.4f}     {avg_test_rmse:.4f}     {avg_test_rpd:.4f}")
    logger.info(f"=================================================")
    logger.info(f"训练完成! 最佳验证集R²: {avg_val_r2:.4f}, 测试集R²: {avg_test_r2:.4f}")
    
    # 保存模型和结果
    # 注意：MPLSR模型不需要n_components参数，应该传入None
    save_results_and_model(config, model, target_cols, train_results, val_results, test_results, None, logger)
    
    # 计算并记录训练时间
    end_time = time.time()
    training_duration = end_time - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"\nMPLSR模型训练完成，耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    return model, train_results, val_results, test_results

def find_optimal_mplsr_components(X_train, y_train, target_names, max_components=30, cv=5, logger=None):
    """
    为多输出PLSR找到最佳组件数
    使用交叉验证在所有目标变量上评估性能
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    
    if logger:
        logger.info("开始寻找MPLSR的最佳组件数...")
    
    # 标准化数据
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_train)
    y_scaled = scaler_y.fit_transform(y_train)
    
    # 测试不同的组件数
    max_comp = min(max_components, X_train.shape[1], X_train.shape[0], y_train.shape[1])
    component_range = list(range(1, max_comp + 1))
    
    cv_scores = {}
    best_score = -float('inf')
    best_components = 1
    
    for n_comp in component_range:
        try:
            # 创建PLSR模型
            plsr = PLSRegression(n_components=n_comp)
            
            # 计算交叉验证分数（所有目标变量的平均）
            scores = cross_val_score(plsr, X_scaled, y_scaled, cv=cv, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
            mean_score = np.mean(scores)
            cv_scores[n_comp] = mean_score
            
            if logger:
                logger.info(f"n_components={n_comp}: CV Score={mean_score:.6f}")
            
            # 更新最佳组件数
            if mean_score > best_score:
                best_score = mean_score
                best_components = n_comp
                
        except Exception as e:
            if logger:
                logger.warning(f"n_components={n_comp}时出错: {str(e)}")
            continue
    
    if logger:
        logger.info(f"MPLSR最佳组件数: {best_components} (CV Score: {best_score:.6f})")
        
        # 显示前几个最佳组件数的性能
        sorted_scores = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("前5个最佳组件数:")
        for comp, score in sorted_scores:
            logger.info(f"  n_components={comp}: {score:.6f}")
    
    return best_components

# 修改save_results_and_model函数以适应不同类型的模型
def save_results_and_model(config, model, target_cols, train_results, val_results, test_results, n_components, logger):
    """保存模型和结果"""
    # 获取模型类型和目录
    model_type = config.get('model_type', 'plsr')
    model_dir_name = config.get('model_dir', f'{model_type}_model')
    
    # 创建模型目录
    models_dir = os.path.join(config['results_dir'], model_dir_name)
    os.makedirs(models_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(models_dir, f'{model_type}_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"已保存{model_type.upper()}模型到: {model_path}")
    
    # 保存结果数据
    results_dir = os.path.join(config['results_dir'], 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存预测结果
    for i, target in enumerate(target_cols):
        # 创建训练集结果
        train_df = pd.DataFrame({
            'True': train_results['targets'][:, i],
            'Predicted': train_results['predictions'][:, i]
        })
        
        val_df = pd.DataFrame({
            'True': val_results['targets'][:, i],
            'Predicted': val_results['predictions'][:, i]
        })
        
        test_df = pd.DataFrame({
            'True': test_results['targets'][:, i],
            'Predicted': test_results['predictions'][:, i]
        })
        
        # 合并数据集
        combined_df = pd.DataFrame({
            'Set': ['Train'] * len(train_df) + ['Validation'] * len(val_df) + ['Test'] * len(test_df),
            'Parameter': [target] * (len(train_df) + len(val_df) + len(test_df)),
            'True': np.concatenate([train_df['True'], val_df['True'], test_df['True']]),
            'Predicted': np.concatenate([train_df['Predicted'], val_df['Predicted'], test_df['Predicted']])
        })
        
        # 保存结果
        train_df.to_csv(f"{results_dir}/{target}_train.csv", index=False)
        val_df.to_csv(f"{results_dir}/{target}_val.csv", index=False)
        test_df.to_csv(f"{results_dir}/{target}_test.csv", index=False)
        combined_df.to_csv(f"{results_dir}/{target}_all.csv", index=False)
    
    # 保存评估指标
    metrics = []
    for i, target in enumerate(target_cols):
        metrics.append({
            'Parameter': target,
            'Train_R2': f"{train_results['r2_scores'][i]:.4f}",  # 格式化为4位小数
            'Val_R2': f"{val_results['r2_scores'][i]:.4f}",
            'Test_R2': f"{test_results['r2_scores'][i]:.4f}",
            'Train_RMSE': f"{train_results['rmse_values'][i]:.4f}",
            'Val_RMSE': f"{val_results['rmse_values'][i]:.4f}",
            'Test_RMSE': f"{test_results['rmse_values'][i]:.4f}",
            'Train_RPD': f"{train_results.get('rpd_values', [0]*len(target_cols))[i]:.4f}",
            'Val_RPD': f"{val_results.get('rpd_values', [0]*len(target_cols))[i]:.4f}",
            'Test_RPD': f"{test_results.get('rpd_values', [0]*len(target_cols))[i]:.4f}",
            'Train_MAE': f"{train_results['mae_values'][i]:.4f}",
            'Val_MAE': f"{val_results['mae_values'][i]:.4f}",
            'Test_MAE': f"{test_results['mae_values'][i]:.4f}"
        })
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f"{results_dir}/all_metrics.csv", index=False)
    logger.info(f"已保存评估指标到: {results_dir}/all_metrics.csv")
    
    # 保存标准格式的指标结果到单独的文件
    # 创建与MT_task13_multitask_vit.py一致的格式
    formatted_metrics = {
        'r2': {},
        'rmse': {},
        'rpd': {}
    }
    
    for i, target in enumerate(target_cols):
        formatted_metrics['r2'][target] = {
            'train': f"{train_results['r2_scores'][i]:.4f}",
            'val': f"{val_results['r2_scores'][i]:.4f}",
            'test': f"{test_results['r2_scores'][i]:.4f}"
        }
        formatted_metrics['rmse'][target] = {
            'train': f"{train_results['rmse_values'][i]:.4f}",
            'val': f"{val_results['rmse_values'][i]:.4f}",
            'test': f"{test_results['rmse_values'][i]:.4f}"
        }
        formatted_metrics['rpd'][target] = {
            'train': f"{train_results.get('rpd_values', [0]*len(target_cols))[i]:.4f}",
            'val': f"{val_results.get('rpd_values', [0]*len(target_cols))[i]:.4f}",
            'test': f"{test_results.get('rpd_values', [0]*len(target_cols))[i]:.4f}"
        }
    
    # 添加平均值
    formatted_metrics['r2']['average'] = {
        'train': f"{np.mean(train_results['r2_scores']):.4f}",
        'val': f"{np.mean(val_results['r2_scores']):.4f}",
        'test': f"{np.mean(test_results['r2_scores']):.4f}"
    }
    formatted_metrics['rmse']['average'] = {
        'train': f"{np.mean(train_results['rmse_values']):.4f}",
        'val': f"{np.mean(val_results['rmse_values']):.4f}",
        'test': f"{np.mean(test_results['rmse_values']):.4f}"
    }
    formatted_metrics['rpd']['average'] = {
        'train': f"{np.mean(train_results.get('rpd_values', [0]*len(target_cols))):.4f}",
        'val': f"{np.mean(val_results.get('rpd_values', [0]*len(target_cols))):.4f}",
        'test': f"{np.mean(test_results.get('rpd_values', [0]*len(target_cols))):.4f}"
    }
    
    # 保存格式化的指标
    with open(f"{results_dir}/formatted_metrics.json", 'w') as f:
        json.dump(formatted_metrics, f, indent=4)
    logger.info(f"Formatted metrics saved to: {results_dir}/formatted_metrics.json")
    
    # 保存模型配置
    model_config = {
        'crop_type': config['crop_type'],
        'target_variables': target_cols,
        'data_path': config['data_path'],
        'avg_train_r2': float(np.mean(train_results['r2_scores'])),
        'avg_val_r2': float(np.mean(val_results['r2_scores'])),
        'avg_test_r2': float(np.mean(test_results['r2_scores'])),
        'model_file': f'{model_type}_model.pkl'
    }
    
    # 根据模型类型添加不同的配置信息
    if model_type == 'plsr' and n_components is not None:
        model_config['n_components'] = n_components
        if hasattr(model, 'models'):
            model_config['feature_count'] = model.models[target_cols[0]].x_weights_.shape[0]
    elif model_type == 'svr':
        # SVR模型的特定配置
        if isinstance(model, dict) and 'models' in model:
            # 旧版本兼容
            if hasattr(model['models'][target_cols[0]], 'get_params'):
                model_config['svr_params'] = model['models'][target_cols[0]].get_params()
        elif hasattr(model, 'models') and hasattr(model.models[target_cols[0]], 'get_params'):
            model_config['svr_params'] = model.models[target_cols[0]].get_params()
    elif model_type == 'rf':
        # 随机森林模型的特定配置
        if isinstance(model, dict) and 'models' in model:
            # 旧版本兼容
            if hasattr(model['models'][target_cols[0]], 'get_params'):
                model_config['rf_params'] = model['models'][target_cols[0]].get_params()
            if 'feature_importances' in model:
                # 保存平均特征重要性
                avg_importance = np.zeros(len(model['feature_importances'][target_cols[0]]))
                for target in target_cols:
                    avg_importance += model['feature_importances'][target]
                avg_importance /= len(target_cols)
                model_config['avg_feature_importance'] = avg_importance.tolist()
        elif hasattr(model, 'models'):
            if hasattr(model.models[target_cols[0]], 'get_params'):
                               model_config['rf_params'] = model.models[target_cols[0]].get_params()
            if hasattr(model, 'feature_importances'):
                # 保存平均特征重要性
                avg_importance = np.zeros(len(model.feature_importances[target_cols[0]]))
                for target in target_cols:
                    avg_importance += model.feature_importances[target]
                avg_importance /= len(target_cols)
                model_config['avg_feature_importance'] = avg_importance.tolist()
    elif model_type == 'xgb':
        # XGBoost模型的特定配置
        if isinstance(model, dict) and 'models' in model:
            # 旧版本兼容
            xgb_params = {
                'max_depth': model['models'][target_cols[0]].get_params().get('max_depth', 6),
                'learning_rate': model['models'][target_cols[0]].get_params().get('learning_rate', 0.1),
                'n_estimators': model['models'][target_cols[0]].get_params().get('n_estimators', 100)
            }
        elif hasattr(model, 'models'):
            # 获取XGBoost模型参数
            xgb_params = {}
            try:
                booster = model.models[target_cols[0]]
                xgb_params = {
                    'max_depth': int(booster.get_params().get('max_depth', 6)),
                    'learning_rate': float(booster.get_params().get('eta', 0.1)),
                    'n_estimators': model.n_estimators
                }
            except (AttributeError, KeyError):
                # 如果无法获取参数，使用默认值
                xgb_params = {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100
                }
        
        model_config['xgb_params'] = xgb_params
    elif model_type == 'mlp':
        # MLP模型的特定配置
        if isinstance(model, dict) and 'models' in model:
            # 旧版本兼容
            if hasattr(model['models'][target_cols[0]], 'get_params'):
                mlp_params = model['models'][target_cols[0]].get_params()
                model_config['mlp_params'] = {
                    'hidden_layer_sizes': mlp_params.get('hidden_layer_sizes', (100, 50)),
                    'activation': mlp_params.get('activation', 'relu'),
                    'solver': mlp_params.get('solver', 'adam'),
                    'alpha': mlp_params.get('alpha', 0.0001)
                }
        elif hasattr(model, 'models'):
            if hasattr(model.models[target_cols[0]], 'get_params'):
                mlp_params = model.models[target_cols[0]].get_params()
                model_config['mlp_params'] = {
                    'hidden_layer_sizes': mlp_params.get('hidden_layer_sizes', (100, 50)),
                    'activation': mlp_params.get('activation', 'relu'),
                    'solver': mlp_params.get('solver', 'adam'),
                    'alpha': mlp_params.get('alpha', 0.0001)
                }
            else:
                # 使用模型初始化参数
                model_config['mlp_params'] = {
                    'hidden_layer_sizes': model.hidden_layer_sizes,
                    'activation': model.activation,
                    'solver': model.solver,
                    'alpha': model.alpha,
                    'max_iter': model.max_iter
                }
    
    with open(f"{models_dir}/model_config.json", 'w') as f:
        json.dump(model_config, f, indent=4)
    logger.info(f"已保存模型配置到: {models_dir}/model_config.json")
    
    # 尝试获取特征重要性
    try:
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance(target_cols)
            
            # 保存特征重要性
            for target, importance in feature_importance.items():
                importance_df = pd.DataFrame({
                    'Feature_Index': list(range(len(importance))),
                    'Importance': importance
                })
                importance_df.to_csv(f"{results_dir}/{target}_feature_importance.csv", index=False)
            
            logger.info(f"已保存特征重要性到: {results_dir}/")
        else:
            logger.info("当前模型不支持特征重要性分析")
    except Exception as e:
        logger.warning(f"保存特征重要性时出错: {str(e)}")

def generate_baseline_model_stats(logger=None):
    """
    生成基线模型的详细统计信息
    """
    if logger is None:
        import logging
        logger = logging.getLogger()
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    logger.info("\n" + "="*120)
    logger.info("机器学习基线模型对比统计表")
    logger.info("="*120)
    
    # 基线模型详细对比数据 - 添加MultiOutput DNN
    baseline_models = [
        # 模型名称, 参数量, 关键参数, 预测方式, 特征学习, 复杂度, 训练时间, 内存占用
        ["PLSR", "~500", "n_components=15", "独立预测", "线性降维", "低", "极快", "2MB"],
        ["MPLSR", "~500", "n_components=15", "同时预测", "共享潜在空间", "低", "极快", "2MB"],
        ["SVR", "~5K", "C=10.0,kernel=rbf", "独立预测", "核变换", "中", "快", "20MB"],
        ["RandomForest", "~50K", "n_estimators=100", "独立预测", "特征选择", "中", "快", "200MB"],
        ["XGBoost", "~30K", "max_depth=6", "独立预测", "梯度提升", "中", "快", "120MB"],
        ["MLP", "~33K", "layers=(100,50)", "独立预测", "非线性变换", "中", "中等", "130MB"],
        ["MultiOutput DNN", "~111K", "layers=(256,128,64)", "同时预测", "深度非线性", "中高", "中等", "450MB"]
    ]
    
    headers = ["模型名称", "参数量", "关键参数", "预测方式", "特征学习", "复杂度", "训练时间", "内存占用"]
    
    # 计算列宽
    col_widths = [max(len(str(row[i])) for row in [headers] + baseline_models) + 2 for i in range(len(headers))]
    
    def print_separator():
        logger.info("+" + "+".join("-" * width for width in col_widths) + "+")
    
    def print_row(row):
        formatted_row = "|" + "|".join(f"{str(item):^{col_widths[i]}}" for i, item in enumerate(row)) + "|"
        logger.info(formatted_row)
    
    print_separator()
    print_row(headers)
    print_separator()
    for row in baseline_models:
        print_row(row)
    print_separator()
    
    # 算法特性分析
    logger.info("\n" + "="*100)
    logger.info("基线模型算法特性分析")
    logger.info("="*100)
    
    algorithm_analysis = [
        ["模型", "算法原理", "优势", "劣势", "最佳应用"],
        ["PLSR", "潜在变量回归", "处理高维共线性", "线性假设局限", "高维数据降维"],
        ["MPLSR", "多输出潜在变量", "参数关联学习", "线性关系假设", "相关参数预测"],
        ["SVR", "支持向量回归", "非线性建模", "参数调节复杂", "复杂非线性关系"],
        ["RandomForest", "集成决策树", "特征重要性", "容易过拟合", "特征选择分析"],
        ["XGBoost", "梯度提升树", "高精度预测", "调参工程量大", "结构化数据"],
        ["MLP", "多层感知机", "通用近似器", "需要大量数据", "神经网络基线"]
    ]
    
    # 算法分析表格
    algo_col_widths = [max(len(str(row[i])) for row in algorithm_analysis) + 2 for i in range(len(algorithm_analysis[0]))]
    
    def print_algo_separator():
        logger.info("+" + "+".join("-" * width for width in algo_col_widths) + "+")
    
    def print_algo_row(row):
        formatted_row = "|" + "|".join(f"{str(item):^{algo_col_widths[i]}}" for i, item in enumerate(row)) + "|"
        logger.info(formatted_row)
    
    print_algo_separator()
    for i, row in enumerate(algorithm_analysis):
        print_algo_row(row)
        if i == 0:
            print_algo_separator()
    print_algo_separator()
    
    return baseline_models

def main():
    # 设置配置参数
    config = {
        # 数据路径和结果目录
        # 'data_path': 'data/processed/subsets/dataset_all_per_class.csv',  # 番茄数据集
        # 'results_dir': 'results/ML_baseline/tomato_mplsr',  # 结果保存目录
        # 'crop_type': 'tomato',  # 作物类型，可选 'tomato' 或 'rice'
        'data_path': 'data/processed/Rice subsets/rice dataset_all_per_class.csv',  # 水稻数据集     
        'results_dir': 'results/ML_baseline/rice_mplsr',  # 水稻结果保存目录 
        'crop_type': 'rice',  # 作物类型
        
        # 其他可选基线模型及其参数:
        'model_type': 'mplsr',  # 'plsr', 'svr', 'rf', 'xgb', 'mlp', 'mplsr'
        'optimize_components': True,  # 是否优化PLSR组件数量
        'max_components': 30,  # 最大PLSR组件数
        'cv_folds': 5,  # 交叉验证折数
        'n_components': 15,  # 如果不优化，使用的默认组件数
                        
        # SVR参数
        'svr_params': {
            'kernel': 'rbf',
            'C': 10.0,
            'epsilon': 0.1,
            'gamma': 'scale'
        },
        
        # 随机森林参数
        'rf_params': {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        },
        
        # XGBoost参数
        'xgb_params': {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        
        # MLP参数
        'mlp_params': {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'max_iter': 500
        }
    }
    
    # 设置日志系统
    logger = setup_logger(config)
    
    # 设置随机种子
    set_seed(42)
    
    logger.info("=== 开始训练基线模型 ===")
    logger.info(f"作物类型: {config['crop_type']}")
    logger.info(f"数据路径: {config['data_path']}")
    
    # 根据选择的模型类型进行训练
    model_type = config.get('model_type', 'plsr')
    logger.info(f"选择的模型类型: {model_type}")
    
    if model_type == 'plsr':
        model, train_results, val_results, test_results = train_plsr_model(config, logger)
    elif model_type == 'mplsr':
        model, train_results, val_results, test_results = train_mplsr_model(config, logger)
    elif model_type == 'svr':
        model, train_results, val_results, test_results = train_svr_model(config, logger)
    elif model_type == 'rf':
        model, train_results, val_results, test_results = train_rf_model(config, logger)
    elif model_type == 'xgb':
        model, train_results, val_results, test_results = train_xgb_model(config, logger)
    elif model_type == 'mlp':
        model, train_results, val_results, test_results = train_mlp_model(config, logger)
    else:
        logger.error(f"不支持的模型类型: {model_type}")
        return
    
    # 在main函数末尾添加模型对比表格
    logger.info("生成基线模型对比统计表...")
    generate_baseline_model_stats(logger)
    
    logger.info(f"=== {model_type.upper()}基线模型训练完成 ===")

if __name__ == "__main__":
    main()