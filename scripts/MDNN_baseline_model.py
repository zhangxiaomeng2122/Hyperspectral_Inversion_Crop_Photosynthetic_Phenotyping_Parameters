import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import pickle
import json
from sklearn.model_selection import train_test_split

# 导入自定义模块
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.plot_setting import setfig
from utils.baseset import setup_logger, set_seed
from mt_hyperspectral.models.MDNN_model import MultiOutputDNNModel

def train_multioutput_dnn(config, logger=None):
    """
    训练MultiOutput DNN模型
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
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建并训练MultiOutput DNN模型
    logger.info("开始创建MultiOutput DNN模型...")
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    model = MultiOutputDNNModel(
        input_size=input_size,
        output_size=output_size,
        use_batch_norm=config.get('use_batch_norm', True),
        device=device
    )
    
    # 输出模型信息
    model_info = model.get_model_info()
    logger.info("MultiOutput DNN模型信息:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")
    
    # 训练模型
    logger.info(f"开始训练MultiOutput DNN模型...")
    model.fit(
        X_train=X_train,
        y_train=y_train,
        target_names=target_cols,
        X_val=X_val,
        y_val=y_val,
        num_epochs=config.get('num_epochs', 200),
        batch_size=config.get('batch_size', 32),
        learning_rate=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-5),
        patience=config.get('patience', 20),
        logger=logger
    )
    
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
    save_results_and_model(config, model, target_cols, train_results, val_results, test_results, logger)
    
    # 计算并记录训练时间
    end_time = time.time()
    training_duration = end_time - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"\nMultiOutput DNN模型训练完成，耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    return model, train_results, val_results, test_results

def save_results_and_model(config, model, target_cols, train_results, val_results, test_results, logger):
    """保存模型和结果"""
    # 获取模型类型和目录
    model_type = config.get('model_type', 'multioutput_dnn')
    model_dir_name = config.get('model_dir', f'{model_type}_model')
    
    # 创建模型目录
    models_dir = os.path.join(config['results_dir'], model_dir_name)
    os.makedirs(models_dir, exist_ok=True)
    
    # 保存PyTorch模型
    model_path = os.path.join(models_dir, f'{model_type}_model.pth')
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'model_config': model.get_model_info(),
        'target_names': model.target_names
    }, model_path)
    logger.info(f"已保存{model_type.upper()}模型到: {model_path}")
    
    # 保存完整的模型对象（用于兼容性）
    model_pkl_path = os.path.join(models_dir, f'{model_type}_model.pkl')
    with open(model_pkl_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"已保存{model_type.upper()}完整模型到: {model_pkl_path}")
    
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
            'Train_R2': f"{train_results['r2_scores'][i]:.4f}",
            'Val_R2': f"{val_results['r2_scores'][i]:.4f}",
            'Test_R2': f"{test_results['r2_scores'][i]:.4f}",
            'Train_RMSE': f"{train_results['rmse_values'][i]:.4f}",
            'Val_RMSE': f"{val_results['rmse_values'][i]:.4f}",
            'Test_RMSE': f"{test_results['rmse_values'][i]:.4f}",
            'Train_RPD': f"{train_results['rpd_values'][i]:.4f}",
            'Val_RPD': f"{val_results['rpd_values'][i]:.4f}",
            'Test_RPD': f"{test_results['rpd_values'][i]:.4f}",
            'Train_MAE': f"{train_results['mae_values'][i]:.4f}",
            'Val_MAE': f"{val_results['mae_values'][i]:.4f}",
            'Test_MAE': f"{test_results['mae_values'][i]:.4f}"
        })
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f"{results_dir}/all_metrics.csv", index=False)
    logger.info(f"已保存评估指标到: {results_dir}/all_metrics.csv")
    
    # 保存标准格式的指标结果
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
            'train': f"{train_results['rpd_values'][i]:.4f}",
            'val': f"{val_results['rpd_values'][i]:.4f}",
            'test': f"{test_results['rpd_values'][i]:.4f}"
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
        'train': f"{np.mean(train_results['rpd_values']):.4f}",
        'val': f"{np.mean(val_results['rpd_values']):.4f}",
        'test': f"{np.mean(test_results['rpd_values']):.4f}"
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
        'model_info': model.get_model_info(),
        'avg_train_r2': float(np.mean(train_results['r2_scores'])),
        'avg_val_r2': float(np.mean(val_results['r2_scores'])),
        'avg_test_r2': float(np.mean(test_results['r2_scores'])),
        'model_file': f'{model_type}_model.pth'
    }
    
    with open(f"{models_dir}/model_config.json", 'w') as f:
        json.dump(model_config, f, indent=4)
    logger.info(f"已保存模型配置到: {models_dir}/model_config.json")

def generate_multioutput_dnn_stats(logger=None):
    """
    生成MultiOutput DNN模型的详细统计信息
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
    
    logger.info("\n" + "="*80)
    logger.info("MultiOutput DNN模型详细统计")
    logger.info("="*80)
    
    # 模型架构分析 - 与Ensemble完全一致的单模型架构
    input_features = 273  # 高光谱特征数
    
    # 与Ensemble单模型完全相同的参数计算
    layer1_params = input_features * 256 + 256  # Linear(273, 256)
    bn1_params = 256 * 2  # BatchNorm1d(256): weight + bias
    layer2_params = 256 * 128 + 128  # Linear(256, 128)
    bn2_params = 128 * 2  # BatchNorm1d(128)
    layer3_params = 128 * 64 + 64  # Linear(128, 64)
    bn3_params = 64 * 2  # BatchNorm1d(64)
    
    # 输出层参数（这是唯一与Ensemble不同的地方）
    output_params_tomato = 64 * 13 + 13  # Linear(64, 13) 番茄 - 多输出
    output_params_rice = 64 * 8 + 8      # Linear(64, 8) 水稻 - 多输出
    
    # Ensemble单模型输出层参数（单输出）
    ensemble_output_params = 64 * 1 + 1  # Linear(64, 1) - 单输出
    
    # 计算总参数量
    base_params = (layer1_params + bn1_params + layer2_params + 
                  bn2_params + layer3_params + bn3_params)
    
    multioutput_total_tomato = base_params + output_params_tomato
    multioutput_total_rice = base_params + output_params_rice
    
    # Ensemble单模型总参数量
    ensemble_single_model = base_params + ensemble_output_params
    
    logger.info("MultiOutput DNN vs Ensemble架构对比:")
    logger.info(f"  共同架构: 273→256→128→64")
    logger.info(f"  激活函数: ReLU→ReLU→ReLU (完全相同)")
    logger.info(f"  正则化: BatchNorm + Dropout(0.5→0.3→0.2) (完全相同)")
    logger.info(f"  初始化策略: Xavier Uniform (完全相同)")
    logger.info("")
    logger.info("输出层差异:")
    logger.info(f"  Ensemble单模型输出层: 64→1 ({ensemble_output_params:,}参数)")
    logger.info(f"  MultiOutput番茄输出层: 64→13 ({output_params_tomato:,}参数)")
    logger.info(f"  MultiOutput水稻输出层: 64→8 ({output_params_rice:,}参数)")
    
    # 计算总参数量
    base_params = (layer1_params + bn1_params + layer2_params + 
                  bn2_params + layer3_params + bn3_params)
    
    multioutput_total_tomato = base_params + output_params_tomato
    multioutput_total_rice = base_params + output_params_rice
    
    # Ensemble单模型总参数量
    ensemble_single_model = base_params + ensemble_output_params
    
    logger.info("MultiOutput DNN vs Ensemble架构对比:")
    logger.info(f"  共同架构: 273→256→128→64")
    logger.info(f"  激活函数: ReLU→ReLU→ReLU (完全相同)")
    logger.info(f"  正则化: BatchNorm + Dropout(0.5→0.3→0.2) (完全相同)")
    logger.info(f"  初始化策略: Xavier Uniform (完全相同)")
    logger.info("")
    logger.info("输出层差异:")
    logger.info(f"  Ensemble单模型输出层: 64→1 ({ensemble_output_params:,}参数)")
    logger.info(f"  MultiOutput番茄输出层: 64→13 ({output_params_tomato:,}参数)")
    logger.info(f"  MultiOutput水稻输出层: 64→8 ({output_params_rice:,}参数)")
    logger.info("")
    logger.info("总参数量对比:")
    logger.info(f"  Ensemble单模型: {ensemble_single_model:,} ({ensemble_single_model/1000:.1f}K)")
    logger.info(f"  MultiOutput番茄: {multioutput_total_tomato:,} ({multioutput_total_tomato/1000:.1f}K)")
    logger.info(f"  MultiOutput水稻: {multioutput_total_rice:,} ({multioutput_total_rice/1000:.1f}K)")
    logger.info("")
    logger.info("参数量差异分析:")
    logger.info(f"  番茄额外参数: {multioutput_total_tomato - ensemble_single_model:,} (仅输出层差异)")
    logger.info(f"  水稻额外参数: {multioutput_total_rice - ensemble_single_model:,} (仅输出层差异)")
    
    # 内存使用分析
    param_memory_single = ensemble_single_model * 4 / (1024**2)  # Ensemble单模型内存
    param_memory_tomato = multioutput_total_tomato * 4 / (1024**2)
    param_memory_rice = multioutput_total_rice * 4 / (1024**2)
    
    batch_size = 32
    activation_memory = (256 + 128 + 64) * batch_size * 4 / (1024**2)  # 激活内存相同
    
    logger.info(f"\n内存使用对比:")
    logger.info(f"  Ensemble单模型参数内存: {param_memory_single:.1f} MB")
    logger.info(f"  MultiOutput番茄参数内存: {param_memory_tomato:.1f} MB")
    logger.info(f"  MultiOutput水稻参数内存: {param_memory_rice:.1f} MB")
    logger.info(f"  训练激活值内存: {activation_memory:.1f} MB (完全相同)")
    
    # 与完整Ensemble系统的对比
    ensemble_total_tomato = ensemble_single_model * 13  # 13个独立模型
    ensemble_total_rice = ensemble_single_model * 8     # 8个独立模型
    
    logger.info(f"\n与完整Ensemble系统对比:")
    logger.info(f"  完整Ensemble番茄: {ensemble_total_tomato:,} ({ensemble_total_tomato/1e6:.2f}M)")
    logger.info(f"  MultiOutput番茄: {multioutput_total_tomato:,} ({multioutput_total_tomato/1000:.1f}K)")
    logger.info(f"  参数量减少: {(1 - multioutput_total_tomato/ensemble_total_tomato)*100:.1f}%")
    logger.info("")
    logger.info(f"  完整Ensemble水稻: {ensemble_total_rice:,} ({ensemble_total_rice/1000:.0f}K)")
    logger.info(f"  MultiOutput水稻: {multioutput_total_rice:,} ({multioutput_total_rice/1000:.1f}K)")
    logger.info(f"  参数量减少: {(1 - multioutput_total_rice/ensemble_total_rice)*100:.1f}%")
    
    # 训练特性对比
    logger.info(f"\n训练特性对比:")
    logger.info(f"  Ensemble: 13个(番茄)/8个(水稻)独立训练的模型")
    logger.info(f"  MultiOutput: 1个模型同时学习所有任务")
    logger.info(f"  优化器: Adam (相同)")
    logger.info(f"  学习率: 0.001 (相同)")
    logger.info(f"  批大小: 32 (相同)")
    logger.info(f"  早停策略: 基于验证损失 (相同)")
    
    return {
        'ensemble_single_params': ensemble_single_model,
        'multioutput_tomato_params': multioutput_total_tomato,
        'multioutput_rice_params': multioutput_total_rice,
        'architecture_identical': True,
        'only_output_layer_differs': True,
        'parameter_sharing': 'MultiOutput共享所有隐藏层，Ensemble完全独立'
    }

def main():
    """主函数"""
    # 设置配置参数
    config = {
        # 数据路径和结果目录
        # 'data_path': 'data/Tomato_subsets/dataset_all_per_class.csv',  # 番茄数据集
        # 'results_dir': 'results/ML_baseline/tomato_dnn',  # 结果保存目录
        # 'crop_type': 'tomato',  # 作物类型，可选 'tomato' 或 'rice'
        'data_path': 'data/Rice_subsets/rice dataset_all_per_class.csv',  # 水稻数据集
        'results_dir': 'results/ML_baseline/rice_dnn',  # 水稻结果保存目录
        'crop_type': 'rice',  # 作物类型
        
        'model_type': 'multioutput_dnn',
        
        # 模型超参数 - 与Ensemble保持一致
        'use_batch_norm': True,
        
        # 训练超参数 - 更稳定的设置
        'num_epochs': 300,  # 增加训练轮数
        'batch_size': 32,
        'learning_rate': 0.001,  # 使用相同的学习率
        'weight_decay': 1e-5,
        'patience': 30  # 增加patience
    }
    
    # 设置日志系统
    logger = setup_logger(config)
    
    # 设置随机种子
    set_seed(42)
    
    logger.info("=== 开始训练MultiOutput DNN模型 ===")
    logger.info(f"作物类型: {config['crop_type']}")
    logger.info(f"数据路径: {config['data_path']}")
    
    # 生成模型统计信息
    logger.info("生成MultiOutput DNN模型统计信息...")
    generate_multioutput_dnn_stats(logger)
    
    # 训练模型
    model, train_results, val_results, test_results = train_multioutput_dnn(config, logger)
    
    logger.info("=== MultiOutput DNN模型训练完成 ===")

if __name__ == "__main__":
    main()
