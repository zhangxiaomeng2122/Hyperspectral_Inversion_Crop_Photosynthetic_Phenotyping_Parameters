import os
os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.multiprocessing as mp
import sys
import datetime
import logging
from torch.cuda.amp import GradScaler, autocast
import time

# 设置多处理方法
mp.set_start_method('spawn', force=True)

# 导入自定义模块
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) #e:\pycharm_program\MT_hyperspectral_inversion
sys.path.append(root_dir)
from utils.plot_setting import setfig

from mt_hyperspectral.data.dataset import HSIDataset
from mt_hyperspectral.training.trainer import train_multitask_model
from utils.baseset import setup_logger, set_seed, count_parameters
from mt_hyperspectral.data.data_split import random_train_val_test_split, stratified_train_val_test_split
from mt_hyperspectral.models.MTI_model import MTIEncoderBlock, MultiTaskTMI

# 创建CSV数据集类
class CSVDataset(Dataset):
    """处理CSV格式的多任务回归数据集"""
    def __init__(self, data, feature_cols, target_cols, indices=None, normalize=True, device=None):
        """
        Args:
            data (DataFrame): 包含特征和目标的DataFrame
            feature_cols (list): 特征列的索引列表
            target_cols (list): 目标列的名称列表
            indices (array): 要使用的样本索引
            normalize (bool): 是否标准化特征
            device (torch.device): 使用的设备
        """
        self.data = data
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.device = device
        
        # 如果提供了索引，筛选数据
        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)
        
        # 提取特征和目标
        self.features = self.data.iloc[:, feature_cols].values.astype(np.float32)
        self.targets = self.data[target_cols].values.astype(np.float32)
        
        # 标准化处理
        if normalize:
            self.feature_scaler = StandardScaler()
            self.features = self.feature_scaler.fit_transform(self.features)
            
            self.target_scaler = StandardScaler()
            self.targets = self.target_scaler.fit_transform(self.targets)
            
            # 保存均值和标准差，用于反标准化
            self.target_mean = self.target_scaler.mean_
            self.target_std = self.target_scaler.scale_
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        if self.device is not None:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
        return features, targets


    def __init__(self, input_size=273, hidden_dim=512, num_layers=3, num_heads=8, num_tasks=13, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        
        # 输入投影层，将特征投影到隐藏维度
        self.input_proj = nn.Linear(input_size, hidden_dim)
        
        # # 位置编码（可学习的）
        # self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # ViT编码器层
        self.encoders = nn.Sequential(
            *[MTIEncoderBlock(dim=hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop) 
              for _ in range(num_layers)]
        )
        
        # 任务特定解码器 - 为处理更复杂的13个参数，增加中间层维度
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

    def forward(self, x):
        # x形状: [batch_size, input_size]
        batch_size = x.shape[0]
        
        # 投影到隐藏维度
        x = self.input_proj(x)  # [batch_size, hidden_dim]
        
        # 添加批次维度以适应Transformer的输入格式
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # # 添加位置编码
        # x = x + self.pos_embedding
        
        # 通过Transformer编码器
        x = self.encoders(x)  # [batch_size, 1, hidden_dim]
        
        # 提取CLS token (第一个token)
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # 通过解码器获得多任务预测
        output = self.decoder(x)  # [batch_size, num_tasks]
        
        return output

def evaluate_model(model, dataloader, device, num_tasks):
    """评估模型在给定数据集上的性能"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in dataloader:
            with autocast():
                outputs = model(features)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    r2, rmse, rpd = calculate_metrics(all_targets, all_preds)
    
    return r2, rmse, rpd

def calculate_metrics(targets, preds):
    """计算R2, RMSE和RPD指标"""
    r2 = r2_score(targets, preds, multioutput='raw_values')
    mse = [mean_squared_error(targets[:,i], preds[:,i]) for i in range(targets.shape[1])]
    rmse = [np.sqrt(m) for m in mse]
    std = [np.std(targets[:,i]) for i in range(targets.shape[1])]
    rpd = [std[i]/rmse[i] for i in range(len(std))]
    return r2, rmse, rpd

def format_metrics_table(task_names, r2_values, rmse_values, rpd_values):
    """将指标格式化为表格形式"""
    # 创建表格头部
    headers = ["指标"] + task_names + ["平均"]
    r2_row = ["R²"] + [f"{val:.4f}" for val in r2_values] + [f"{np.mean(r2_values):.4f}"]
    rmse_row = ["RMSE"] + [f"{val:.4f}" for val in rmse_values] + [f"{np.mean(rmse_values):.4f}"]
    rpd_row = ["RPD"] + [f"{val:.4f}" for val in rpd_values] + [f"{np.mean(rpd_values):.4f}"]
    
    # 计算每列最大宽度
    widths = []
    for col_idx in range(len(headers)):
        col_items = [headers[col_idx], r2_row[col_idx], rmse_row[col_idx], rpd_row[col_idx]]
        widths.append(max(len(item) for item in col_items) + 2)
    
    # 创建分隔线
    separator = "+" + "+".join("-" * width for width in widths) + "+"
    
    # 格式化每行
    def format_row(row):
        return "|" + "|".join(f"{item:^{widths[i]}}" for i, item in enumerate(row)) + "|"
    
    # 组装表格
    table = [
        separator,
        format_row(headers),
        separator,
        format_row(r2_row),
        format_row(rmse_row),
        format_row(rpd_row),
        separator
    ]
    
    return "\n".join(table)

def save_training_history(history, task_names, results_dir):
    """保存训练历史记录到CSV文件"""
    # 保存总体指标历史记录
    main_metrics = {
        'epoch': history['epoch'],
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_r2': history['train_r2'],
        'val_r2': history['val_r2'],
        'train_rmse': history['train_rmse'],
        'val_rmse': history['val_rmse'],
        'train_rpd': history['train_rpd'],
        'val_rpd': history['val_rpd'],
        'lr': history['lr']
    }
    
    history_df = pd.DataFrame(main_metrics)
    history_path = os.path.join(results_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False, encoding='utf-8')
    
    # 为每个任务单独保存指标历史记录
    for i, task in enumerate(task_names):
        task_metrics = {
            'epoch': history['epoch'],
            'train_r2': history['task_train_r2'][i],
            'val_r2': history['task_val_r2'][i],
            'train_rmse': history['task_train_rmse'][i],
            'val_rmse': history['task_val_rmse'][i],
            'train_rpd': history['task_train_rpd'][i],
            'val_rpd': history['task_val_rpd'][i]
        }
        task_df = pd.DataFrame(task_metrics)
        task_history_path = os.path.join(results_dir, f'training_history_{task}.csv')
        task_df.to_csv(task_history_path, index=False, encoding='utf-8')


def generate_multitask_tmi_stats(model, config, logger=None):
    """
    生成MultiTaskTMI模型的详细统计信息
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
    logger.info("MultiTaskTMI模型详细统计")
    logger.info("="*80)
    
    # 模型架构分析
    input_size = len(config['feature_cols'])
    hidden_dim = config['hidden_dim']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    num_tasks = len(config['target_cols'])
    
    # 计算各层参数量
    input_proj_params = input_size * hidden_dim + hidden_dim
    
    # Transformer编码器参数
    attention_params_per_layer = (hidden_dim * hidden_dim * 3 +  # Q,K,V projections
                                 hidden_dim * hidden_dim +       # output projection
                                 hidden_dim * 2)                 # LayerNorm1
    
    mlp_params_per_layer = (hidden_dim * (hidden_dim * 4) +     # 第一个MLP层
                           (hidden_dim * 4) * hidden_dim +      # 第二个MLP层
                           hidden_dim * 2)                      # LayerNorm2
    
    encoder_params_per_layer = attention_params_per_layer + mlp_params_per_layer
    total_encoder_params = encoder_params_per_layer * num_layers
    
    # 解码器参数
    decoder_params = (hidden_dim * 2 +                          # LayerNorm
                     hidden_dim * hidden_dim + hidden_dim +     # 第一层
                     hidden_dim * (hidden_dim // 2) + (hidden_dim // 2) +  # 第二层
                     (hidden_dim // 2) * num_tasks + num_tasks)  # 输出层
    
    total_params = input_proj_params + total_encoder_params + decoder_params
    
    logger.info("MultiTaskTMI架构统计:")
    logger.info(f"  输入投影: {input_size}→{hidden_dim} ({input_proj_params:,}参数)")
    logger.info(f"  Transformer编码器: {num_layers}层×{num_heads}头")
    logger.info(f"    - 注意力机制: {attention_params_per_layer:,}参数/层")
    logger.info(f"    - MLP层: {mlp_params_per_layer:,}参数/层")
    logger.info(f"    - 编码器总计: {total_encoder_params:,}参数")
    logger.info(f"  多任务解码器: {hidden_dim}→{hidden_dim//2}→{num_tasks} ({decoder_params:,}参数)")
    logger.info(f"  总参数量: {total_params:,} ({total_params/1000:.1f}K)")
    
    # 内存使用分析
    param_memory = total_params * 4 / (1024**2)  # MB
    activation_memory = (hidden_dim * 4 + hidden_dim * num_heads) * 4 / (1024**2)  # MB
    total_memory = param_memory + activation_memory
    
    logger.info(f"\n内存使用分析:")
    logger.info(f"  参数内存: {param_memory:.1f} MB")
    logger.info(f"  激活内存: {activation_memory:.1f} MB")
    logger.info(f"  总内存需求: {total_memory:.1f} MB")
    
    # 模型特性
    logger.info(f"\n模型特性:")
    logger.info(f"  架构类型: Transformer for Tabular Data")
    logger.info(f"  学习方式: 多任务学习 (单模型→{num_tasks}参数)")
    logger.info(f"  注意力机制: {num_heads}头自注意力")
    logger.info(f"  特征学习: 全局依赖关系建模")
    logger.info(f"  正则化: LayerNorm + Dropout({config['drop_rate']})")
    
    # 详细说明权重调整机制
    logger.info(f"\n权重调整机制:")
    logger.info(f"  1. 注意力权重: 由MultiheadAttention自动学习，动态关注重要特征")
    logger.info(f"  2. 任务权重: 训练过程中自适应调整各任务的损失权重")
    logger.info(f"  3. 特征权重: 通过注意力机制实现特征间的动态加权")
    logger.info(f"  4. 层次权重: 残差连接实现不同层级特征的自适应融合")
    
    # 详细说明损失函数设计
    logger.info(f"\n损失函数设计:")
    logger.info(f"  基础损失: MSE损失 - L = Σᵢ ||yᵢ - ŷᵢ||²")
    logger.info(f"  任务权重: wᵢ参数 - 加权损失 = Σᵢ wᵢ × MSEᵢ")
    logger.info(f"  正则化项: 0.01 × Σᵢ (log(wᵢ) + 1/wᵢ)")
    logger.info(f"  总损失: L_total = Σᵢ wᵢ × MSEᵢ + λ × Σᵢ (log(wᵢ) + 1/wᵢ)")
    
    # 详细说明优化器配置
    logger.info(f"\n优化器配置:")
    logger.info(f"  主优化器: AdamW (lr={config.get('learning_rate', 0.001)}, wd={config.get('weight_decay', 5e-5)})")
    logger.info(f"  权重优化器: Adam (任务权重专用)")
    logger.info(f"  学习率调度: ReduceLROnPlateau (factor=0.5, patience={config.get('patience', 30)//2})")
    logger.info(f"  梯度缩放: AMP混合精度训练")
    
    return {
        'total_params': total_params,
        'param_memory_mb': param_memory,
        'total_memory_mb': total_memory,
        'architecture_details': {
            'input_proj_params': input_proj_params,
            'encoder_params': total_encoder_params,
            'decoder_params': decoder_params,
            'num_tasks': num_tasks
        }
    }

def main():
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        
        # MT,逐像素反射率
        # # 'csv_path': 'data/Tomato_subsets/dataseall_per_class.csv',
        # 'target_cols': ['SPAD', 'Pn', 'LAW'],  # MT 3个参数
        # 'results_dir': os.path.join('results', 'MT csv_multitask_vit_3param'),   # MT 3结果目录名

        # 'target_cols': ['RUE', 'WUE', 'Pn', 'SPAD', 'LAW'],  # MT 5个参数  ['RUE', 'SPAD', 'LAW', 'WUE', 'Ic']
        # 'results_dir': os.path.join('results', 'MT csv_multitask_vit_5param'),   # MT 5 结果目录名
        
        # 'target_cols': ['RUE', 'Pn', 'Tr', 'WUE', 'Pmax','Ic', 'SPAD', 'LAW'],  # MT 8个参数 R²=0.9358
        # 'results_dir': os.path.join('results', 'MT csv_multitask_vit_8param'),   # MT 8 结果目录名

        # # 'target_cols': ['RUE', 'Pn', 'Gs', 'Ci', 'Tr', 'Ci-Ca', 'WUE', 'iWUE', 'Pmax', 'Rd', 'Ic', 'SPAD', 'LAW'],  # MT 13个参数
      #   # 'results_dir': os.path.join('results', 'MT csv_multitask_vit_13param'),   # MT 13 结果目录名

      'csv_path': 'data/Rice_subsets/rice dataset_max7_per_class.csv', # Rice,逐像素反射率
        # 'target_cols': ['SPAD','Pn','LAW'],  # Rice 3个水稻参数
        # 'results_dir': os.path.join('results', 'Rice csv_multitask_vit_3param'),  # Rice 结果目录名
        
        # 'target_cols': ['SPAD','Pn', 'LAW', 'LNC', 'Chl'],  # Rice 5个水稻参数
        # 'results_dir': os.path.join('results', 'Rice csv_multitask_vit_5param'),  # Rice 结果目录名
        
      'target_cols': ['SPAD','Pn', 'LNC', 'Chl-a', 'Chl-b','LAW', 'Cx', 'Chl'],  # Rice 8个水稻参数
      'results_dir': os.path.join('results', 'Rice csv_multitask_vit_8param'),  # Rice 结果目录名

        'feature_cols': list(range(3, 276)),  # 特征列的索引（从0开始），第4-276列
        'batch_size': 32,
        'num_epochs': 500, 
        'learning_rate': 0.001,
        'weight_decay': 5e-5,          
        
        'split_method': 'sklearn',  # 'random' 、 'stratified'或 'sklearn'，用于划分数据集
        'val_size': 0.15,               
        'test_size': 0.15,             
        'seed': 42,
        'hidden_dim': 128,              
        'num_layers': 1,               
        'num_heads': 1,                
        'drop_rate': 0.15,             
        'early_stopping': True,        
        'patience': 30,               
        'min_delta': 0.0005,          
        'metric_monitor': 'val_r2',      
        'compute_final_metrics': False,  
        'consistent_evaluation': True,  
    }
    
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config['results_dir'] = os.path.join(config['results_dir'], timestamp)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # 设置记录器和随机种子
    logger = setup_logger(config)
    set_seed(config['seed'])
    
    # 加载CSV数据
    logger.info(f"从 {config['csv_path']} 加载数据...")
    try:
        data = pd.read_csv(config['csv_path'])
        logger.info(f"成功加载数据，共 {len(data)} 条记录，{len(config['feature_cols'])} 个特征")
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        return
    
    # 检查目标列是否存在
    missing_cols = [col for col in config['target_cols'] if col not in data.columns]
    if missing_cols:
        logger.error(f"以下目标列不存在于数据中: {missing_cols}")
        return
    
    # 显示各列的基本统计信息
    logger.info("目标变量统计信息:")
    for col in config['target_cols']:
        logger.info(f"{col}: 均值={data[col].mean():.4f}, 标准差={data[col].std():.4f}, 范围=[{data[col].min():.4f}, {data[col].max():.4f}]")
    
    # 划分数据集 - 更新为三部分划分
    logger.info(f"使用 {config['split_method']} 方法将数据集划分为训练集、验证集和测试集")
    if config['split_method'] == 'stratified':
        train_indices, val_indices, test_indices = stratified_train_val_test_split(
            data, 
            config['target_cols'], 
            val_size=config['val_size'], 
            test_size=config['test_size'], 
            seed=config['seed']
        )
    elif config['split_method'] == 'random': 
        train_indices, val_indices, test_indices = random_train_val_test_split(
            data, 
            val_size=config['val_size'], 
            test_size=config['test_size'], 
            seed=config['seed']
        )
    else:  # sklearn method
        # 首先将数据分成训练集和临时集(临时集包含验证集和测试集)
        train_indices, temp_indices = train_test_split(
            np.arange(len(data)), 
            test_size=config['val_size'] + config['test_size'], 
            random_state=config['seed'], 
            stratify=data[config['target_cols']].values if config['split_method'] == 'sklearn_stratified' else None
        )
        
        # 确定验证集和测试集的比例
        val_ratio = config['val_size'] / (config['val_size'] + config['test_size'])
        
        # 然后将临时集分成验证集和测试集
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=1-val_ratio,  # 调整为验证集占比
            random_state=config['seed'],
            stratify=data.iloc[temp_indices][config['target_cols']].values if config['split_method'] == 'sklearn_stratified' else None
        )
        
        logger.info(f"使用sklearn方法划分数据集: 训练集{len(train_indices)}样本, 验证集{len(val_indices)}样本, 测试集{len(test_indices)}样本")
    
    # 创建训练集、验证集和测试集
    train_dataset = CSVDataset(
        data, 
        config['feature_cols'], 
        config['target_cols'], 
        indices=train_indices, 
        normalize=True, 
        device=config['device']
    )
    
    val_dataset = CSVDataset(
        data, 
        config['feature_cols'], 
        config['target_cols'], 
        indices=val_indices, 
        normalize=True, 
        device=config['device']
    )
    
    test_dataset = CSVDataset(
        data, 
        config['feature_cols'], 
        config['target_cols'], 
        indices=test_indices, 
        normalize=True, 
        device=config['device']
    )
    
    logger.info(f"数据集划分: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本, 测试集 {len(test_dataset)} 样本")
    
    # 数据加载器 - 确保评估时不会打乱数据顺序
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    # 验证和测试集明确设置shuffle=False，确保预测值和目标值顺序一致
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # 构建模型 - 为13个参数更新
    input_size = len(config['feature_cols'])
    num_tasks = len(config['target_cols'])  # 现在是13
    
    model = MultiTaskTMI(
        input_size=input_size,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        num_tasks=num_tasks,
        drop=config['drop_rate']
    ).to(config['device'])
    
    # 计算和输出模型参数数量
    num_params = count_parameters(model)
    logger.info(f"模型结构:\n{model}")
    logger.info(f"模型总参数量: {num_params:,} 参数")
    logger.info(f"模型参数占用内存: {num_params * 4 / (1024 * 1024):.2f} MB")
    
    # 保存数据划分索引
    indices_file = os.path.join(config['results_dir'], 'data_indices.npz')
    np.savez(indices_file, 
             train_indices=train_indices, 
             val_indices=val_indices, 
             test_indices=test_indices)
    logger.info(f"保存数据划分索引到 {indices_file}")
    
    # 使用多任务训练函数训练模型
    results = train_multitask_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        task_names=config['target_cols'],
        device=config['device'],
        results_dir=config['results_dir'],
        logger=logger,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        patience=config['patience'],
        min_delta=config['min_delta'],
        early_stopping=config['early_stopping'],
        metric_monitor=config['metric_monitor'],
        compute_final_metrics=config.get('compute_final_metrics', False),
        consistent_evaluation=config.get('consistent_evaluation', True),
    )
    
    # 获取训练结果
    metrics = results['metrics']
    
    # 详细输出各参数性能
    logger.info("\n=================== 各参数测试集性能 ===================")
    logger.info("参数         R²          RMSE        RPD")
    logger.info("-------------------------------------------------")
    for i, task in enumerate(config['target_cols']):
        logger.info(f"{task:<12} {metrics['test_r2'][i]:.4f}     {metrics['test_rmse'][i]:.4f}     {metrics['test_rpd'][i]:.4f}")
    logger.info("-------------------------------------------------")
    logger.info(f"平均        {np.mean(metrics['test_r2']):.4f}     {np.mean(metrics['test_rmse']):.4f}     {np.mean(metrics['test_rpd']):.4f}")
    logger.info("=================================================")
    
    # 最终信息
    logger.info(f"训练完成! 最佳验证集R²: {metrics['val_mean_r2']:.4f}, 测试集R²: {metrics['test_mean_r2']:.4f}")
    
    # 在模型创建后生成统计信息
    logger.info("生成MultiTaskTMI模型统计信息...")
    model_stats = generate_multitask_tmi_stats(model, config, logger)
    
    # 保存预测结果详细对比
    logger.info("保存预测结果详细对比...")
    predictions_file = save_predictions_comparison(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        task_names=config['target_cols'],
        device=config['device'],
        results_dir=config['results_dir'],
        logger=logger
    )
    
    # 在训练完成后输出统计摘要
    logger.info(f"\nMultiTaskTMI训练完成统计:")
    logger.info(f"  模型参数量: {model_stats['total_params']:,}")
    logger.info(f"  内存占用: {model_stats['total_memory_mb']:.1f} MB")
    logger.info(f"  预测任务数: {model_stats['architecture_details']['num_tasks']}")
    logger.info(f"  最终测试R²: {metrics['test_mean_r2']:.4f}")
    logger.info(f"  预测结果文件: {predictions_file}")
    
    print(f"所有结果已保存到: {config['results_dir']}")

def save_predictions_comparison(model, train_loader, val_loader, test_loader, task_names, device, results_dir, logger):
    """
    保存所有数据集的真值和预测值对比到CSV文件，包含输入光谱数据
    """
    logger.info("开始保存预测结果详细对比...")
    
    model.eval()
    all_predictions = []
    
    # 处理训练集
    logger.info("处理训练集预测...")
    with torch.no_grad():
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            with autocast():
                outputs = model(features)
            
            # 转换到CPU并添加数据集标签
            batch_size = outputs.shape[0]
            dataset_labels = ['train'] * batch_size
            
            for i in range(batch_size):
                row_data = {'dataset': dataset_labels[i]}
                
                # 添加输入光谱数据 (273个波段)
                for j in range(features.shape[1]):
                    row_data[f'band_{j+1}'] = features[i, j].cpu().numpy()
                
                # 添加每个参数的真值和预测值
                for j, task in enumerate(task_names):
                    row_data[f'{task}_true'] = targets[i, j].cpu().numpy()
                    row_data[f'{task}_pred'] = outputs[i, j].cpu().numpy()
                
                all_predictions.append(row_data)
    
    # 处理验证集
    logger.info("处理验证集预测...")
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            with autocast():
                outputs = model(features)
            
            # 转换到CPU并添加数据集标签
            batch_size = outputs.shape[0]
            dataset_labels = ['validation'] * batch_size
            
            for i in range(batch_size):
                row_data = {'dataset': dataset_labels[i]}
                
                # 添加输入光谱数据
                for j in range(features.shape[1]):
                    row_data[f'band_{j+1}'] = features[i, j].cpu().numpy()
                
                # 添加每个参数的真值和预测值
                for j, task in enumerate(task_names):
                    row_data[f'{task}_true'] = targets[i, j].cpu().numpy()
                    row_data[f'{task}_pred'] = outputs[i, j].cpu().numpy()
                
                all_predictions.append(row_data)
    
    # 处理测试集
    logger.info("处理测试集预测...")
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            with autocast():
                outputs = model(features)
            
            # 转换到CPU并添加数据集标签
            batch_size = outputs.shape[0]
            dataset_labels = ['test'] * batch_size
            
            for i in range(batch_size):
                row_data = {'dataset': dataset_labels[i]}
                
                # 添加输入光谱数据
                for j in range(features.shape[1]):
                    row_data[f'band_{j+1}'] = features[i, j].cpu().numpy()
                
                # 添加每个参数的真值和预测值
                for j, task in enumerate(task_names):
                    row_data[f'{task}_true'] = targets[i, j].cpu().numpy()
                    row_data[f'{task}_pred'] = outputs[i, j].cpu().numpy()
                
                all_predictions.append(row_data)
    
    # 创建DataFrame并保存
    predictions_df = pd.DataFrame(all_predictions)
    
    # 重新排列列顺序：数据集列在前，然后是光谱数据，最后是每个参数的true和pred交替排列
    column_order = ['dataset']
    
    # 添加所有光谱波段列
    num_bands = len([col for col in predictions_df.columns if col.startswith('band_')])
    for i in range(1, num_bands + 1):
        column_order.append(f'band_{i}')
    
    # 添加参数的真值和预测值列
    for task in task_names:
        column_order.extend([f'{task}_true', f'{task}_pred'])
    
    predictions_df = predictions_df[column_order]
    
    # 保存到CSV文件
    predictions_file = os.path.join(results_dir, 'predictions_comparison.csv')
    predictions_df.to_csv(predictions_file, index=False, encoding='utf-8')
    
    logger.info(f"预测结果对比已保存到: {predictions_file}")
    logger.info(f"总共保存了 {len(predictions_df)} 条记录")
    logger.info(f"包含 {num_bands} 个光谱波段和 {len(task_names)} 个预测参数")
    
    # 输出各数据集的样本数量统计
    dataset_counts = predictions_df['dataset'].value_counts()
    logger.info("各数据集样本数量:")
    for dataset, count in dataset_counts.items():
        logger.info(f"  {dataset}: {count} 样本")
    
    return predictions_file

if __name__ == "__main__":
    main()