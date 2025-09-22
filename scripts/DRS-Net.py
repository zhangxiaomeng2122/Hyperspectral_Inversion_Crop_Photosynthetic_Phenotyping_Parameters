import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.multiprocessing as mp
import sys
import datetime
import logging
from torch.cuda.amp import GradScaler, autocast
import time


# 导入 MultiTaskViT 模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from utils.plot_setting import setfig
from mt_hyperspectral.utils.baseset import setup_logger, set_seed, count_parameters
from mt_hyperspectral.data.data_split import random_train_val_test_split, stratified_train_val_test_split
from scripts.MTI_Net import MultiTaskViT, CSVDataset, evaluate_model, calculate_metrics, format_metrics_table
from mt_hyperspectral.training.trainer import get_predictions_and_targets

# 1. BandwiseAffineLayer
class BandwiseAffineLayer(nn.Module):#BandwiseAffineLayer 
    """
    为每个波段应用独立的权重和偏置，实现波段选择功能。
    对每个波段应用独立的1x1卷积，严格按照论文公式实现。
    """
    def __init__(self, num_bands):
        super(BandwiseAffineLayer, self).__init__()
        self.num_bands = num_bands
        # 改进权重初始化
        self.weights = nn.Parameter(torch.ones(num_bands) + 0.01 * torch.randn(num_bands))
        self.bias = nn.Parameter(torch.zeros(num_bands))
        # 添加批量归一化，提高训练稳定性
        self.batch_norm = nn.BatchNorm1d(num_bands)
        
    def forward(self, x):
        # 先进行批量归一化
        if self.training:
            x = self.batch_norm(x)
        # 然后应用权重和偏置
        weighted_bands = x * self.weights + self.bias
        return weighted_bands

    def get_band_importance(self):
        """获取每个波段的重要性权重"""
        # 返回权重绝对值作为波段重要性
        return torch.abs(self.weights).detach().cpu().numpy()

# 2. 定义硬阈值选择机制
class TopKBandGatingLayer(nn.Module):
    """
    基于BandwiseIndependentConv学习的权重进行波段选择
    严格按照论文中的硬阈值方法实现
    """
    def __init__(self, num_bands, k_bands):
        """
        Args:
            num_bands: 总波段数
            k_bands: 要选择的波段数量 (论文中的u参数)
        """
        super(TopKBandGatingLayer, self).__init__()
        self.num_bands = num_bands
        self.k_bands = k_bands
            
    def forward(self, x, weights, bias):
        """
        实现论文中的公式(1)
        Args:
            x: 输入特征 [batch_size, num_bands]
            weights: 波段权重 [num_bands]
            bias: 波段偏置 [num_bands]
        """
        # 计算波段重要性
        band_importance = torch.abs(weights)
        
        # 根据论文中的硬阈值方法，获取第(u+1)个最大值作为阈值
        # 即排序后，第k个位置的值（从0开始计数）
        sorted_importance, _ = torch.sort(band_importance, descending=True)
        if self.k_bands < self.num_bands:
            threshold = sorted_importance[self.k_bands-1]  # 第k个最大值作为阈值
        else:
            threshold = torch.min(band_importance) - 1e-6  # 如果k等于波段总数，选择所有波段
        
        # 创建掩码，仅保留重要性大于阈值的波段
        mask = (band_importance >= threshold).float()
        
        # 实现公式(1)的选择逻辑:
        # 对于重要的波段: w_k ⊙ x_i,k + b_k
        # 对于不重要的波段: b_k
        selected_features = torch.zeros_like(x)
        important_bands = torch.where(mask > 0)[0]
        
        # 对重要波段应用权重和偏置
        selected_features[:, important_bands] = x[:, important_bands] * weights[important_bands] + bias[important_bands]
        
        # 对不重要波段只保留偏置
        unimportant_bands = torch.where(mask == 0)[0]
        if len(unimportant_bands) > 0:
            selected_features[:, unimportant_bands] = bias[unimportant_bands]
        
        return selected_features, mask

# 3. 构建完整的BHCNN回归模型
class BHCNNRegression(nn.Module):
    def __init__(self, input_size, num_tasks, k_bands=None, threshold=None, hidden_dims=(128, 64)):
        super(BHCNNRegression, self).__init__()
        
        self.input_size = input_size
        self.num_tasks = num_tasks
        
        # 波段独立卷积层，学习波段重要性
        self.bandwise_conv = BandwiseAffineLayer(input_size)
        
        # 波段选择层
        self.band_selection = TopKBandGatingLayer(input_size, k_bands, threshold)
        
        # 特征提取层
        layers = []
        prev_dim = input_size  # 输入维度是选择后的波段数
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        self.feature_extraction = nn.Sequential(*layers)
        
        # 回归输出层
        self.regression_head = nn.Linear(prev_dim, num_tasks)
        
    def forward(self, x):
        # 应用波段权重
        weighted_bands = self.bandwise_conv(x)
        
        # 在训练模式下，不进行硬选择，让所有波段参与训练
        if self.training:
            features = weighted_bands
        else:
            # 在推理模式下，应用硬选择
            features, _ = self.band_selection(x, self.bandwise_conv.weights)
        
        # 特征提取
        features = self.feature_extraction(features)
        
        # 回归输出
        outputs = self.regression_head(features)
        
        return outputs
    
    def get_selected_bands(self):
        """获取模型选择的波段索引"""
        band_importance = self.bandwise_conv.get_band_importance()
        
        if hasattr(self.band_selection, 'k_bands') and self.band_selection.k_bands is not None:
            # 如果设置了k_bands，返回最重要的k个波段
            top_indices = np.argsort(band_importance)[::-1][:self.band_selection.k_bands]
            return top_indices
        else:
            # 基于阈值返回重要波段
            threshold = self.band_selection.threshold
            selected_indices = np.where(band_importance > threshold)[0]
            return selected_indices
            
# 4. 将 BandwiseAffineLayer 与 MultiTaskViT 集成的新模型，采用CFL双分支架构
class BandwiseCFLViT(nn.Module):
    """
    结合波段选择和 ViT 模型的架构，采用CFL (Coarse-to-Fine Loss)双分支结构
    """
    def __init__(self, input_size, num_tasks, k_bands=None,
                hidden_dim=512, num_layers=4, num_heads=8, drop_rate=0.1):
        super(BandwiseCFLViT, self).__init__()
        
        self.input_size = input_size
        self.num_tasks = num_tasks
        
        # 波段独立卷积层 - 共享权重用于两个分支
        self.bandwise_conv = BandwiseAffineLayer(input_size)
        
        # 硬阈值选择层 - 仅用于选择波段分支
        self.band_selection = TopKBandGatingLayer(input_size, k_bands)

        # 共享的 ViT 回归模型 - 用于两个分支
        self.vit_model = MultiTaskViT(
            input_size=input_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_tasks=num_tasks,
            drop=drop_rate
        )
        
        # 当前迭代数和总迭代数，用于计算粗到细损失的权重因子
        self.current_iter = 0
        self.total_iters = 0
        
    def forward(self, x, return_extra=True):
        """
        前向传播函数 - 实现双分支结构
        
        Args:
            x: 输入特征 [batch_size, num_bands]
            return_extra: 是否返回额外分支的输出，训练时为True，推理时为False
            
        Returns:
            outputs: 原始分支的输出
            outputs_extra: 额外分支的输出(如果return_extra为True)
        """
        # 全波段分支 - 应用波段权重后直接输入回归模型
        weighted_bands = self.bandwise_conv(x)
        coarse_output = self.vit_model(weighted_bands)
        
        # 波段选择分支 - 应用波段权重后进行硬阈值选择
        selected_bands, _ = self.band_selection(x, self.bandwise_conv.weights, self.bandwise_conv.bias)
        fine_output = self.vit_model(selected_bands)
        
        if self.training and return_extra:
            # 训练时，返回两个分支的输出供损失函数使用
            # fine_output - 选择波段后的输出 (更精细的输出)
            # coarse_output - 全波段输出 (更粗糙的输出)
            return fine_output, coarse_output
        else:
            # 推理时只使用选择波段后的输出
            return fine_output
    
    def get_selected_bands(self):
        """获取模型选择的波段索引"""
        band_importance = self.bandwise_conv.get_band_importance()
        
        # 根据波段重要性排序
        sorted_indices = np.argsort(band_importance)[::-1]
        
        # 返回最重要的k个波段
        return sorted_indices[:self.band_selection.k_bands]
    
    def set_iteration_params(self, current, total):
        """设置当前迭代和总迭代数，用于计算粗到细损失权重"""
        self.current_iter = current
        self.total_iters = total

# 添加粗到细损失函数
class CoarseToFineLoss(nn.Module):
    """
    实现论文中的粗到细损失函数
    随着训练进行，逐步将焦点从粗粒度损失(所有波段)转移到细粒度损失(选择的波段)
    """
    def __init__(self, base_criterion=nn.MSELoss()):
        super(CoarseToFineLoss, self).__init__()
        self.base_criterion = base_criterion
    
    def forward(self, fine_outputs, coarse_outputs, targets, current_iter, total_iters):
        """
        计算粗到细损失
        
        Args:
            fine_outputs: 细粒度输出（选择波段后）
            coarse_outputs: 粗粒度输出（所有波段）
            targets: 目标值
            current_iter: 当前迭代数
            total_iters: 总迭代数
            
        Returns:
            loss: 组合损失
        """
        # 计算调整因子 σ
        sigma = 1.0 - current_iter / total_iters
        
        # 计算细粒度损失（选择波段后）
        fine_loss = self.base_criterion(fine_outputs, targets)
        
        # 计算粗粒度损失（所有波段）
        coarse_loss = self.base_criterion(coarse_outputs, targets)
        
        # 组合损失
        loss = sigma * coarse_loss + (1.0 - sigma) * fine_loss
        
        return loss, coarse_loss, fine_loss, sigma

# 修改训练函数以支持CFL模型
def train_cfl_model(model, train_loader, val_loader, test_loader, task_names, device, results_dir, logger,
                     num_epochs=500, learning_rate=0.001, weight_decay=1e-5, patience=40, min_delta=0.0005,
                     early_stopping=True, metric_monitor='val_r2'):
    """训练BHCNN-CFL模型，使用粗到细损失函数"""
    
    # 优化器 - 使用AdamW而非Adam，通常有更好的性能
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 基础损失函数 - 考虑使用SmoothL1Loss以增强对异常值的鲁棒性
    base_criterion = nn.SmoothL1Loss()
    
    # 粗到细损失函数
    cfl_criterion = CoarseToFineLoss(base_criterion)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience//2, 
                                                   factor=0.5, min_lr=1e-6)
    
    # 计算总训练迭代次数
    total_iters = num_epochs * len(train_loader)
    current_iter = 0
    
    # 初始化早停参数
    best_val_metric = -float('inf')
    best_epoch = 0
    no_improve_count = 0
    best_model_path = os.path.join(results_dir, 'best_model.pth')
    
    # 初始化训练历史记录
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_r2': [], 'val_r2': [], 'test_r2': [],
        'train_rmse': [], 'val_rmse': [], 'test_rmse': [],
        'train_rpd': [], 'val_rpd': [], 'test_rpd': [],
        'lr': [], 'sigma': [],  # 添加sigma记录
        'coarse_loss': [], 'fine_loss': [],  # 添加粗细损失记录
        # 任务特定指标
        'task_train_r2': [[] for _ in range(len(task_names))],
        'task_val_r2': [[] for _ in range(len(task_names))],
        'task_test_r2': [[] for _ in range(len(task_names))],
        'task_train_rmse': [[] for _ in range(len(task_names))],
        'task_val_rmse': [[] for _ in range(len(task_names))],
        'task_test_rmse': [[] for _ in range(len(task_names))],
        'task_train_rpd': [[] for _ in range(len(task_names))],
        'task_val_rpd': [[] for _ in range(len(task_names))],
        'task_test_rpd': [[] for _ in range(len(task_names))],
    }
    
    # 创建梯度缩放器用于混合精度训练
    scaler = GradScaler()
    
    # 主训练循环
    logger.info("开始训练...")
    start_time = time.time()

    # 定义学习率预热轮数
    warmup_epochs = 10
    
    # 添加调试信息
    logger.info("模型训练配置:")
    logger.info(f"学习率: {learning_rate}, 权重衰减: {weight_decay}")
    logger.info(f"预热轮数: {warmup_epochs}, 总训练轮数: {num_epochs}")
    logger.info(f"批量大小: {train_loader.batch_size}, 总训练样本: {len(train_loader.dataset)}")
    logger.info(f"共享波段卷积层权重，使用粗到细损失函数(CFL)进行训练")
    logger.info(f"选择波段数量: {model.band_selection.k_bands}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_losses = []
        coarse_losses = []
        fine_losses = []
        sigma_values = []
        
        # 学习率预热
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * ((epoch + 1) / warmup_epochs)
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            # 更新当前迭代计数
            current_iter += 1
            
            # 设置模型的迭代参数
            model.set_iteration_params(current_iter, total_iters)
            
            # 确保数据在正确的设备上
            if features.device != device:
                features = features.to(device)
            if targets.device != device:
                targets = targets.to(device)
                
            optimizer.zero_grad()
            
            # 使用混合精度并添加梯度裁剪
            with autocast():
                # 前向传播 - 获取双分支输出
                # fine_output - 选择波段后的输出
                # coarse_output - 全波段输出
                fine_output, coarse_output = model(features, return_extra=True)
                
                # 计算CFL损失
                loss, coarse_loss, fine_loss, sigma = cfl_criterion(
                    fine_output, coarse_output, targets, current_iter, total_iters)
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 添加梯度裁剪，防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 记录损失值
            train_losses.append(loss.item())
            coarse_losses.append(coarse_loss.item())
            fine_losses.append(fine_loss.item())
            sigma_values.append(sigma)
            
        # 计算平均损失
        train_loss = np.mean(train_losses)
        avg_coarse_loss = np.mean(coarse_losses)
        avg_fine_loss = np.mean(fine_losses)
        avg_sigma = np.mean(sigma_values)
        
        # 验证阶段
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                if features.device != device:
                    features = features.to(device)
                if targets.device != device:
                    targets = targets.to(device)
                    
                # 推理模式只使用fine输出
                outputs = model(features, return_extra=False)
                loss = base_criterion(outputs, targets)
                val_losses.append(loss.item())
        
        # 计算验证损失
        val_loss = np.mean(val_losses)
        
        # 计算各项指标
        train_r2, train_rmse, train_rpd = evaluate_model(model, train_loader, device, len(task_names))
        val_r2, val_rmse, val_rpd = evaluate_model(model, val_loader, device, len(task_names))
        test_r2, test_rmse, test_rpd = evaluate_model(model, test_loader, device, len(task_names))
        
        # 更新学习率
        current_val_metric = np.mean(val_r2) if metric_monitor == 'val_r2' else -np.mean(val_rmse)
        scheduler.step(current_val_metric)
        
        # 记录历史
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(np.mean(train_r2))
        history['val_r2'].append(np.mean(val_r2))
        history['test_r2'].append(np.mean(test_r2))
        history['train_rmse'].append(np.mean(train_rmse))
        history['val_rmse'].append(np.mean(val_rmse))
        history['test_rmse'].append(np.mean(test_rmse))
        history['train_rpd'].append(np.mean(train_rpd))
        history['val_rpd'].append(np.mean(val_rpd))
        history['test_rpd'].append(np.mean(test_rpd))
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['sigma'].append(avg_sigma)
        history['coarse_loss'].append(avg_coarse_loss)
        history['fine_loss'].append(avg_fine_loss)
        
        # 记录每个任务的指标
        for i in range(len(task_names)):
            history['task_train_r2'][i].append(train_r2[i])
            history['task_val_r2'][i].append(val_r2[i])
            history['task_test_r2'][i].append(test_r2[i])
            history['task_train_rmse'][i].append(train_rmse[i])
            history['task_val_rmse'][i].append(val_rmse[i])
            history['task_test_rmse'][i].append(test_rmse[i])
            history['task_train_rpd'][i].append(train_rpd[i])
            history['task_val_rpd'][i].append(val_rpd[i])
            history['task_test_rpd'][i].append(test_rpd[i])
        
        # 打印进度
        if (epoch+1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                      f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
                      f"粗损失: {avg_coarse_loss:.4f}, 细损失: {avg_fine_loss:.4f}, σ: {avg_sigma:.4f}, "
                      f"训练R??: {np.mean(train_r2):.4f}, 验证R??: {np.mean(val_r2):.4f}")
        
        # 格式化并输出指标
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")

        logger.info("训练集指标:")
        logger.info(format_metrics_table(task_names, train_r2, train_rmse, train_rpd))
        
        logger.info("验证集指标:")
        logger.info(format_metrics_table(task_names, val_r2, val_rmse, val_rpd))
        
        # 保存最佳模型
        if current_val_metric > best_val_metric + min_delta:
            best_val_metric = current_val_metric
            best_epoch = epoch
            no_improve_count = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metric': best_val_metric,
                'band_importance': model.bandwise_conv.get_band_importance(),
                'selected_bands': model.get_selected_bands(),
                'current_iter': current_iter,
                'total_iters': total_iters
            }, best_model_path)
            
            logger.info(f"Epoch {epoch+1}: 保存最佳模型，验证 {metric_monitor}: {best_val_metric:.4f}")
        else:
            no_improve_count += 1
        
        # 早停检查
        if early_stopping and no_improve_count >= patience:
            logger.info(f"早停触发！{patience}个轮次内无改善。")
            break
    
    # 加载最佳模型
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 计算最佳模型的所有指标
    train_r2, train_rmse, train_rpd = evaluate_model(model, train_loader, device, len(task_names))
    val_r2, val_rmse, val_rpd = evaluate_model(model, val_loader, device, len(task_names))
    test_r2, test_rmse, test_rpd = evaluate_model(model, test_loader, device, len(task_names))
    
    # 计算训练时间
    total_time = time.time() - start_time
    logger.info(f"训练完成！总用时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    logger.info(f"最佳模型来自第{best_epoch+1}轮，验证R??: {np.mean(val_r2):.4f}，测试R??: {np.mean(test_r2):.4f}")
    
    # 获取波段选择结果
    band_importance = checkpoint['band_importance']
    selected_bands = checkpoint['selected_bands']
    
    logger.info(f"波段选择结果：选择了{len(selected_bands)}个波段")
    
    # 保存波段重要性数据
    importance_df = pd.DataFrame({
        'Band_Index': np.arange(len(band_importance)),
        'Importance': band_importance
    })
    importance_df.to_csv(os.path.join(results_dir, 'band_importance.csv'), index=False)
    
    # 保存选择的波段
    selected_df = pd.DataFrame({
        'Selected_Band_Index': selected_bands
    })
    selected_df.to_csv(os.path.join(results_dir, 'selected_bands.csv'), index=False)
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': history['epoch'],
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_r2': history['train_r2'],
        'val_r2': history['val_r2'],
        'test_r2': history['test_r2'],
        'train_rmse': history['train_rmse'],
        'val_rmse': history['val_rmse'],
        'test_rmse': history['test_rmse'],
        'train_rpd': history['train_rpd'],
        'val_rpd': history['val_rpd'],
        'test_rpd': history['test_rpd'],
        'lr': history['lr'],
        'sigma': history['sigma'],
        'coarse_loss': history['coarse_loss'],
        'fine_loss': history['fine_loss']
    })
    history_df.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
    
    # 不再使用一个大的figure，而是为每个子图创建单独的figure并设置格式
    
    # 1. R?? 曲线图
    setfig(column=1, x=2.5, y=2.8)
    plt.plot(history['epoch'], history['train_r2'], label='Train R??')
    plt.plot(history['epoch'], history['val_r2'], label='Val R??')
    plt.plot(history['epoch'], history['test_r2'], label='Test R??')
    plt.xlabel('Epoch')
    plt.ylabel('R??')
    plt.legend(prop={'size':6, 'family': 'Arial'}, frameon=False) 
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'r2_history.pdf'), format='PDF', transparent=True, bbox_inches='tight')
    plt.close()
    
    # 2. RMSE 曲线图
    setfig(column=1, x=2.5, y=2.8)
    plt.plot(history['epoch'], history['train_rmse'], label='Train RMSE')
    plt.plot(history['epoch'], history['val_rmse'], label='Val RMSE')
    plt.plot(history['epoch'], history['test_rmse'], label='Test RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(prop={'size':6, 'family': 'Arial'}, frameon=False) 
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'rmse_history.pdf'), format='PDF', transparent=True, bbox_inches='tight')
    plt.close()
    
    # 3. 损失曲线图
    setfig(column=1, x=2.5, y=2.8)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(prop={'size':6, 'family': 'Arial'}, frameon=False) 
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'loss_history.pdf'), format='PDF', transparent=True, bbox_inches='tight')
    plt.close()
    
    # 4. 波段重要性条形图
    setfig(column=1, x=2.5, y=2.8)
    # 使用模型的k_bands属性或直接使用selected_bands的长度
    k_bands = model.band_selection.k_bands
    selected_bands_set = set(selected_bands[:k_bands])
    
    # 创建颜色列表，被选中的波段用红色，未选中的用蓝色
    colors = ['red' if i in selected_bands_set else 'blue' for i in range(len(band_importance))]
    
    # 绘制波段重要性条形图
    bars = plt.bar(np.arange(len(band_importance)), band_importance, color=colors)
    
    # 添加图例说明
    red_patch = plt.Rectangle((0, 0), 1, 1, fc="red")
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc="blue")
    plt.legend([red_patch, blue_patch], ['Selected', 'Not Selected'], prop={'size':6, 'family': 'Arial'}, frameon=False)

    plt.xlabel('Band Index')
    plt.ylabel('Importance')
    plt.title(f'Band Importance (Selected: {len(selected_bands_set)} bands)')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'band_importance.pdf'), format='PDF', transparent=True, bbox_inches='tight')
    plt.close()
    
    # 5. 粗到细损失曲线图
    setfig(column=1, x=2.5, y=2.8)
    plt.plot(history['epoch'], history['coarse_loss'], label='Coarse Loss')
    plt.plot(history['epoch'], history['fine_loss'], label='Fine Loss')
    plt.plot(history['epoch'], history['sigma'], label='Sigma')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Coarse-to-Fine Loss Transition')
    plt.legend(prop={'size':6, 'family': 'Arial'}, frameon=False) 
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'cfl_transition.pdf'), format='PDF', transparent=True, bbox_inches='tight')
    plt.close()
    
    # 返回结果
    results = {
        'metrics': {
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'test_rmse': test_rmse,
            'train_rpd': train_rpd,
            'val_rpd': val_rpd,
            'test_rpd': test_rpd,
            'train_mean_r2': np.mean(train_r2),
            'val_mean_r2': np.mean(val_r2),
            'test_mean_r2': np.mean(test_r2),
            'train_mean_rmse': np.mean(train_rmse),
            'val_mean_rmse': np.mean(val_rmse),
            'test_mean_rmse': np.mean(test_rmse),
            'train_mean_rpd': np.mean(train_rpd),
            'val_mean_rpd': np.mean(val_rpd),
            'test_mean_rpd': np.mean(test_rpd),
        },
        'history': history,
        'band_importance': band_importance,
        'selected_bands': selected_bands,
        'best_epoch': best_epoch
    }
    
    # 输出三个数据集的对比表格
    logger.info("\n========= 三个数据集性能对比 =========")
    logger.info("数据集        平均R??      平均RMSE    平均RPD")
    logger.info(f"训练集      {np.mean(train_r2):.4f}     {np.mean(train_rmse):.4f}     {np.mean(train_rpd):.4f}")
    logger.info(f"验证集      {np.mean(val_r2):.4f}     {np.mean(val_rmse):.4f}     {np.mean(val_rpd):.4f}")
    logger.info(f"测试集      {np.mean(test_r2):.4f}     {np.mean(test_rmse):.4f}     {np.mean(test_rpd):.4f}")
    logger.info("========================================")
    
    # 添加详细评估指标汇总 - 修改表头包含MAPE列
    logger.info("\n========= 详细任务评估指标 =========")
    logger.info(f"{'参数':<10}{'数据集':<12}{'R??':<10}{'RMSE':<10}{'MAE':<10}{'MAPE(%)':<10}{'解释方差':<12}")
    logger.info(f"{'-'*70}")  # 加长分隔线以适应新增的MAPE列
    
    # 获取所有数据集的预测值和目标值，确保它们匹配
    train_preds, train_targets_np = get_predictions_and_targets(model, train_loader)
    val_preds, val_targets_np = get_predictions_and_targets(model, val_loader)
    test_preds, test_targets_np = get_predictions_and_targets(model, test_loader)
    
    # 添加一个小的epsilon值，避免在计算MAPE时除以0
    epsilon = 1e-10
    
    for i, target in enumerate(task_names):
        # 训练集 - 使用配对的预测值和目标值
        train_r2_i = train_r2[i]
        train_rmse_i = train_rmse[i]
        train_mae = np.mean(np.abs(train_targets_np[:,i] - train_preds[:,i]))
        # 计算训练集MAPE
        # train_mape = np.mean(np.abs((train_targets_np[:,i] - train_preds[:,i]) / (np.abs(train_targets_np[:,i]) + epsilon))) * 100
        absolute_percentage_errors = np.abs((train_targets_np[:,i] - train_preds[:,i]) / (train_targets_np[:,i] + epsilon))
        train_mape = np.mean(absolute_percentage_errors) * 100
        train_ev = 1 - np.var(train_targets_np[:,i] - train_preds[:,i]) / np.var(train_targets_np[:,i])
        
        # 验证集
        val_r2_i = val_r2[i]
        val_rmse_i = val_rmse[i]
        val_mae = np.mean(np.abs(val_targets_np[:,i] - val_preds[:,i]))
        # 计算验证集MAPE
        # val_mape = np.mean(np.abs((val_targets_np[:,i] - val_preds[:,i]) / (np.abs(val_targets_np[:,i]) + epsilon))) * 100
        absolute_percentage_errors = np.abs((val_targets_np[:,i] - val_preds[:,i]) / (val_targets_np[:,i] + epsilon))
        val_mape = np.mean(absolute_percentage_errors) * 100
        val_ev = 1 - np.var(val_targets_np[:,i] - val_preds[:,i]) / np.var(val_targets_np[:,i])
        
        # 测试集
        test_r2_i = test_r2[i]
        test_rmse_i = test_rmse[i]
        test_mae = np.mean(np.abs(test_targets_np[:,i] - test_preds[:,i]))
        # 计算测试集MAPE
        # test_mape = np.mean(np.abs((test_targets_np[:,i] - test_preds[:,i]) / (np.abs(test_targets_np[:,i]) + epsilon))) * 100
        absolute_percentage_errors = np.abs((test_targets_np[:,i] - test_preds[:,i]) / (test_targets_np[:,i] + epsilon))
        test_mape = np.mean(absolute_percentage_errors) * 100
        test_ev = 1 - np.var(test_targets_np[:,i] - test_preds[:,i]) / np.var(test_targets_np[:,i])
        
        # 记录到日志 - 增加MAPE列
        logger.info(f"{target:<10}{'训练集':<12}{train_r2_i:<10.4f}{train_rmse_i:<10.4f}{train_mae:<10.4f}{train_mape:<10.2f}{train_ev:<12.4f}")
        logger.info(f"{'':<10}{'验证集':<12}{val_r2_i:<10.4f}{val_rmse_i:<10.4f}{val_mae:<10.4f}{val_mape:<10.2f}{val_ev:<12.4f}")
        logger.info(f"{'':<10}{'测试集':<12}{test_r2_i:<10.4f}{test_rmse_i:<10.4f}{test_mae:<10.4f}{test_mape:<10.2f}{test_ev:<12.4f}")
        logger.info(f"{'-'*70}")
        
        # 将这些额外指标添加到结果中
        results['metrics'][f'{target}_train_mae'] = train_mae
        results['metrics'][f'{target}_train_ev'] = train_ev
        results['metrics'][f'{target}_train_mape'] = train_mape  # 添加MAPE到结果中
        results['metrics'][f'{target}_val_mae'] = val_mae
        results['metrics'][f'{target}_val_ev'] = val_ev
        results['metrics'][f'{target}_val_mape'] = val_mape  # 添加MAPE到结果中
        results['metrics'][f'{target}_test_mae'] = test_mae
        results['metrics'][f'{target}_test_ev'] = test_ev
        results['metrics'][f'{target}_test_mape'] = test_mape  # 添加MAPE到结果中
    
    # 计算平均MAPE并添加到结果中
    results['metrics']['train_mean_mape'] = np.mean([results['metrics'][f'{target}_train_mape'] for target in task_names])
    results['metrics']['val_mean_mape'] = np.mean([results['metrics'][f'{target}_val_mape'] for target in task_names])
    results['metrics']['test_mean_mape'] = np.mean([results['metrics'][f'{target}_test_mape'] for target in task_names])
    
    return results

def main():
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        
        # 番茄
        'csv_path': 'data/processed/subsets/dataset_max15_per_class.csv',
        'target_cols': ['RUE', 'Pn', 'Gs', 'Ci', 'Tr', 'Ci-Ca', 'WUE', 'iWUE', 'Pmax', 'Rd', 'Ic', 'SPAD', 'LAW'],
        'results_dir': os.path.join('results', 'MT_bandwise_cfl_selection'),  # 修改目录名
        
        # 水稻
        # 'csv_path': 'data/processed/Rice subsets/rice dataset_all_per_class.csv',
        # 'target_cols': ['SPAD','Pn', 'LNC', 'Chl-a', 'Chl-b','LAW', 'Cx', 'Chl'],  # Rice 8个水稻参数
        # 'results_dir': os.path.join('results', 'Rice_bandwise_cfl_selection'),  # 修改目录名
        'feature_cols': list(range(3, 276)),  # 273个波段
        'batch_size': 32,               
        'num_epochs': 500,             
        'learning_rate': 0.001,        
        'weight_decay': 5e-5,         
        'split_method': 'sklearn', #sklearn, random
        'val_size': 0.15,
        'test_size': 0.15,
        'seed': 42,
        'hidden_dim': 128,              
        'num_layers': 1,               
        'num_heads': 1,
        'k_bands': 20,                
        'drop_rate': 0.15,              
        'threshold': None,
        # 早停参数
        'early_stopping': True,
        'patience': 20,               
        'min_delta': 0.0005,
        'metric_monitor': 'val_r2',
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
    
    # 划分数据集
    logger.info(f"使用 {config['split_method']} 方法将数据集划分为训练集、验证集和测试集")
    # 使用sklearn方法进行数据集划分
    train_indices, temp_indices = train_test_split(
        np.arange(len(data)), 
        test_size=config['val_size'] + config['test_size'], 
        random_state=config['seed']
    )
    
    # 确定验证集和测试集的比例
    val_ratio = config['val_size'] / (config['val_size'] + config['test_size'])
    
    # 然后将临时集分成验证集和测试集
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=1-val_ratio,
        random_state=config['seed']
    )
    
    logger.info(f"数据集划分: 训练集{len(train_indices)}样本, 验证集{len(val_indices)}样本, 测试集{len(test_indices)}样本")
    
    # 创建数据集
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
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    # 验证和测试集明确设置shuffle=False，确保预测值和目标值顺序一致
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 构建带波段选择的 ViT 模型
    input_size = len(config['feature_cols'])
    num_tasks = len(config['target_cols'])
    
    model = BandwiseCFLViT(
        input_size=input_size,
        num_tasks=num_tasks,
        k_bands=config['k_bands'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        drop_rate=config['drop_rate']
    ).to(config['device'])
    
    # 使用CFL训练函数
    training_func = train_cfl_model
    logger.info("使用CFL双分支模型架构进行训练")
    
    # 输出模型结构
    num_params = count_parameters(model)
    logger.info(f"模型结构:\n{model}")
    logger.info(f"模型参数量: {num_params}")
    
    # 训练模型
    results = training_func(
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
        metric_monitor=config['metric_monitor']
    )
    
    # 输出每个任务的性能指标
    logger.info("\n=================== 各参数测试集性能 ===================")
    logger.info("参数         R??          RMSE        RPD")
    logger.info("-------------------------------------------------")
    metrics = results['metrics']
    for i, task in enumerate(config['target_cols']):
        logger.info(f"{task:<12} {metrics['test_r2'][i]:.4f}     {metrics['test_rmse'][i]:.4f}     {metrics['test_rpd'][i]:.4f}")
    logger.info("-------------------------------------------------")
    logger.info(f"平均        {metrics['test_mean_r2']:.4f}     {metrics['test_mean_rmse']:.4f}     {metrics['test_mean_rpd']:.4f}")
    logger.info("=================================================")
    
    # 输出波段选择结果
    logger.info("\n=================== 波段选择结果 ===================")
    
    # 获取重要的波段索引（实际列索引）
    selected_bands = results['selected_bands']
    actual_indices = [config['feature_cols'][idx] for idx in selected_bands]
    
    logger.info(f"选择了{len(selected_bands)}个波段")
    logger.info(f"选择的波段索引: {actual_indices}")
    logger.info("=================================================")
    
    logger.info(f"训练完成！最佳验证R??: {metrics['val_mean_r2']:.4f}, 测试R??: {metrics['test_mean_r2']:.4f}")
    print(f"所有结果已保存到: {config['results_dir']}")

if __name__ == "__main__":
    main()