import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error

import os
import time
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from mt_hyperspectral.models.ensemble import calculate_rpd

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


def train_model_with_amp(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, device='cpu', patience=20):
    """混合精度训练函数"""
    try:
        from torch.cuda.amp import autocast, GradScaler
        use_amp = True
        scaler = GradScaler()
    except ImportError:
        use_amp = False
        print("Mixed precision training not available, using regular training")
    
    model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                # 使用混合精度训练
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # 缩放梯度并优化
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 常规训练
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if use_amp:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # 更新学习率
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            
            # 如果学习率发生变化，记录到日志
            if new_lr != old_lr:
                print(f"Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")
        
        # 打印进度
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        
        # 检查是否需要保存模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

# def train_advanced_ensemble(models, train_loader, val_loader, test_loader, target_names, device, 
#                             num_epochs=100, patience=10, verbose=False):
#     """
#     训练集成模型，使用验证集进行早停
    
#     参数:
#         models: 模型列表
#         train_loader: 训练数据加载器
#         val_loader: 验证数据加载器
#         test_loader: 测试数据加载器
#         target_names: 目标变量名称列表
#         device: 训练设备
#         num_epochs: 最大训练轮数
#         patience: 早停耐心值
#         verbose: 是否打印详细日志
        
#     返回:
#         results: 每个模型的评估结果列表
#     """
#     import numpy as np
#     import torch.nn as nn
#     import torch.optim as optim
#     from tqdm import tqdm
    
#     results = []
    
#     for i, model in enumerate(models):
#         print(f"\n训练模型 {i+1}/{len(models)}: {target_names[i]}")
        
#         # 定义损失函数和优化器
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
        
#         # 早停设置
#         best_val_loss = float('inf')
#         no_improve_epochs = 0
#         best_model_state = None
        
#         # 存储训练过程中的指标
#         train_losses = []
#         val_losses = []
        
#         # 训练循环
#         for epoch in range(num_epochs):
#             # 训练阶段
#             model.train()
#             epoch_loss = 0
#             for batch_x, batch_y in train_loader:
#                 batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
#                 # 获取对应目标变量的列
#                 targets = batch_y[:, i].unsqueeze(1)
                
#                 optimizer.zero_grad()
#                 outputs = model(batch_x)
#                 loss = criterion(outputs, targets)
#                 loss.backward()
#                 optimizer.step()
                
#                 epoch_loss += loss.item()
                
#             avg_train_loss = epoch_loss / len(train_loader)
#             train_losses.append(avg_train_loss)
            
#             # 验证阶段
#             model.eval()
#             val_loss = 0
#             with torch.no_grad():
#                 for batch_x, batch_y in val_loader:
#                     batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#                     targets = batch_y[:, i].unsqueeze(1)
#                     outputs = model(batch_x)
#                     loss = criterion(outputs, targets)
#                     val_loss += loss.item()
            
#             avg_val_loss = val_loss / len(val_loader)
#             val_losses.append(avg_val_loss)
            
#             # 输出进度
#             if (epoch + 1) % 10 == 0:
#                 print(f"Epoch {epoch+1}/{num_epochs}, 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")
            
#             # 早停检查
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 no_improve_epochs = 0
#                 best_model_state = model.state_dict().copy()
#                 if verbose:
#                     print(f"Target {target_names[i]} - Epoch {epoch}: Val loss improved to {best_val_loss:.6f}")
#             else:
#                 no_improve_epochs += 1
#                 if no_improve_epochs >= patience:
#                     if verbose:
#                         print(f"Target {target_names[i]} - Early stopping at epoch {epoch}")
#                     break
        
#         # 恢复最佳模型状态(由验证集确定)
#         if best_model_state is not None:
#             model.load_state_dict(best_model_state)
        
#         # 评估模型
#         model.eval()
#         train_preds, train_targets = [], []
#         val_preds, val_targets = [], []
#         test_preds, test_targets = [], []
        
#         with torch.no_grad():
#             # 收集训练集预测
#             for batch_x, batch_y in train_loader:
#                 batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#                 targets = batch_y[:, i].unsqueeze(1)
#                 outputs = model(batch_x)
                
#                 train_preds.append(outputs.cpu().numpy())
#                 train_targets.append(targets.cpu().numpy())
            
#             # 收集验证集预测
#             for batch_x, batch_y in val_loader:
#                 batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#                 targets = batch_y[:, i].unsqueeze(1)
#                 outputs = model(batch_x)
                
#                 val_preds.append(outputs.cpu().numpy())
#                 val_targets.append(targets.cpu().numpy())
            
#             # 收集测试集预测
#             for batch_x, batch_y in test_loader:
#                 batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#                 targets = batch_y[:, i].unsqueeze(1)
#                 outputs = model(batch_x)
                
#                 test_preds.append(outputs.cpu().numpy())
#                 test_targets.append(targets.cpu().numpy())
        
#         # 连接预测结果
#         train_preds = np.vstack(train_preds)
#         train_targets = np.vstack(train_targets)
#         val_preds = np.vstack(val_preds)
#         val_targets = np.vstack(val_targets)
#         test_preds = np.vstack(test_preds)
#         test_targets = np.vstack(test_targets)
        
#         # 使用calculate_metrics计算评估指标，保持一致性
#         train_r2, train_rmse, train_rpd = calculate_metrics(train_targets, train_preds)
#         val_r2, val_rmse, val_rpd = calculate_metrics(val_targets, val_preds)
#         test_r2, test_rmse, test_rpd = calculate_metrics(test_targets, test_preds)
        
#         print(f"{target_names[i]} 评估结果:")
#         print(f"训练集: R² = {train_r2:.4f}, RMSE = {train_rmse:.4f}")
#         print(f"验证集: R² = {val_r2:.4f}, RMSE = {val_rmse:.4f}")
#         print(f"测试集: R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}")
        
#         # 修改结果字典以包含预测值和真实值
#         results.append({
#             'target': target_names[i],
#             'train_r2': train_r2,
#             'train_rmse': train_rmse,
#             'val_r2': val_r2,
#             'val_rmse': val_rmse,
#             'test_r2': test_r2,
#             'test_rmse': test_rmse,
#             'train_losses': train_losses,
#             'val_losses': val_losses,
#             'train_preds': train_preds,     # 添加这行
#             'train_targets': train_targets, # 添加这行
#             'val_preds': val_preds,         # 添加这行
#             'val_targets': val_targets,     # 添加这行
#             'test_preds': test_preds,       # 添加这行
#             'test_targets': test_targets    # 添加这行
#         })
    
#     return results



def train_multitask_model(
    model, 
    train_loader, 
    val_loader, 
    test_loader,
    task_names,
    device,
    results_dir,
    logger,
    num_epochs=100,
    learning_rate=0.001,
    weight_decay=1e-5,
    patience=15,
    min_delta=0.0005,
    early_stopping=True,
    metric_monitor='val_r2',
    compute_final_metrics=False,
    consistent_evaluation=True
):
    """
    多任务学习模型训练函数
    
    详细说明权重调整机制和损失函数设计:
    
    1. 动态权重调整机制:
       - 注意力权重: MultiheadAttention中的Q、K、V权重，通过反向传播自动学习
       - 任务权重: 可学习参数task_weights，用于平衡不同任务的损失贡献
       - 特征权重: 注意力分数α_ij，表示特征i对特征j的重要性
       - 层次权重: 残差连接中的自适应权重分配
    
    2. 损失函数设计:
       基础损失: L_base = Σᵢ MSE(yᵢ, ŷᵢ)
       加权损失: L_weighted = Σᵢ wᵢ × MSE(yᵢ, ŷᵢ)
       正则化: L_reg = λ × Σᵢ (log(wᵢ) + 1/wᵢ)
       总损失: L_total = L_weighted + L_reg
    
    3. 优化器设计:
       - 主模型参数: AdamW优化器，带权重衰减
       - 任务权重: 独立的Adam优化器
       - 学习率调度: 基于验证损失的自适应调整
    """
    
    num_tasks = len(task_names)
    
    # 1. 主模型参数优化器 - AdamW具有解耦权重衰减
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        # betas=(0.9, 0.999),  # Adam的动量参数
        # eps=1e-8            # 数值稳定性参数
    )
    logger.info(f"主优化器配置: AdamW(lr={learning_rate}, wd={weight_decay})")
    
    # 2. 学习率调度器 - 基于验证损失的自适应调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=patience//2
    )
    
    # # 2. 学习率调度器 - 基于验证损失的自适应调整
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode='min',           # 监控损失下降
    #     factor=0.5,           # 学习率衰减因子
    #     patience=patience//2, # 等待轮数
    #     threshold=1e-4,       # 改善阈值
    #     min_lr=1e-7          # 最小学习率
    # )
    logger.info(f"学习率调度器: ReduceLROnPlateau(factor=0.5, patience={patience//2})")
    
    
    # 3. 混合精度训练的梯度缩放器
    scaler = GradScaler()
    logger.info("启用混合精度训练(AMP)")
    
    # 4. 任务权重初始化和优化器
    # 这是关键的动态权重调整机制
    task_weights = nn.ParameterList([
        nn.Parameter(torch.ones(1, device=device)) for _ in range(num_tasks)
    ])
    
    # task_weights = nn.ParameterList([
    #     nn.Parameter(torch.ones(1, device=device, requires_grad=True)) 
    #     for _ in range(num_tasks)
    # ])
    
    # 任务权重专用优化器
    task_weights_optimizer = optim.Adam(
        task_weights.parameters(), 
        lr=learning_rate,
        # betas=(0.9, 0.999)
    )
    logger.info(f"任务权重优化器: Adam(lr={learning_rate}) - 用于{num_tasks}个任务权重")
    
    # 详细解释权重调整机制
    logger.info("\n权重调整机制详解:")
    logger.info("1. 注意力权重: 在MTIEncoderBlock中通过self-attention自动学习")
    logger.info("   - Q, K, V矩阵: 将输入特征映射到查询、键、值空间")
    logger.info("   - 注意力分数: α_ij = softmax(QK^T/√d), 表示特征间重要性")
    logger.info("   - 动态更新: 每个前向传播都重新计算注意力权重")
    logger.info("2. 任务权重: 可学习参数wᵢ, 平衡不同任务的损失贡献")
    logger.info("   - 初始值: 全为1.0, 表示所有任务等权重")
    logger.info("   - 动态调整: 根据各任务的学习难度自适应调整")
    logger.info("   - 正则化: 防止权重过大或过小")
    
    # 定义评估数据集的辅助函数
    def evaluate_dataset(loader):
        """评估模型在给定数据加载器上的性能"""
        return evaluate_model(model, loader, device, num_tasks)
    
    # 训练循环初始化
    best_val_r2 = -float('inf')
    best_model_path = os.path.join(results_dir, 'best_model.pt')
    early_stop_counter = 0
    best_epoch = 0
    
    # 记录训练历史
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_r2': [],
        'val_r2': [],
        'train_rmse': [],
        'val_rmse': [],
        'train_rpd': [],
        'val_rpd': [],
        'lr': [],
        'task_weights': [[] for _ in range(num_tasks)],  # 记录任务权重变化
        'task_train_r2': [[] for _ in range(num_tasks)],
        'task_val_r2': [[] for _ in range(num_tasks)],
        'task_train_rmse': [[] for _ in range(num_tasks)],
        'task_val_rmse': [[] for _ in range(num_tasks)],
        'task_train_rpd': [[] for _ in range(num_tasks)],
        'task_val_rpd': [[] for _ in range(num_tasks)]
    }
    
    # 记录训练开始时间
    train_start_time = time.time()
    logger.info(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_start_time))}")
    logger.info("开始训练...")
    
    # 主训练循环
    for epoch in range(num_epochs):
        # =================== 训练阶段 ===================
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        train_preds_list = []
        train_targets_list = []
        epoch_task_losses = [0.0] * num_tasks  # 记录每个任务的损失
        
        for features, targets in train_loader:
            # 清零梯度
            optimizer.zero_grad()
            task_weights_optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                outputs = model(features)
                
                # 计算每个任务的基础损失
                task_losses = []
                for i in range(num_tasks):
                    task_loss = F.mse_loss(outputs[:, i], targets[:, i])
                    task_losses.append(task_loss)
                    epoch_task_losses[i] += task_loss.item()
                
                # 动态加权损失计算
                # L_weighted = Σᵢ wᵢ × MSE(yᵢ, ŷᵢ)
                weighted_loss = sum(task_weights[i] * task_losses[i] for i in range(num_tasks))
                
                # 任务权重正则化项
                # L_reg = λ × Σᵢ (log(wᵢ) + 1/wᵢ)
                # 这个正则化项防止权重趋向极端值
                weight_regularization = 0.01 * sum(
                    torch.log(w.clamp(min=1e-8)) + 1.0/(w.clamp(min=1e-8)) 
                    for w in task_weights
                )
                
                # 总损失
                total_loss = weighted_loss + weight_regularization
            
            # 混合精度反向传播
            scaler.scale(total_loss).backward()
            
            # # 梯度裁剪防止梯度爆炸
            # scaler.unscale_(optimizer)
            # scaler.unscale_(task_weights_optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(task_weights.parameters(), max_norm=1.0)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.step(task_weights_optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            batch_count += 1
            
            # 收集预测和目标用于计算指标
            train_preds_list.append(outputs.detach().cpu().numpy())
            train_targets_list.append(targets.detach().cpu().numpy())
        
        # 计算训练指标
        avg_train_loss = train_loss / batch_count
        train_preds = np.vstack(train_preds_list)
        train_targets = np.vstack(train_targets_list)
        train_r2, train_rmse, train_rpd = calculate_metrics(train_targets, train_preds)
        
        # =================== 验证阶段 ===================
        model.eval()
        val_loss = 0.0
        batch_count = 0
        
        val_preds_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                with autocast():
                    outputs = model(features)
                    
                    # 验证时使用简单的平均损失，不使用任务权重
                    losses = [F.mse_loss(outputs[:, i], targets[:, i]) for i in range(num_tasks)]
                    loss = sum(losses) / num_tasks
                
                val_loss += loss.item()
                batch_count += 1
                
                val_preds_list.append(outputs.cpu().numpy())
                val_targets_list.append(targets.cpu().numpy())
        
        # 计算验证指标
        avg_val_loss = val_loss / batch_count
        val_preds = np.vstack(val_preds_list)
        val_targets = np.vstack(val_targets_list)
        val_r2, val_rmse, val_rpd = calculate_metrics(val_targets, val_preds)
        
        # =================== 模型保存和早停 ===================
        mean_val_r2 = np.mean(val_r2)
        improved = False
        
        # 根据监控指标决定是否保存模型
        combined_r2 = np.mean(val_r2) * 0.6 + np.mean(train_r2) * 0.4
        current_metric = combined_r2 if metric_monitor == 'combined_r2' else mean_val_r2
        
        if current_metric > best_val_r2 + min_delta:
            improved = True
            early_stop_counter = 0
            best_val_r2 = current_metric
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            
            # 保存训练和验证指标
            best_train_r2 = train_r2.copy()
            best_train_rmse = train_rmse.copy() 
            best_train_rpd = train_rpd.copy()
            best_val_r2_values = val_r2.copy()
            best_val_rmse = val_rmse.copy()
            best_val_rpd = val_rpd.copy()
            
            # 保存最佳模型及任务权重
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'task_weights': [w.detach().cpu() for w in task_weights],
                'optimizer_state_dict': optimizer.state_dict(),
                'task_weights_optimizer_state_dict': task_weights_optimizer.state_dict(),
                'train_metrics': {
                    'r2': best_train_r2,
                    'rmse': best_train_rmse,
                    'rpd': best_train_rpd,
                    'mean_r2': np.mean(best_train_r2)
                },
                'val_metrics': {
                    'r2': best_val_r2_values,
                    'rmse': best_val_rmse,
                    'rpd': best_val_rpd,
                    'mean_r2': np.mean(best_val_r2_values)
                }
            }, best_model_path)
            logger.info(f"保存最佳模型! 验证集R²: {np.mean(val_r2):.4f}, 监控指标: {current_metric:.4f}")
        else:
            early_stop_counter += 1
            logger.info(f"验证集R²未提升，早停计数: {early_stop_counter}/{patience}")
            
        # 早停检查
        if early_stopping and early_stop_counter >= patience:
            logger.info(f"早停触发! {patience} 个epoch内验证集R²没有提升。")
            logger.info(f"最佳模型是第 {best_epoch} 个epoch，验证集R²: {mean_val_r2:.4f}")
            break
        
        # 学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if current_lr != old_lr:
            logger.info(f"学习率更新: {old_lr:.6f} → {current_lr:.6f}")
        
        # 记录历史数据
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_r2'].append(np.mean(train_r2))
        history['val_r2'].append(np.mean(val_r2))
        history['train_rmse'].append(np.mean(train_rmse))
        history['val_rmse'].append(np.mean(val_rmse))
        history['train_rpd'].append(np.mean(train_rpd))
        history['val_rpd'].append(np.mean(val_rpd))
        history['lr'].append(current_lr)
        
        # 记录任务权重变化
        for i in range(num_tasks):
            history['task_weights'][i].append(task_weights[i].detach().cpu().item())
            history['task_train_r2'][i].append(train_r2[i])
            history['task_val_r2'][i].append(val_r2[i])
            history['task_train_rmse'][i].append(train_rmse[i])
            history['task_val_rmse'][i].append(val_rmse[i])
            history['task_train_rpd'][i].append(train_rpd[i])
            history['task_val_rpd'][i].append(val_rpd[i])
        
        # # 格式化并输出指标
        # logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        # logger.info(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
        
        # logger.info("训练集指标:")
        # logger.info(format_metrics_table(task_names, train_r2, train_rmse, train_rpd))
        
        # logger.info("验证集指标:")
        # logger.info(format_metrics_table(task_names, val_r2, val_rmse, val_rpd))
        # 定期输出训练状态
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info(f"损失: 训练={avg_train_loss:.4f}, 验证={avg_val_loss:.4f}")
            logger.info(f"学习率: {current_lr:.6f}")
            
            # 输出任务权重变化
            task_weights_str = ", ".join([f"{w.item():.3f}" for w in task_weights])
            logger.info(f"任务权重: [{task_weights_str}]")
            
            logger.info("训练集指标:")
            logger.info(format_metrics_table(task_names, train_r2, train_rmse, train_rpd))
            
            logger.info("验证集指标:")
            logger.info(format_metrics_table(task_names, val_r2, val_rmse, val_rpd))
    
    # =================== 训练结束处理 ===================
    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    hours, remainder = divmod(total_train_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 输出训练完成信息
    if early_stopping and early_stop_counter >= patience:
        logger.info(f"训练由于早停在第 {epoch+1} 个epoch结束")
    else:
        logger.info(f"训练完成所有 {num_epochs} 个epoch")
    
    logger.info(f"总训练时长: {int(hours):02d}小时{int(minutes):02d}分钟{int(seconds):02d}秒")
    logger.info(f"最佳模型在第 {best_epoch} 个epoch，验证集R²: {best_val_r2:.4f}")
    
    # 加载最佳模型进行最终评估
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 输出最终的任务权重
    final_task_weights = checkpoint['task_weights']
    logger.info("\n最终任务权重分布:")
    for i, (task_name, weight) in enumerate(zip(task_names, final_task_weights)):
        logger.info(f"  {task_name}: {weight.item():.4f}")
    
    # 根据一致性选项决定如何处理训练集指标
    if consistent_evaluation and 'train_metrics' in checkpoint:
        logger.info("使用保存的训练集指标，以确保结果一致性")
        train_r2 = checkpoint['train_metrics']['r2']
        train_rmse = checkpoint['train_metrics']['rmse']
        train_rpd = checkpoint['train_metrics']['rpd']
    else:
        if not consistent_evaluation:
            logger.info("重新评估训练集性能...")
        train_r2, train_rmse, train_rpd = evaluate_model(model, train_loader, device, num_tasks)
    
    # 重新评估验证集和测试集
    val_r2, val_rmse, val_rpd = evaluate_model(model, val_loader, device, num_tasks)
    test_r2, test_rmse, test_rpd = evaluate_model(model, test_loader, device, num_tasks)
    
    # 构建结果对象
    results = {
        'model': model,
        'task_weights': final_task_weights,
        'best_model_path': best_model_path,
        'best_epoch': best_epoch,
        'history': history,
        'metrics': {
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'train_rpd': train_rpd,
            'val_r2': val_r2,
            'val_rmse': val_rmse,
            'val_rpd': val_rpd,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_rpd': test_rpd,
            'train_mean_r2': np.mean(train_r2),
            'train_mean_rmse': np.mean(train_rmse),
            'train_mean_rpd': np.mean(train_rpd),
            'val_mean_r2': np.mean(val_r2),
            'val_mean_rmse': np.mean(val_rmse),
            'val_mean_rpd': np.mean(val_rpd),
            'test_mean_r2': np.mean(test_r2),
            'test_mean_rmse': np.mean(test_rmse),
            'test_mean_rpd': np.mean(test_rpd)
        }
    }
    
    # 输出详细的损失函数和优化器总结
    logger.info("\n=========== 训练机制总结 ===========")
    logger.info("损失函数设计:")
    logger.info("  基础损失: MSE(y, ŷ) = 1/n Σᵢ (yᵢ - ŷᵢ)²")
    logger.info("  任务加权: L_weighted = Σⱼ wⱼ × MSEⱼ")
    logger.info("  正则化项: L_reg = 0.01 × Σⱼ (log(wⱼ) + 1/wⱼ)")
    logger.info("  总损失: L_total = L_weighted + L_reg")
    logger.info("\n权重调整机制:")
    logger.info("  1. 注意力权重: 自注意力机制学习特征间依赖关系")
    logger.info("  2. 任务权重: 可学习参数，平衡多任务损失贡献")
    logger.info("  3. 模型权重: AdamW优化器更新网络参数")
    logger.info("  4. 学习率: 自适应调度，基于验证损失")
    
    # 输出三个数据集的对比表格
    logger.info("\n========= 三个数据集性能对比 =========")
    logger.info("数据集        平均R²      平均RMSE    平均RPD")
    logger.info(f"训练集      {np.mean(train_r2):.4f}     {np.mean(train_rmse):.4f}     {np.mean(train_rpd):.4f}")
    logger.info(f"验证集      {np.mean(val_r2):.4f}     {np.mean(val_rmse):.4f}     {np.mean(val_rpd):.4f}")
    logger.info(f"测试集      {np.mean(test_r2):.4f}     {np.mean(test_rmse):.4f}     {np.mean(test_rpd):.4f}")
    logger.info("========================================")
    
    # 添加详细评估指标汇总 - 修复MAE和解释方差计算问题
    logger.info("\n========= 详细任务评估指标 =========")
    logger.info(f"{'参数':<10}{'数据集':<12}{'R²':<10}{'RMSE':<10}{'MAE':<10}{'RPD':<10}{'解释方差':<12}")
    logger.info(f"{'-'*70}")
    
    # 获取所有数据集的预测值和目标值，确保它们匹配
    train_preds, train_targets_np = get_predictions_and_targets(model, train_loader)
    val_preds, val_targets_np = get_predictions_and_targets(model, val_loader)
    test_preds, test_targets_np = get_predictions_and_targets(model, test_loader)
    
    for i, target in enumerate(task_names):
        # 训练集 - 使用配对的预测值和目标值
        train_r2_i = train_r2[i]
        train_rmse_i = train_rmse[i]
        train_rpd_i = train_rpd[i]
        train_mae = np.mean(np.abs(train_targets_np[:,i] - train_preds[:,i]))
        train_ev = 1 - np.var(train_targets_np[:,i] - train_preds[:,i]) / np.var(train_targets_np[:,i])
        
        # 验证集
        val_r2_i = val_r2[i]
        val_rmse_i = val_rmse[i]
        val_rpd_i = val_rpd[i]
        val_mae = np.mean(np.abs(val_targets_np[:,i] - val_preds[:,i]))
        val_ev = 1 - np.var(val_targets_np[:,i] - val_preds[:,i]) / np.var(val_targets_np[:,i])
        
        # 测试集
        test_r2_i = test_r2[i]
        test_rmse_i = test_rmse[i]
        test_rpd_i = test_rpd[i]
        test_mae = np.mean(np.abs(test_targets_np[:,i] - test_preds[:,i]))
        test_ev = 1 - np.var(test_targets_np[:,i] - test_preds[:,i]) / np.var(test_targets_np[:,i])
        
        # 记录到日志
        logger.info(f"{target:<10}{'训练集':<12}{train_r2_i:<10.4f}{train_rmse_i:<10.4f}{train_mae:<10.4f}{train_rpd_i:<10.4f}{train_ev:<12.4f}")
        logger.info(f"{'':<10}{'验证集':<12}{val_r2_i:<10.4f}{val_rmse_i:<10.4f}{val_mae:<10.4f}{val_rpd_i:<10.4f}{val_ev:<12.4f}")
        logger.info(f"{'':<10}{'测试集':<12}{test_r2_i:<10.4f}{test_rmse_i:<10.4f}{test_mae:<10.4f}{test_rpd_i:<10.4f}{test_ev:<12.4f}")
        logger.info(f"{'-'*70}")
        
        # 将这些额外指标添加到结果中
        results['metrics'][f'{target}_train_mae'] = train_mae
        results['metrics'][f'{target}_train_ev'] = train_ev
        results['metrics'][f'{target}_train_rpd'] = train_rpd_i
        results['metrics'][f'{target}_val_mae'] = val_mae
        results['metrics'][f'{target}_val_ev'] = val_ev
        results['metrics'][f'{target}_val_rpd'] = val_rpd_i
        results['metrics'][f'{target}_test_mae'] = test_mae
        results['metrics'][f'{target}_test_ev'] = test_ev
        results['metrics'][f'{target}_test_rpd'] = test_rpd_i

    # 保存训练历史和评估结果到CSV文件
    save_history_and_results(history, task_names, results, results_dir)
    
    return results

# 替换原来的两个独立函数
def get_predictions_and_targets(model, data_loader):
    """从数据加载器中同时获取模型预测值和真实目标值，确保顺序一致"""
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for features, batch_targets in data_loader:
            outputs = model(features)
            preds.append(outputs.cpu().numpy())
            targets.append(batch_targets.cpu().numpy())
    return np.vstack(preds), np.vstack(targets)

# 保留原来的独立函数，但添加警告注释
def get_predictions(model, data_loader):
    """从数据加载器中获取模型预测值
    警告：如果data_loader有shuffle=True，返回顺序可能与get_targets不一致"""
    model.eval()
    preds = []
    with torch.no_grad():
        for features, _ in data_loader:
            outputs = model(features)
            preds.append(outputs.cpu().numpy())
    return np.vstack(preds)

def get_targets(data_loader):
    """从数据加载器中提取目标值
    警告：如果data_loader有shuffle=True，返回顺序可能与get_predictions不一致"""
    targets = []
    for _, batch_targets in data_loader:
        targets.append(batch_targets.cpu().numpy())
    return np.vstack(targets)

def save_history_and_results(history, task_names, results, results_dir):
    """保存训练历史和评估结果到CSV文件"""
    # 总体训练历史
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
    
    # 每个任务的训练历史
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
    
    # 保存最终评估结果
    metrics = results['metrics']
    final_results = {
        'dataset': ['训练集', '验证集', '测试集'],
        'mean_r2': [metrics['train_mean_r2'], metrics['val_mean_r2'], metrics['test_mean_r2']],
        'mean_rmse': [metrics['train_mean_rmse'], metrics['val_mean_rmse'], metrics['test_mean_rmse']],
        'mean_rpd': [metrics['train_mean_rpd'], metrics['val_mean_rpd'], metrics['test_mean_rpd']]
    }
    
    # 添加每个任务的单独指标
    for i, task in enumerate(task_names):
        final_results[f'{task}_r2'] = [metrics['train_r2'][i], metrics['val_r2'][i], metrics['test_r2'][i]]
        final_results[f'{task}_rmse'] = [metrics['train_rmse'][i], metrics['val_rmse'][i], metrics['test_rmse'][i]]
        final_results[f'{task}_rpd'] = [metrics['train_rpd'][i], metrics['val_rpd'][i], metrics['test_rpd'][i]]
    
    results_df = pd.DataFrame(final_results)
    results_path = os.path.join(results_dir, 'final_evaluation_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8')

def safe_format(value, format_spec='.4f'):
    """
    安全地格式化值，确保NumPy数组被转换为标量后再进行格式化
    解决"unsupported format string passed to numpy.ndarray.__format__"错误
    """
    try:
        # 检查是否是PyTorch张量
        if hasattr(value, 'item'):
            scalar_value = value.item()
        # 检查是否是NumPy数组
        elif isinstance(value, np.ndarray):
            # 如果是标量数组（只有一个元素），转换为Python标量
            if value.size == 1:
                scalar_value = float(value)
            else:
                # 如果是多元素数组，可能需要取平均值或特定元素
                scalar_value = float(np.mean(value))
        # 检查是否是列表
        elif isinstance(value, list):
            scalar_value = float(np.mean(value))
        else:
            # 已经是标量
            scalar_value = value
            
        # 进行格式化
        return format(scalar_value, format_spec)
    except Exception:
        # 如果格式化失败，直接返回未格式化的值
        return str(value)

def train_ensemble(models, train_loader, val_loader, test_loader, target_names, device, 
                  num_epochs=100, patience=10, logger=None):
    """
    训练集成模型，使用验证集进行早停 - 增强版本带统计信息
    """
    
    # 如果没有传入logger，创建一个简单的默认logger
    if logger is None:
        import logging
        logger = logging.getLogger()
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    # 生成Ensemble模型统计信息
    logger.info("\n开始Ensemble模型训练前统计:")
    logger.info(f"总模型数量: {len(models)}")
    logger.info(f"目标参数: {target_names}")
    
    # 计算单个模型参数量（假设所有模型架构相同）
    if models:
        single_model_params = sum(p.numel() for p in models[0].parameters())
        total_params = single_model_params * len(models)
        logger.info(f"单模型参数量: {single_model_params:,}")
        logger.info(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    results = []
    
    for i, (model, target) in enumerate(zip(models, target_names)):
        logger.info(f"训练模型 {i+1}/{len(models)}: {target}")
        
        # 使用与原始函数相同的训练逻辑，但避免格式化错误
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        best_val_loss = float('inf')
        best_epoch = 0
        no_improve_count = 0
        best_model_state = None
        
        train_losses = []
        val_losses = []
        
        # 训练循环
        for epoch in range(num_epochs):
            model.train()
            epoch_train_losses = []
            
            for X, y in train_loader:
                X = X.to(device)
                # 提取当前目标变量的值
                y_i = y[:, i:i+1].to(device)
                
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y_i)
                loss.backward()
                optimizer.step()
                
                epoch_train_losses.append(loss.item())
            
            model.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(device)
                    y_i = y[:, i:i+1].to(device)
                    
                    pred = model(X)
                    loss = criterion(pred, y_i)
                    epoch_val_losses.append(loss.item())
            
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_train_loss)
            
            avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
            val_losses.append(avg_val_loss)
            
            # 输出进度
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, 训练损失: {safe_format(avg_train_loss)}, 验证损失: {safe_format(avg_val_loss)}")

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                no_improve_count = 0
                best_model_state = model.state_dict().copy()
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # 加载最佳模型权重
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # 评估最终模型
        model.eval()
        
        # 收集训练集结果
        train_preds = []
        train_targets = []
        with torch.no_grad():
            for X, y in train_loader:
                X = X.to(device)
                y_i = y[:, i:i+1]
                
                pred = model(X)
                train_preds.append(pred.cpu().numpy())
                train_targets.append(y_i.numpy())
        
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        
        # 收集验证集结果
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y_i = y[:, i:i+1]
                
                pred = model(X)
                val_preds.append(pred.cpu().numpy())
                val_targets.append(y_i.numpy())
        
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        
        # 收集测试集结果
        test_preds = []
        test_targets = []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y_i = y[:, i:i+1]
                
                pred = model(X)
                test_preds.append(pred.cpu().numpy())
                test_targets.append(y_i.numpy())
        
        test_preds = np.vstack(test_preds)
        test_targets = np.vstack(test_targets)
        
        # 计算指标
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        train_r2 = r2_score(train_targets, train_preds)
        val_r2 = r2_score(val_targets, val_preds)
        test_r2 = r2_score(test_targets, test_preds)
        
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
        
        # 计算MAE指标
        train_mae = mean_absolute_error(train_targets, train_preds)
        val_mae = mean_absolute_error(val_targets, val_preds)
        test_mae = mean_absolute_error(test_targets, test_preds)
        
        # 计算RPD指标
        from mt_hyperspectral.models.ensemble import calculate_rpd
        train_rpd = calculate_rpd(train_targets.flatten(), train_preds.flatten())
        val_rpd = calculate_rpd(val_targets.flatten(), val_preds.flatten())
        test_rpd = calculate_rpd(test_targets.flatten(), test_preds.flatten())
        
        # 使用安全格式化打印结果
        logger.info(f"训练集: R² = {safe_format(train_r2)}, RMSE = {safe_format(train_rmse)}, MAE = {safe_format(train_mae)}, RPD = {safe_format(train_rpd)}")
        logger.info(f"验证集: R² = {safe_format(val_r2)}, RMSE = {safe_format(val_rmse)}, MAE = {safe_format(val_mae)}, RPD = {safe_format(val_rpd)}")
        logger.info(f"测试集: R² = {safe_format(test_r2)}, RMSE = {safe_format(test_rmse)}, MAE = {safe_format(test_mae)}, RPD = {safe_format(test_rpd)}")

        # 添加结果
        result = {
            'train_preds': train_preds,
            'train_targets': train_targets,
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_rpd': train_rpd,
            
            'val_preds': val_preds,
            'val_targets': val_targets,
            'val_r2': val_r2,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_rpd': val_rpd,
            
            'test_preds': test_preds,
            'test_targets': test_targets,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_rpd': test_rpd
        }
        
        results.append(result)
    
    # 训练完成后的统计摘要
    logger.info("\n=== Ensemble模型训练完成统计 ===")
    avg_test_r2 = np.mean([r['test_r2'] for r in results])
    avg_test_rmse = np.mean([r['test_rmse'] for r in results])
    avg_test_mae = np.mean([r['test_mae'] for r in results])
    avg_test_rpd = np.mean([r['test_rpd'] for r in results])
    
    logger.info(f"平均测试集性能: R²={avg_test_r2:.4f}, RMSE={avg_test_rmse:.4f}, MAE={avg_test_mae:.4f}, RPD={avg_test_rpd:.4f}")
    logger.info(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    return results
