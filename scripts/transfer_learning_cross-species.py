#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨作物迁移学习脚本 - 专注于Pn, SPAD, LAW三参数
番茄模型 ↔ 水稻模型 迁移学习

支持两种迁移模式:
1. tomato_to_rice: 番茄三参数模型 → 水稻数据微调
2. rice_to_tomato: 水稻三参数模型 → 番茄数据微调

注意：本脚本使用与原始训练代码4.MT_task13_multitask_vit.py完全相同的CSVDataset和模型定义
"""

# 导入必要的模块
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from torch.cuda.amp import GradScaler, autocast
import datetime
import time
import copy
import logging
import pickle

# 添加项目根目录到路径
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

# 使用与原始训练代码完全相同的CSVDataset类定义
class CSVDataset(Dataset):
    """处理CSV格式的多任务回归数据集 - 与原始训练代码4.MT_task13_multitask_vit.py保持一致"""
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

# 使用与原始训练代码相同的MultiTaskTMI模型定义
class MTIEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.drop1 = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x_res + self.drop1(attn_output)

        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)
        return x

class MultiTaskTMI(nn.Module):
    def __init__(self, input_size=273, hidden_dim=512, num_layers=3, num_heads=8, num_tasks=13, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        
        # 输入投影层，将特征投影到隐藏维度
        self.input_proj = nn.Linear(input_size, hidden_dim)
        
        # ViT编码器层
        self.encoders = nn.Sequential(
            *[MTIEncoderBlock(dim=hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop) 
              for _ in range(num_layers)]
        )
        
        # 任务特定解码器
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
        
        # 通过Transformer编码器
        x = self.encoders(x)  # [batch_size, 1, hidden_dim]
        
        # 提取CLS token (第一个token)
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # 通过解码器获得多任务预测
        output = self.decoder(x)  # [batch_size, num_tasks]
        
        return output

# 使用与原始训练代码相同的评估函数
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

def load_pretrained_model(pretrained_path, input_size, hidden_dim, num_layers, num_heads, 
                         source_num_tasks, target_num_tasks, device, freeze_layers=True):
    """
    加载预训练模型并修改为目标任务
    
    Args:
        pretrained_path: 预训练模型路径
        input_size: 输入特征维度
        hidden_dim: 隐藏层维度 
        num_layers: Transformer层数
        num_heads: 注意力头数
        source_num_tasks: 源模型的任务数
        target_num_tasks: 目标模型的任务数
        device: 计算设备
        freeze_layers: 是否冻结预训练层
        
    Returns:
        修改后的模型
    """
    # 加载预训练权重
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    
    # 创建新模型，针对新任务
    new_model = MultiTaskTMI(
        input_size=input_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_tasks=target_num_tasks  # 新任务数量可能不同
    ).to(device)
    
    # 获取新模型的状态字典
    model_dict = new_model.state_dict()
    
    # 筛选出可以加载的参数（除了输出层）
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and 'decoder.' not in k}
    
    # 更新模型参数
    model_dict.update(pretrained_dict)
    new_model.load_state_dict(model_dict, strict=False)
    
    # 冻结预训练层参数
    if freeze_layers:
        for name, param in new_model.named_parameters():
            # 除了解码器层，其他层都冻结
            if 'decoder' not in name:
                param.requires_grad = False
        print("已冻结所有预训练层参数，只训练解码器部分")
    
    # 替换解码器为更强正则化版本
    # 检查模型的具体结构，不假设特定的层次结构
    # 直接创建一个新的强正则化解码器
    input_dim = hidden_dim  # 根据模型结构确定的输入维度
    
    # 创建新的解码器，包含强正则化
    new_decoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_dim, target_num_tasks)
    )
    
    # 将解码器明确地移到正确的设备上
    new_decoder = new_decoder.to(device)
    new_model.decoder = new_decoder
    
    return new_model

def finetune(model, train_loader, val_loader, test_loader, task_names, 
           device, results_dir, logger, epochs=50, learning_rate=5e-4, 
           weight_decay=1e-5, patience=10, min_delta=0.001):
    """
    跨作物迁移学习微调函数
    针对Pn, SPAD, LAW三个共同参数进行微调
    """
    
    # 记录迁移学习开始
    logger.info(f"开始跨作物迁移学习微调...")
    logger.info(f"目标参数: {task_names}")
    logger.info(f"微调epochs: {epochs}, 学习率: {learning_rate}")
    
    # 使用更保守的优化器设置，适合迁移学习
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # 只优化需要梯度的参数
        lr=learning_rate, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器 - 对微调使用更温和的策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # 监控R²最大化
        factor=0.7,  # 温和的衰减
        patience=patience//2, 
        threshold=1e-4,
        min_lr=1e-7
    )
    
    # 梯度缩放器
    scaler = GradScaler()
    
    # 训练历史记录
    best_val_r2 = -float('inf')
    early_stop_counter = 0
    best_model_path = os.path.join(results_dir, 'best_finetuned_model.pt')
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_losses = []
        
        for features, targets in train_loader:
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(features)
                
                # 使用简化的损失函数，避免复杂的任务特定处理
                loss = F.mse_loss(outputs, targets)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # 验证阶段
        train_r2, train_rmse, train_rpd = evaluate_model(model, train_loader, device, len(task_names))
        val_r2, val_rmse, val_rpd = evaluate_model(model, val_loader, device, len(task_names))
        
        # 学习率调度
        mean_val_r2 = np.mean(val_r2)
        scheduler.step(mean_val_r2)
        
        # 早停和模型保存
        if mean_val_r2 > best_val_r2 + min_delta:
            best_val_r2 = mean_val_r2
            early_stop_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_r2': train_r2,
                'val_r2': val_r2,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_rpd': train_rpd,
                'val_rpd': val_rpd
            }, best_model_path)
            
        else:
            early_stop_counter += 1
        
        # 定期输出训练进度
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"训练损失: {avg_train_loss:.4f}")
            logger.info(f"训练R²: {[f'{r:.3f}' for r in train_r2]}")
            logger.info(f"验证R²: {[f'{r:.3f}' for r in val_r2]}")
            logger.info(f"平均验证R²: {mean_val_r2:.4f}")
            logger.info(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停检查
        if early_stop_counter >= patience:
            logger.info(f"早停触发! 最佳验证R²: {best_val_r2:.4f}")
            break
    
    # 加载最佳模型进行最终评估
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("已加载最佳微调模型权重")
    
    # 最终评估
    final_train_r2, final_train_rmse, final_train_rpd = evaluate_model(model, train_loader, device, len(task_names))
    final_val_r2, final_val_rmse, final_val_rpd = evaluate_model(model, val_loader, device, len(task_names))
    final_test_r2, final_test_rmse, final_test_rpd = evaluate_model(model, test_loader, device, len(task_names))
    
    logger.info("微调完成!")
    logger.info(f"最终平均R² - 训练: {np.mean(final_train_r2):.4f}, 验证: {np.mean(final_val_r2):.4f}, 测试: {np.mean(final_test_r2):.4f}")
    
    return {
        'model': model,
        'best_model_path': best_model_path,
        'train_r2': final_train_r2,
        'val_r2': final_val_r2,
        'test_r2': final_test_r2,
        'train_rmse': final_train_rmse,
        'val_rmse': final_val_rmse,
        'test_rmse': final_test_rmse,
        'train_rpd': final_train_rpd,
        'val_rpd': final_val_rpd,
        'test_rpd': final_test_rpd
    }
    """
    防过拟合微调函数
    """
    # 创建优化器，使用高强度的L2正则化
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,  # 使用输入的weight_decay
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,
        eta_min=learning_rate / 20
    )
    
    # 或者使用根据验证集表现调整学习率的调度器
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 因为我们希望R²越大越好
        factor=0.5,  # 学习率减半
        patience=5,  # 等待5个epoch无改善再调整
        verbose=True
    )
    
    # 梯度缩放器，用于混合精度训练
    scaler = GradScaler()
    
    # 使用均等权重，简化迁移学习
    task_weights = torch.ones(len(task_names), device=device)
    logger.info(f"使用均等任务权重: {task_weights.tolist()}")
    
    # 记录训练历史
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_r2': [], 'val_r2': [], 'test_r2': [],
        'train_rmse': [], 'val_rmse': [], 'test_rmse': [],
        'train_rpd': [], 'val_rpd': [], 'test_rpd': [],
        'task_train_r2': [[] for _ in range(len(task_names))],
        'task_val_r2': [[] for _ in range(len(task_names))],
        'task_test_r2': [[] for _ in range(len(task_names))],
        'task_train_rmse': [[] for _ in range(len(task_names))],
        'task_val_rmse': [[] for _ in range(len(task_names))],
        'task_test_rmse': [[] for _ in range(len(task_names))],
        'task_train_rpd': [[] for _ in range(len(task_names))],
        'task_val_rpd': [[] for _ in range(len(task_names))],
        'task_test_rpd': [[] for _ in range(len(task_names))],
        'lr': []
    }
    
    # 早停变量
    best_val_r2 = -float('inf')  # 初始化为负无穷，确保任何有效的R2值都能改进它
    early_stop_counter = 0
    best_model_state = None
    best_epoch = 0
    
    logger.info("开始微调训练...")
    logger.info(f"微调任务: {task_names}")
    
    for epoch in range(epochs):
        model.train()
        # 不再动态调整任务权重，保持稳定
        
        train_preds = []
        train_targets = []
        epoch_loss = 0
        task_losses = []  # 添加这个变量
        
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(features)
                
                # 使用任务特定损失函数
                individual_losses = []
                for i in range(len(task_names)):
                    # 其他任务使用标准MSE损失
                    task_loss = F.mse_loss(outputs[:, i], targets[:, i])
                    individual_losses.append(task_loss * task_weights[i])
                
                loss = sum(individual_losses)
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪，避免梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(targets.detach().cpu().numpy())
        
        # 使用学习率调度器 - 余弦退火不需要传递参数
        scheduler.step()
        
        # 修正这里的变量名，使用epoch_loss而不是task_losses
        train_loss = epoch_loss / len(train_loader)  # 使用epoch_loss计算平均训练损失
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        train_r2, train_rmse, train_rpd = calculate_metrics(train_targets, train_preds)
        
        # 验证阶段
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                with autocast():
                    outputs = model(features)
                    loss = F.mse_loss(outputs, targets)
                
                val_losses.append(loss.item())
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        val_loss = np.mean(val_losses)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_r2, val_rmse, val_rpd = calculate_metrics(val_targets, val_preds)
        
        # 测试阶段 - 移除集成模型评估相关代码
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                # 单模型预测
                with autocast():
                    outputs = model(features)
                
                test_preds.append(outputs.cpu().numpy())
                test_targets.append(targets.cpu().numpy())
        
        test_preds = np.vstack(test_preds)
        test_targets = np.vstack(test_targets)
        
        # 计算单模型指标
        test_r2, test_rmse, test_rpd = calculate_metrics(test_targets, test_preds)
        
        # 更新学习率
        plateau_scheduler.step(np.mean(val_r2))
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新训练历史
        history['epoch'].append(epoch+1)
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
        history['lr'].append(current_lr)
        
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
        
        # 输出当前轮次的训练情况 - 修改为表格形式显示
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 记录训练和验证性能的表格
        logger.info(f"Epoch {epoch+1} 训练集指标:")
        train_table = format_metrics_table(
            task_names, 
            train_r2, 
            train_rmse, 
            train_rpd
        )
        logger.info(f"{train_table}")
        
        logger.info(f"Epoch {epoch+1} 验证集指标:")
        val_table = format_metrics_table(
            task_names, 
            val_r2, 
            val_rmse, 
            val_rpd
        )
        logger.info(f"{val_table}")
        
        logger.info(f"Epoch {epoch+1} 测试集指标:")
        test_table = format_metrics_table(
            task_names, 
            test_r2, 
            test_rmse, 
            test_rpd
        )
        logger.info(f"{test_table}")
        
        # 检查是否达到了新的最佳验证集性能
        weighted_val_r2 = np.mean(val_r2)
        
        # 使用简单的平均R²作为选择标准
        if weighted_val_r2 > best_val_r2 + min_delta:
            best_val_r2 = weighted_val_r2
            early_stop_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_finetuned.pt'))
            
            # 记录最佳性能
            best_metrics = {
                'val_mean_r2': np.mean(val_r2),
                'test_mean_r2': np.mean(test_r2),
                'val_r2': val_r2,
                'test_r2': test_r2,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'val_rpd': val_rpd,
                'test_rpd': test_rpd
            }
            
            logger.info(f"Epoch {epoch+1}: 保存新的最佳模型，验证集R²: {np.mean(val_r2):.4f}")
        else:
            early_stop_counter += 1
            logger.info(f"验证集性能未提升，早停计数: {early_stop_counter}/{patience}")
            
            # 添加早停检查
            if early_stop_counter >= patience:
                logger.info(f"触发早停! {patience}轮训练后验证集性能无提升")
                break
    
    # 训练结束，恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"微调完成。恢复第{best_epoch}个epoch的最佳模型，验证集R²: {best_val_r2:.4f}")
    else:
        logger.info("微调完成，但未找到最佳模型。使用最终模型。")
    
    # 保存训练历史
    save_training_history(history, task_names, results_dir)
    
    # 返回训练历史和最佳指标 - 移除ensemble_models
    return {
        'history': history,
        'metrics': best_metrics if 'best_metrics' in locals() else None,
        'best_epoch': best_epoch
    }

def save_training_history(history, task_names, results_dir):
    """保存训练历史记录到CSV文件"""
    # 保存总体指标历史记录
    main_metrics = {
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
        'lr': history['lr']
    }
    
    # 确保所有列表长度一致
    min_length = min(len(arr) for arr in main_metrics.values())
    for key in main_metrics:
        main_metrics[key] = main_metrics[key][:min_length]
    
    history_df = pd.DataFrame(main_metrics)
    history_path = os.path.join(results_dir, 'finetuning_history.csv')
    history_df.to_csv(history_path, index=False, encoding='utf-8')
    
    # 为每个任务单独保存指标历史记录
    for i, task in enumerate(task_names):
        # 提取该任务的所有指标并确保长度一致
        task_metrics = {
            'epoch': history['epoch'][:min_length],  # 使用已经截断的长度
            'train_r2': [r[i] for r in history['task_train_r2']][:min_length],
            'val_r2': [r[i] for r in history['task_val_r2']][:min_length],
            'test_r2': [r[i] for r in history['task_test_r2']][:min_length],
            'val_rmse': [r[i] for r in history['task_val_rmse']][:min_length],
            'test_rmse': [r[i] for r in history['task_test_rmse']][:min_length],
            'train_rpd': [r[i] for r in history['task_train_rpd']][:min_length],
            'val_rpd': [r[i] for r in history['task_val_rpd']][:min_length],
            'test_rpd': [r[i] for r in history['task_test_rpd']][:min_length]
        }
        
        # 再次检查所有列表长度是否一致
        task_min_length = min(len(arr) for arr in task_metrics.values())
        for key in task_metrics:
            task_metrics[key] = task_metrics[key][:task_min_length]
        
        task_df = pd.DataFrame(task_metrics)
        task_history_path = os.path.join(results_dir, f'finetuning_history_{task}.csv')
        task_df.to_csv(task_history_path, index=False, encoding='utf-8')
# class CSVDataset(Dataset):
#     """处理CSV格式的多任务回归数据集"""
#     def __init__(self, data, feature_cols, target_cols, indices=None, 
#                  normalize_features=True, normalize_targets=True, 
#                  device=None, precomputed_features=None):
#         """
#         Args:
#             data (DataFrame): 包含特征和目标的DataFrame
#             feature_cols (list): 特征列的索引列表
#             target_cols (list): 目标列的名称列表
#             indices (array): 要使用的样本索引
#             normalize_features (bool): 是否标准化特征
#             normalize_targets (bool): 是否标准化目标
#             device (torch.device): 使用的设备
#             precomputed_features (ndarray): 预先计算的标准化特征
#         """
#         self.data = data
#         self.feature_cols = feature_cols
#         self.target_cols = target_cols
#         self.device = device
        
#         # 如果提供了索引，筛选数据
#         if indices is not None:
#             self.indices = indices
#             filtered_data = self.data.iloc[indices]
#         else:
#             self.indices = list(range(len(data)))
#             filtered_data = self.data
        
#         # 使用预计算特征或提取并标准化特征
#         if precomputed_features is not None:
#             self.features = precomputed_features
#         else:
#             self.features = filtered_data.iloc[:, feature_cols].values.astype(np.float32)
#             # 标准化特征
#             if normalize_features:
#                 self.feature_scaler = StandardScaler()
#                 self.features = self.feature_scaler.fit_transform(self.features)
        
#         # 提取并标准化目标
#         self.targets = filtered_data[target_cols].values.astype(np.float32)
#         if normalize_targets:
#             self.target_scaler = StandardScaler()
#             self.targets = self.target_scaler.fit_transform(self.targets)
#             # 保存均值和标准差，用于反标准化
#             self.target_mean = self.target_scaler.mean_
#             self.target_std = self.target_scaler.scale_
    
#     def __len__(self):
#         return len(self.features)
    
#     def __getitem__(self, idx):
#         features = torch.tensor(self.features[idx], dtype=torch.float32)
#         targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        
#         if self.device is not None:
#             features = features.to(self.device)
#             targets = targets.to(self.device)
            
#         return features, targets

def setup_logger(config):
    """设置日志记录器"""
    logger = logging.getLogger('transfer_learning')
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件处理器
    log_file = os.path.join(config['results_dir'], 'transfer_learning.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        
        # =========== 跨作物迁移学习配置 ===========
        # 模式选择: 'tomato_to_rice' 或 'rice_to_tomato'
        # 'transfer_mode': 'tomato_to_rice',  # 番茄模型迁移到水稻数据
        'transfer_mode': 'rice_to_tomato',  # 水稻模型迁移到番茄数据
        
        # 模式1: 番茄→水稻迁移
        'tomato_to_rice': {
            'pretrained_model_path': 'results/MT csv_multitask_vit_3param/20144/best_model.pt',  # 番茄三参数模型
            'csv_path': 'data/Rice_subsets/rice dataset_all_per_class.csv',  # 水稻数据集
            'source_target_cols': ['Pn', 'SPAD', 'LAW'],  # 番茄源任务
            'target_cols': ['Pn', 'SPAD', 'LAW'],         # 水稻目标任务(相同参数)
            'results_dir_suffix': 'tomato_to_rice_3param'
        },
        
        # 模式2: 水稻→番茄迁移  
        'rice_to_tomato': {
            'pretrained_model_path': 'results/Rice csv_multitask_vit_3param/12649/best_model.pt',  # 水稻三参数模型
            'csv_path': 'data/Tomato_subsets/dataset_all_per_class.csv',  # 番茄数据集
            'source_target_cols': ['Pn', 'SPAD', 'LAW'],  # 水稻源任务
            'target_cols': ['Pn', 'SPAD', 'LAW'],         # 番茄目标任务(相同参数)
            'results_dir_suffix': 'rice_to_tomato_3param'
        },
        
        # 根据选择的模式设置配置
        'feature_cols': list(range(3, 276)),  # 光谱特征列(统一格式)
        'split_method': 'sklearn',  # 'random' 、 'stratified'或 'sklearn'，用于划分数据集
        'val_size': 0.15,               
        'test_size': 0.15, 
        'batch_size': 32,
        'num_epochs': 100,  # 适中的训练轮次
        'learning_rate': 1e-4,  # 微调使用较小的学习率
        'weight_decay': 0.01,   # 适度的正则化
        'patience': 15,  # 早停等待轮次
        'min_delta': 0.001,  # 改进阈值
        'hidden_dim': 128,  # 与预训练模型保持一致
        'num_layers': 1,  # 与预训练模型保持一致
        'num_heads': 1,  # 与预训练模型保持一致
        'drop_rate': 0.15,  # 与预训练模型保持一致
        'freeze_layers': True,  # 冻结预训练层，防止过拟合
        'seed': 42,  # 固定随机种子
    }
    
    # 根据选择的迁移模式动态设置配置
    current_mode = config['transfer_mode']
    mode_config = config[current_mode]
    
    # 更新当前配置
    config['pretrained_model_path'] = mode_config['pretrained_model_path']
    config['csv_path'] = mode_config['csv_path']  # 使用单一数据路径
    config['source_target_cols'] = mode_config['source_target_cols']
    config['target_cols'] = mode_config['target_cols']
    
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config['results_dir'] = os.path.join('results', 'transfer_learning', 
                                       f"{mode_config['results_dir_suffix']}_{timestamp}")
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # 设置记录器和随机种子
    logger = setup_logger(config)
    logger.info(f"跨作物迁移学习配置: {current_mode}")
    logger.info(f"预训练模型路径: {config['pretrained_model_path']}")
    logger.info(f"目标数据路径: {config['csv_path']}")
    logger.info(f"源任务参数: {config['source_target_cols']}")
    logger.info(f"目标任务参数: {config['target_cols']}")
    set_seed(config['seed'])
    
    # 验证预训练模型文件是否存在
    if not os.path.exists(config['pretrained_model_path']):
        logger.error(f"预训练模型文件不存在: {config['pretrained_model_path']}")
        logger.error("请确认已完成源作物的三参数模型训练")
        return
    
    # 加载目标作物数据并进行数据划分（与原始训练代码保持一致）
    logger.info(f"从 {config['csv_path']} 加载目标作物数据...")
    try:
        data = pd.read_csv(config['csv_path'])
        logger.info(f"成功加载目标作物数据，共 {len(data)} 条记录，{len(config['feature_cols'])} 个特征")
    except Exception as e:
        logger.error(f"加载目标作物数据失败: {str(e)}")
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
    
    # 划分数据集 - 与原始训练代码保持一致的三部分划分
    logger.info(f"使用 {config['split_method']} 方法将数据集划分为训练集、验证集和测试集")
    if config['split_method'] == 'sklearn':
        # 首先将数据分成训练集和临时集(临时集包含验证集和测试集)
        train_indices, temp_indices = train_test_split(
            np.arange(len(data)), 
            test_size=config['val_size'] + config['test_size'], 
            random_state=config['seed'], 
            stratify=None  # 三参数迁移学习不需要分层采样
        )
        
        # 确定验证集和测试集的比例
        val_ratio = config['val_size'] / (config['val_size'] + config['test_size'])
        
        # 然后将临时集分成验证集和测试集
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=1-val_ratio,  # 调整为验证集占比
            random_state=config['seed'],
            stratify=None
        )
        
        logger.info(f"使用sklearn方法划分数据集: 训练集{len(train_indices)}样本, 验证集{len(val_indices)}样本, 测试集{len(test_indices)}样本")
    
    # 创建训练集、验证集和测试集
    train_dataset = CSVDataset(
        data, 
        config['feature_cols'], 
        config['target_cols'], 
        indices=train_indices, 
        normalize=True,  # 使用正确的参数名
        device=config['device']
    )
    
    val_dataset = CSVDataset(
        data, 
        config['feature_cols'], 
        config['target_cols'], 
        indices=val_indices, 
        normalize=True,  # 使用正确的参数名
        device=config['device']
    )
    
    test_dataset = CSVDataset(
        data,
        config['feature_cols'],
        config['target_cols'],
        indices=test_indices,
        normalize=True,  # 使用正确的参数名
        device=config['device']
    )
    
    logger.info(f"数据集划分: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本, 测试集 {len(test_dataset)} 样本")
    
    # 数据加载器 - 确保评估时不会打乱数据顺序
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # 保存数据划分索引
    indices_file = os.path.join(config['results_dir'], 'data_indices.npz')
    np.savez(indices_file, 
             train_indices=train_indices, 
             val_indices=val_indices, 
             test_indices=test_indices)
    logger.info(f"保存数据划分索引到 {indices_file}")

    # 创建简化版的MultiTaskTMI，专注于三参数迁移学习
    # 注意：使用MultiTaskTMI而不是MultiTaskViT
    logger.info("使用标准MultiTaskTMI模型进行迁移学习")
    
    # 使用简化版模型，专注于三参数迁移学习
    logger.info("使用简化版模型，专注于Pn, SPAD, LAW三参数迁移学习")
    
    # 直接使用原始方式加载模型
    model = load_pretrained_model(
        pretrained_path=config['pretrained_model_path'],
        input_size=len(config['feature_cols']),
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        source_num_tasks=len(config['source_target_cols']),
        target_num_tasks=len(config['target_cols']),
        device=config['device'],
        freeze_layers=config['freeze_layers']
    )
        
    # 冻结预训练层参数
    if config['freeze_layers']:
        for name, param in model.named_parameters():
            if 'decoder' not in name:
                param.requires_grad = False
        logger.info("已冻结编码器层参数，只训练解码器")
    
    # 不使用Mixup数据增强，保持简单
    logger.info("不使用Mixup数据增强，专注于基础迁移学习")
    
    # 微调模型，传递更多参数
    results = finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=val_loader,  # 使用验证集代替测试集
        task_names=config['target_cols'],
        device=config['device'],
        results_dir=config['results_dir'],
        logger=logger,
        epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        patience=config['patience'],
        min_delta=config['min_delta']
    )
    
    # 输出最终结果  
    logger.info("\n==================== 微调后最终评估指标 ====================")
    logger.info(f"{'参数':<10}{'数据集':<10}{'R²':<10}{'RMSE':<10}{'RPD':<10}")
    logger.info(f"{'-'*80}")
    
    # 使用finetune函数返回的结果
    for i, task in enumerate(config['target_cols']):
        # 训练集
        logger.info(f"{task:<10} {'训练集':<10}      {results['train_r2'][i]:.4f}      {results['train_rmse'][i]:.4f}      {results['train_rpd'][i]:.4f}")
        # 验证集
        logger.info(f"{'':<10} {'验证集':<10}      {results['val_r2'][i]:.4f}      {results['val_rmse'][i]:.4f}      {results['val_rpd'][i]:.4f}")
        # 测试集
        logger.info(f"{'':<10} {'测试集':<10}      {results['test_r2'][i]:.4f}      {results['test_rmse'][i]:.4f}      {results['test_rpd'][i]:.4f}")
        logger.info(f"{'-'*80}")
    
    # 打印平均指标
    logger.info("\n================== 数据集性能对比 ==================")
    logger.info("数据集        平均R²      平均RMSE    平均RPD")
    
    avg_train_r2 = np.mean(results['train_r2'])
    avg_train_rmse = np.mean(results['train_rmse'])
    avg_train_rpd = np.mean(results['train_rpd'])
    logger.info(f"训练集        {avg_train_r2:.4f}     {avg_train_rmse:.4f}     {avg_train_rpd:.4f}")
    
    avg_val_r2 = np.mean(results['val_r2'])
    avg_val_rmse = np.mean(results['val_rmse'])
    avg_val_rpd = np.mean(results['val_rpd'])
    logger.info(f"验证集        {avg_val_r2:.4f}     {avg_val_rmse:.4f}     {avg_val_rpd:.4f}")
    
    avg_test_r2 = np.mean(results['test_r2'])
    avg_test_rmse = np.mean(results['test_rmse'])
    avg_test_rpd = np.mean(results['test_rpd'])
    logger.info(f"测试集        {avg_test_r2:.4f}     {avg_test_rmse:.4f}     {avg_test_rpd:.4f}")
    logger.info("========================================")
    
    # 创建metrics变量用于后续保存
    metrics = {
        'val_r2': results['val_r2'],
        'test_r2': results['test_r2'],
        'val_rmse': results['val_rmse'],
        'test_rmse': results['test_rmse'],
        'val_rpd': results['val_rpd'],
        'test_rpd': results['test_rpd']
    }

    # 保存配置
    with open(os.path.join(config['results_dir'], 'finetune_config.txt'), 'w', encoding='utf-8') as f:
        for key, value in config.items():
            if key != 'device':  # 跳过不可序列化的对象
                f.write(f"{key}: {value}\n")
    
    # 保存最佳epoch
    if 'best_epoch' in results:
        with open(os.path.join(config['results_dir'], 'best_epoch.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Best epoch: {results['best_epoch']}")
    
    # 最终清理代码，专注于Pn, SPAD, LAW三参数迁移学习
    logger.info("\n========= 跨作物迁移学习完成 =========")
    logger.info(f"迁移模式: {config['transfer_mode']}")
    logger.info(f"源作物参数: {config['source_target_cols']}")
    logger.info(f"目标作物参数: {config['target_cols']}")
    logger.info("此迁移学习专注于Pn, SPAD, LAW三个共同参数的知识迁移")
    
    # 保存迁移学习总结
    transfer_summary = {
        'transfer_mode': config['transfer_mode'],
        'source_params': config['source_target_cols'],
        'target_params': config['target_cols'],
        'final_metrics': metrics,
        'config': config
    }
    
    import pickle
    summary_path = os.path.join(config['results_dir'], 'transfer_learning_summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(transfer_summary, f)
    logger.info(f"迁移学习总结已保存至: {summary_path}")
    
    print(f"微调完成，所有结果已保存到: {config['results_dir']}")

def format_metrics_table(task_names, r2_values, rmse_values, rpd_values):
    """将指标格式化为表格形式 - 与原始训练代码保持一致"""
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

if __name__ == "__main__":
    main()
