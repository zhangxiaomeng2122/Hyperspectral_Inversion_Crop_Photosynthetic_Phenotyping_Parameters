import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def calculate_rpd(y_true, y_pred):
    """
    计算RPD (Ratio of Performance to Deviation)
    RPD = SD(y_true) / RMSE
    """
    if len(y_true) <= 1:
        return 0.0  # 避免除以零或计算单个样本的标准差
    
    std_dev = np.std(y_true)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 避免除以零
    if rmse == 0:
        return float('inf')  # 理论上完美的预测
    
    return std_dev / rmse

# def create_ensemble_model(input_dim, output_dim, device):
#     """
#     创建集成模型：为每个目标变量创建一个单独的模型
#     """
#     models = []
#     for _ in range(output_dim):
#         # 创建简单的前馈神经网络
#         model = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 1)
#         ).to(device)
#         models.append(model)
#     return models

def create_ensemble_model(input_size, output_size=15, device='cpu'):
    """创建一个基本集成模型，为每个目标变量训练单独的模型"""
    models = []
    for i in range(output_size):
        model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        ).to(device)
        models.append(model)
    return models


# def train_ensemble(models, train_loader, val_loader, test_loader, target_names, device, 
#                   num_epochs=100, patience=10, logger=None):
#     """
#     训练集成模型：每个目标变量使用一个单独的模型
#     """
#     if logger is None:
#         import logging
#         logger = logging.getLogger()
#         if not logger.handlers:
#             handler = logging.StreamHandler()
#             formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#             handler.setFormatter(formatter)
#             logger.addHandler(handler)
#             logger.setLevel(logging.INFO)
    
#     results = []
    
#     for i, (model, target_name) in enumerate(zip(models, target_names)):
#         logger.info(f"训练目标变量 {target_name} ({i+1}/{len(models)})...")
        
#         # 为每个模型设置优化器
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#         criterion = torch.nn.MSELoss()
        
#         # 记录最佳验证损失
#         best_val_loss = float('inf')
#         best_model_state = None
#         patience_counter = 0
        
#         # 记录结果
#         all_train_losses = []
#         all_val_losses = []
        
#         # 训练循环
#         for epoch in range(num_epochs):
#             model.train()
#             train_loss = 0.0
            
#             for X_batch, y_batch in train_loader:
#                 X_batch = X_batch.to(device)
#                 y_target = y_batch[:, i].unsqueeze(1).to(device)
                
#                 # 前向传播
#                 optimizer.zero_grad()
#                 outputs = model(X_batch)
#                 loss = criterion(outputs, y_target)
                
#                 # 反向传播和优化
#                 loss.backward()
#                 optimizer.step()
                
#                 train_loss += loss.item() * X_batch.size(0)
            
#             train_loss /= len(train_loader.dataset)
#             all_train_losses.append(train_loss)
            
#             # 验证
#             model.eval()
#             val_loss = 0.0
            
#             with torch.no_grad():
#                 for X_batch, y_batch in val_loader:
#                     X_batch = X_batch.to(device)
#                     y_target = y_batch[:, i].unsqueeze(1).to(device)
                    
#                     outputs = model(X_batch)
#                     loss = criterion(outputs, y_target)
                    
#                     val_loss += loss.item() * X_batch.size(0)
            
#             val_loss /= len(val_loader.dataset)
#             all_val_losses.append(val_loss)
            
#             # 输出进度
#             if (epoch + 1) % 10 == 0 or epoch == 0:
#                 logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
#             # 检查是否需要保存最佳模型
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_model_state = model.state_dict().copy()
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
                
#             # 早停
#             if patience_counter >= patience:
#                 logger.info(f"Early stopping at epoch {epoch+1}")
#                 break
        
#         # 加载最佳模型
#         if best_model_state is not None:
#             model.load_state_dict(best_model_state)
        
#         # 评估模型
#         model.eval()
        
#         # 收集预测结果和真实值
#         train_preds = []
#         train_targets = []
#         val_preds = []
#         val_targets = []
#         test_preds = []
#         test_targets = []
        
#         with torch.no_grad():
#             # 训练集预测
#             for X_batch, y_batch in train_loader:
#                 X_batch = X_batch.to(device)
#                 y_target = y_batch[:, i].unsqueeze(1).to(device)
                
#                 outputs = model(X_batch)
                
#                 train_preds.append(outputs.cpu().numpy())
#                 train_targets.append(y_target.cpu().numpy())
            
#             # 验证集预测
#             for X_batch, y_batch in val_loader:
#                 X_batch = X_batch.to(device)
#                 y_target = y_batch[:, i].unsqueeze(1).to(device)
                
#                 outputs = model(X_batch)
                
#                 val_preds.append(outputs.cpu().numpy())
#                 val_targets.append(y_target.cpu().numpy())
            
#             # 测试集预测
#             for X_batch, y_batch in test_loader:
#                 X_batch = X_batch.to(device)
#                 y_target = y_batch[:, i].unsqueeze(1).to(device)
                
#                 outputs = model(X_batch)
                
#                 test_preds.append(outputs.cpu().numpy())
#                 test_targets.append(y_target.cpu().numpy())
        
#         # 合并批次结果
#         train_preds = np.vstack(train_preds)
#         train_targets = np.vstack(train_targets)
#         val_preds = np.vstack(val_preds)
#         val_targets = np.vstack(val_targets)
#         test_preds = np.vstack(test_preds)
#         test_targets = np.vstack(test_targets)
        
#         # 计算评价指标
#         train_r2 = r2_score(train_targets, train_preds)
#         train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
#         train_mae = mean_absolute_error(train_targets, train_preds)
#         train_rpd = calculate_rpd(train_targets.flatten(), train_preds.flatten())
        
#         val_r2 = r2_score(val_targets, val_preds)
#         val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
#         val_mae = mean_absolute_error(val_targets, val_preds)
#         val_rpd = calculate_rpd(val_targets.flatten(), val_preds.flatten())
        
#         test_r2 = r2_score(test_targets, test_preds)
#         test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
#         test_mae = mean_absolute_error(test_targets, test_preds)
#         test_rpd = calculate_rpd(test_targets.flatten(), test_preds.flatten())
        
#         # 保存结果
#         result = {
#             'model': model,
#             'target': target_name,
#             'train_loss': all_train_losses,
#             'val_loss': all_val_losses,
#             'train_r2': train_r2,
#             'train_rmse': train_rmse,
#             'train_mae': train_mae,
#             'train_rpd': train_rpd,
#             'train_preds': train_preds,
#             'train_targets': train_targets,
#             'val_r2': val_r2,
#             'val_rmse': val_rmse,
#             'val_mae': val_mae,
#             'val_rpd': val_rpd,
#             'val_preds': val_preds,
#             'val_targets': val_targets,
#             'test_r2': test_r2,
#             'test_rmse': test_rmse,
#             'test_mae': test_mae,
#             'test_rpd': test_rpd,
#             'test_preds': test_preds,
#             'test_targets': test_targets
#         }
        
#         results.append(result)
        
#         logger.info(f"目标变量 {target_name} 的评估结果:")
#         logger.info(f"  训练集: R²={train_r2:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, RPD={train_rpd:.4f}")
#         logger.info(f"  验证集: R²={val_r2:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, RPD={val_rpd:.4f}")
#         logger.info(f"  测试集: R²={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, RPD={test_rpd:.4f}")
    
#     # 显示各参数测试集性能
#     logger.info(f"\n=================== 各参数测试集性能 ===================")
#     logger.info(f"{'参数':<12}{'R²':<12}{'RMSE':<12}{'MAE':<12}{'RPD':<12}")
#     logger.info(f"-------------------------------------------------")
    
#     # 计算所有参数的平均指标
#     avg_test_r2 = np.mean([r['test_r2'] for r in results])
#     avg_test_rmse = np.mean([r['test_rmse'] for r in results])
#     avg_test_mae = np.mean([r['test_mae'] for r in results])
#     avg_test_rpd = np.mean([r['test_rpd'] for r in results])
    
#     for i, result in enumerate(results):
#         target_name = target_names[i]
#         test_r2 = result['test_r2']
#         test_rmse = result['test_rmse']
#         test_mae = result['test_mae']
#         test_rpd = result['test_rpd']
        
#         logger.info(f"{target_name:<12}{test_r2:<12.4f}{test_rmse:<12.4f}{test_mae:<12.4f}{test_rpd:<12.4f}")
    
#     logger.info(f"-------------------------------------------------")
#     logger.info(f"{'平均':<12}{avg_test_r2:<12.4f}{avg_test_rmse:<12.4f}{avg_test_mae:<12.4f}{avg_test_rpd:<12.4f}")
#     logger.info(f"=================================================")
    
#     return results

# def train_advanced_ensemble(models, train_loader, val_loader, test_loader, target_names, device, 
#                            num_epochs=100, patience=10, logger=None):
#     """
#     增强版集成模型训练函数 - 可以添加更多高级特性
#     """
#     # 目前这个函数是train_ensemble的副本，您可以根据需要添加更多高级特性
#     return train_ensemble(models, train_loader, val_loader, test_loader, target_names, device, 
#                          num_epochs, patience, logger)
