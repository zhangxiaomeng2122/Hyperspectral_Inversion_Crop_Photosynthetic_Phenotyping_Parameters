import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def calculate_rpd(y_true, y_pred):
    """
    计算RPD (Ratio of Performance to Deviation)
    RPD = SD(y_true) / RMSE
    """
    if len(y_true) <= 1:
        return 0.0
    
    std_dev = np.std(y_true)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    if rmse == 0:
        return float('inf')
    
    return std_dev / rmse

class MultiOutputDNN(nn.Module):
    """
    多输出深度神经网络模型 - 与Ensemble中单个模型架构完全一致，只是输出层不同
    使用与Ensemble完全相同的架构: 273→256→128→64→N (N为目标参数数量)
    """
    def __init__(self, input_size, output_size, use_batch_norm=True):
        super(MultiOutputDNN, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        
        # 与Ensemble模型完全相同的架构: 273→256→128→64→N
        # 第一层: input_size → 256
        self.layer1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        
        # 第二层: 256 → 128
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        if use_batch_norm:
            self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        # 第三层: 128 → 64
        self.layer3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # 修正：与Ensemble保持一致，使用ReLU而不是GELU
        if use_batch_norm:
            self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        
        # 输出层: 64 → output_size (多输出，这是唯一与Ensemble不同的地方)
        self.output = nn.Linear(64, output_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重 - 与Ensemble使用相同的初始化策略"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # 第一层
        x = self.layer1(x)
        x = self.relu1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.dropout1(x)
        
        # 第二层
        x = self.layer2(x)
        x = self.relu2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.dropout2(x)
        
        # 第三层
        x = self.layer3(x)
        x = self.relu3(x)  # 修正：使用ReLU与Ensemble保持一致
        if self.use_batch_norm:
            x = self.bn3(x)
        x = self.dropout3(x)
        
        # 输出层
        x = self.output(x)
        
        return x
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'architecture': '273→256→128→64→N (与Ensemble单模型架构完全一致)',
            'activation_functions': 'ReLU→ReLU→ReLU (与Ensemble完全一致)',
            'dropout_rates': [0.5, 0.3, 0.2],
            'use_batch_norm': self.use_batch_norm
        }

class MultiOutputDNNModel:
    """
    多输出深度神经网络模型包装器类
    """
    def __init__(self, input_size, output_size, use_batch_norm=True, device='cpu'):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.device = device
        
        # 创建模型 - 使用与Ensemble相同的架构
        self.model = MultiOutputDNN(
            input_size=input_size,
            output_size=output_size,
            use_batch_norm=use_batch_norm
        ).to(device)
        
        # 添加数据预处理器
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_fitted = False
        self.y_fitted = False
        
        self.target_names = None
        self.is_fitted = False
    
    def fit(self, X_train, y_train, target_names, X_val=None, y_val=None, 
            num_epochs=200, batch_size=32, learning_rate=0.001, weight_decay=1e-5,
            patience=20, logger=None):
        """
        训练模型 - 使用数据标准化和更稳定的训练策略
        """
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from torch.cuda.amp import GradScaler, autocast
        
        if logger is None:
            import logging
            logger = logging.getLogger()
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
        
        self.target_names = target_names
        
        # 数据预处理 - 标准化
        logger.info("开始数据预处理...")
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        self.X_fitted = True
        self.y_fitted = True
        
        logger.info(f"原始数据范围 - X: [{X_train.min():.3f}, {X_train.max():.3f}], y: [{y_train.min():.3f}, {y_train.max():.3f}]")
        logger.info(f"标准化后数据范围 - X: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}], y: [{y_train_scaled.min():.3f}, {y_train_scaled.max():.3f}]")
        
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 验证集处理
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val)
            
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_scaled).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 使用更温和的学习率和优化器设置
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=patience//3, verbose=True)
        criterion = nn.MSELoss()
        scaler = GradScaler()
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [] if val_loader else None,
            'train_r2': [],
            'val_r2': [] if val_loader else None
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        logger.info("开始训练MultiOutput DNN模型...")
        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info("注意: 使用与Ensemble完全相同的网络架构(273→256→128→64→N)")
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_preds_list = []
            train_targets_list = []
            
            for features, targets in train_loader:
                optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model(features)
                    loss = criterion(outputs, targets)
                
                # 检查损失是否为NaN
                if torch.isnan(loss):
                    logger.warning(f"检测到NaN损失值在epoch {epoch+1}")
                    continue
                
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                train_preds_list.append(outputs.detach().cpu().numpy())
                train_targets_list.append(targets.detach().cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 计算训练集R2 - 使用标准化后的数据
            train_preds = np.vstack(train_preds_list)
            train_targets = np.vstack(train_targets_list)
            train_r2_scores = []
            
            for i in range(self.output_size):
                try:
                    r2 = r2_score(train_targets[:, i], train_preds[:, i])
                    train_r2_scores.append(r2)
                except:
                    train_r2_scores.append(-999)  # 标记计算失败
            
            avg_train_r2 = np.mean([r2 for r2 in train_r2_scores if r2 > -999])
            
            history['train_loss'].append(avg_train_loss)
            history['train_r2'].append(avg_train_r2)
            
            # 验证阶段
            val_loss = 0.0
            avg_val_r2 = 0.0
            
            if val_loader:
                self.model.eval()
                val_preds_list = []
                val_targets_list = []
                
                with torch.no_grad():
                    for features, targets in val_loader:
                        with autocast():
                            outputs = self.model(features)
                            loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                        val_preds_list.append(outputs.cpu().numpy())
                        val_targets_list.append(targets.cpu().numpy())
                
                avg_val_loss = val_loss / len(val_loader)
                
                # 计算验证集R2
                val_preds = np.vstack(val_preds_list)
                val_targets = np.vstack(val_targets_list)
                val_r2_scores = []
                
                for i in range(self.output_size):
                    try:
                        r2 = r2_score(val_targets[:, i], val_preds[:, i])
                        val_r2_scores.append(r2)
                    except:
                        val_r2_scores.append(-999)
                
                avg_val_r2 = np.mean([r2 for r2 in val_r2_scores if r2 > -999])
                
                history['val_loss'].append(avg_val_loss)
                history['val_r2'].append(avg_val_r2)
                
                # 检查最佳模型 - 基于验证损失而不是R²
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 学习率调度
                scheduler.step(avg_val_loss)
                
                # 早停检查 - 放宽条件
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                # 输出进度 - 每10个epoch输出一次
                if (epoch + 1) % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                              f"Train Loss={avg_train_loss:.4f}, Train R²={avg_train_r2:.4f}, "
                              f"Val Loss={avg_val_loss:.4f}, Val R²={avg_val_r2:.4f}, LR={current_lr:.6f}")
            else:
                # 没有验证集的情况
                if (epoch + 1) % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                              f"Train Loss={avg_train_loss:.4f}, Train R²={avg_train_r2:.4f}, LR={current_lr:.6f}")
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("已加载验证集上的最佳模型")
        
        self.is_fitted = True
        self.history = history
        
        logger.info("MultiOutput DNN模型训练完成")
        
        return self
    
    def predict(self, X, target_names=None):
        """
        预测 - 包含数据预处理
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        self.model.eval()
        
        # 标准化输入数据
        if self.X_fitted:
            X_scaled = self.scaler_X.transform(X)
        else:
            print("输入数据未标准化，可能影响预测效果")
            X_scaled = X
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        # 反标准化预测结果
        if self.y_fitted:
            predictions = self.scaler_y.inverse_transform(predictions_scaled)
        else:
            predictions = predictions_scaled
        
        return predictions
    
    def evaluate(self, X, y, target_names=None):
        """
        评估模型性能
        """
        if target_names is None:
            target_names = self.target_names
        
        y_pred = self.predict(X, target_names)
        
        results = {
            'r2_scores': [],
            'rmse_values': [],
            'mae_values': [],
            'rpd_values': [],
            'predictions': y_pred,
            'targets': y
        }
        
        for i in range(len(target_names)):
            y_true = y[:, i]
            y_predicted = y_pred[:, i]
            
            # 计算评估指标
            r2 = r2_score(y_true, y_predicted)
            rmse = np.sqrt(mean_squared_error(y_true, y_predicted))
            mae = mean_absolute_error(y_true, y_predicted)
            rpd = calculate_rpd(y_true, y_predicted)
            
            results['r2_scores'].append(r2)
            results['rmse_values'].append(rmse)
            results['mae_values'].append(mae)
            results['rpd_values'].append(rpd)
        
        return results
    
    def get_model_info(self):
        """获取模型信息"""
        return self.model.get_model_info()

def create_multioutput_dnn_model(input_size, output_size, device='cpu'):
    """
    创建多输出深度神经网络模型的工厂函数
    与Ensemble中单个模型架构完全一致，只是输出维度不同
    """
    return MultiOutputDNNModel(
        input_size=input_size,
        output_size=output_size,
        use_batch_norm=True,
        device=device
    )
