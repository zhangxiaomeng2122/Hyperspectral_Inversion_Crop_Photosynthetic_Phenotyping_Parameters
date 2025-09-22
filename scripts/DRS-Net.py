import os
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


# ���� MultiTaskViT ģ��
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# �����Զ���ģ��
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
    Ϊÿ������Ӧ�ö�����Ȩ�غ�ƫ�ã�ʵ�ֲ���ѡ���ܡ�
    ��ÿ������Ӧ�ö�����1x1������ϸ������Ĺ�ʽʵ�֡�
    """
    def __init__(self, num_bands):
        super(BandwiseAffineLayer, self).__init__()
        self.num_bands = num_bands
        # �Ľ�Ȩ�س�ʼ��
        self.weights = nn.Parameter(torch.ones(num_bands) + 0.01 * torch.randn(num_bands))
        self.bias = nn.Parameter(torch.zeros(num_bands))
        # ���������һ�������ѵ���ȶ���
        self.batch_norm = nn.BatchNorm1d(num_bands)
        
    def forward(self, x):
        # �Ƚ���������һ��
        if self.training:
            x = self.batch_norm(x)
        # Ȼ��Ӧ��Ȩ�غ�ƫ��
        weighted_bands = x * self.weights + self.bias
        return weighted_bands

    def get_band_importance(self):
        """��ȡÿ�����ε���Ҫ��Ȩ��"""
        # ����Ȩ�ؾ���ֵ��Ϊ������Ҫ��
        return torch.abs(self.weights).detach().cpu().numpy()

# 2. ����Ӳ��ֵѡ�����
class TopKBandGatingLayer(nn.Module):
    """
    ����BandwiseIndependentConvѧϰ��Ȩ�ؽ��в���ѡ��
    �ϸ��������е�Ӳ��ֵ����ʵ��
    """
    def __init__(self, num_bands, k_bands):
        """
        Args:
            num_bands: �ܲ�����
            k_bands: Ҫѡ��Ĳ������� (�����е�u����)
        """
        super(TopKBandGatingLayer, self).__init__()
        self.num_bands = num_bands
        self.k_bands = k_bands
            
    def forward(self, x, weights, bias):
        """
        ʵ�������еĹ�ʽ(1)
        Args:
            x: �������� [batch_size, num_bands]
            weights: ����Ȩ�� [num_bands]
            bias: ����ƫ�� [num_bands]
        """
        # ���㲨����Ҫ��
        band_importance = torch.abs(weights)
        
        # ���������е�Ӳ��ֵ��������ȡ��(u+1)�����ֵ��Ϊ��ֵ
        # ������󣬵�k��λ�õ�ֵ����0��ʼ������
        sorted_importance, _ = torch.sort(band_importance, descending=True)
        if self.k_bands < self.num_bands:
            threshold = sorted_importance[self.k_bands-1]  # ��k�����ֵ��Ϊ��ֵ
        else:
            threshold = torch.min(band_importance) - 1e-6  # ���k���ڲ���������ѡ�����в���
        
        # �������룬��������Ҫ�Դ�����ֵ�Ĳ���
        mask = (band_importance >= threshold).float()
        
        # ʵ�ֹ�ʽ(1)��ѡ���߼�:
        # ������Ҫ�Ĳ���: w_k �� x_i,k + b_k
        # ���ڲ���Ҫ�Ĳ���: b_k
        selected_features = torch.zeros_like(x)
        important_bands = torch.where(mask > 0)[0]
        
        # ����Ҫ����Ӧ��Ȩ�غ�ƫ��
        selected_features[:, important_bands] = x[:, important_bands] * weights[important_bands] + bias[important_bands]
        
        # �Բ���Ҫ����ֻ����ƫ��
        unimportant_bands = torch.where(mask == 0)[0]
        if len(unimportant_bands) > 0:
            selected_features[:, unimportant_bands] = bias[unimportant_bands]
        
        return selected_features, mask

# 3. ����������BHCNN�ع�ģ��
class BHCNNRegression(nn.Module):
    def __init__(self, input_size, num_tasks, k_bands=None, threshold=None, hidden_dims=(128, 64)):
        super(BHCNNRegression, self).__init__()
        
        self.input_size = input_size
        self.num_tasks = num_tasks
        
        # ���ζ�������㣬ѧϰ������Ҫ��
        self.bandwise_conv = BandwiseAffineLayer(input_size)
        
        # ����ѡ���
        self.band_selection = TopKBandGatingLayer(input_size, k_bands, threshold)
        
        # ������ȡ��
        layers = []
        prev_dim = input_size  # ����ά����ѡ���Ĳ�����
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        self.feature_extraction = nn.Sequential(*layers)
        
        # �ع������
        self.regression_head = nn.Linear(prev_dim, num_tasks)
        
    def forward(self, x):
        # Ӧ�ò���Ȩ��
        weighted_bands = self.bandwise_conv(x)
        
        # ��ѵ��ģʽ�£�������Ӳѡ�������в��β���ѵ��
        if self.training:
            features = weighted_bands
        else:
            # ������ģʽ�£�Ӧ��Ӳѡ��
            features, _ = self.band_selection(x, self.bandwise_conv.weights)
        
        # ������ȡ
        features = self.feature_extraction(features)
        
        # �ع����
        outputs = self.regression_head(features)
        
        return outputs
    
    def get_selected_bands(self):
        """��ȡģ��ѡ��Ĳ�������"""
        band_importance = self.bandwise_conv.get_band_importance()
        
        if hasattr(self.band_selection, 'k_bands') and self.band_selection.k_bands is not None:
            # ���������k_bands����������Ҫ��k������
            top_indices = np.argsort(band_importance)[::-1][:self.band_selection.k_bands]
            return top_indices
        else:
            # ������ֵ������Ҫ����
            threshold = self.band_selection.threshold
            selected_indices = np.where(band_importance > threshold)[0]
            return selected_indices
            
# 4. �� BandwiseAffineLayer �� MultiTaskViT ���ɵ���ģ�ͣ�����CFL˫��֧�ܹ�
class BandwiseCFLViT(nn.Module):
    """
    ��ϲ���ѡ��� ViT ģ�͵ļܹ�������CFL (Coarse-to-Fine Loss)˫��֧�ṹ
    """
    def __init__(self, input_size, num_tasks, k_bands=None,
                hidden_dim=512, num_layers=4, num_heads=8, drop_rate=0.1):
        super(BandwiseCFLViT, self).__init__()
        
        self.input_size = input_size
        self.num_tasks = num_tasks
        
        # ���ζ�������� - ����Ȩ������������֧
        self.bandwise_conv = BandwiseAffineLayer(input_size)
        
        # Ӳ��ֵѡ��� - ������ѡ�񲨶η�֧
        self.band_selection = TopKBandGatingLayer(input_size, k_bands)

        # ����� ViT �ع�ģ�� - ����������֧
        self.vit_model = MultiTaskViT(
            input_size=input_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_tasks=num_tasks,
            drop=drop_rate
        )
        
        # ��ǰ���������ܵ����������ڼ���ֵ�ϸ��ʧ��Ȩ������
        self.current_iter = 0
        self.total_iters = 0
        
    def forward(self, x, return_extra=True):
        """
        ǰ�򴫲����� - ʵ��˫��֧�ṹ
        
        Args:
            x: �������� [batch_size, num_bands]
            return_extra: �Ƿ񷵻ض����֧�������ѵ��ʱΪTrue������ʱΪFalse
            
        Returns:
            outputs: ԭʼ��֧�����
            outputs_extra: �����֧�����(���return_extraΪTrue)
        """
        # ȫ���η�֧ - Ӧ�ò���Ȩ�غ�ֱ������ع�ģ��
        weighted_bands = self.bandwise_conv(x)
        coarse_output = self.vit_model(weighted_bands)
        
        # ����ѡ���֧ - Ӧ�ò���Ȩ�غ����Ӳ��ֵѡ��
        selected_bands, _ = self.band_selection(x, self.bandwise_conv.weights, self.bandwise_conv.bias)
        fine_output = self.vit_model(selected_bands)
        
        if self.training and return_extra:
            # ѵ��ʱ������������֧���������ʧ����ʹ��
            # fine_output - ѡ�񲨶κ����� (����ϸ�����)
            # coarse_output - ȫ������� (���ֲڵ����)
            return fine_output, coarse_output
        else:
            # ����ʱֻʹ��ѡ�񲨶κ�����
            return fine_output
    
    def get_selected_bands(self):
        """��ȡģ��ѡ��Ĳ�������"""
        band_importance = self.bandwise_conv.get_band_importance()
        
        # ���ݲ�����Ҫ������
        sorted_indices = np.argsort(band_importance)[::-1]
        
        # ��������Ҫ��k������
        return sorted_indices[:self.band_selection.k_bands]
    
    def set_iteration_params(self, current, total):
        """���õ�ǰ�������ܵ����������ڼ���ֵ�ϸ��ʧȨ��"""
        self.current_iter = current
        self.total_iters = total

# ��Ӵֵ�ϸ��ʧ����
class CoarseToFineLoss(nn.Module):
    """
    ʵ�������еĴֵ�ϸ��ʧ����
    ����ѵ�����У��𲽽�����Ӵ�������ʧ(���в���)ת�Ƶ�ϸ������ʧ(ѡ��Ĳ���)
    """
    def __init__(self, base_criterion=nn.MSELoss()):
        super(CoarseToFineLoss, self).__init__()
        self.base_criterion = base_criterion
    
    def forward(self, fine_outputs, coarse_outputs, targets, current_iter, total_iters):
        """
        ����ֵ�ϸ��ʧ
        
        Args:
            fine_outputs: ϸ���������ѡ�񲨶κ�
            coarse_outputs: ��������������в��Σ�
            targets: Ŀ��ֵ
            current_iter: ��ǰ������
            total_iters: �ܵ�����
            
        Returns:
            loss: �����ʧ
        """
        # ����������� ��
        sigma = 1.0 - current_iter / total_iters
        
        # ����ϸ������ʧ��ѡ�񲨶κ�
        fine_loss = self.base_criterion(fine_outputs, targets)
        
        # �����������ʧ�����в��Σ�
        coarse_loss = self.base_criterion(coarse_outputs, targets)
        
        # �����ʧ
        loss = sigma * coarse_loss + (1.0 - sigma) * fine_loss
        
        return loss, coarse_loss, fine_loss, sigma

# �޸�ѵ��������֧��CFLģ��
def train_cfl_model(model, train_loader, val_loader, test_loader, task_names, device, results_dir, logger,
                     num_epochs=500, learning_rate=0.001, weight_decay=1e-5, patience=40, min_delta=0.0005,
                     early_stopping=True, metric_monitor='val_r2'):
    """ѵ��BHCNN-CFLģ�ͣ�ʹ�ôֵ�ϸ��ʧ����"""
    
    # �Ż��� - ʹ��AdamW����Adam��ͨ���и��õ�����
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # ������ʧ���� - ����ʹ��SmoothL1Loss����ǿ���쳣ֵ��³����
    base_criterion = nn.SmoothL1Loss()
    
    # �ֵ�ϸ��ʧ����
    cfl_criterion = CoarseToFineLoss(base_criterion)
    
    # ѧϰ�ʵ�����
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience//2, 
                                                   factor=0.5, min_lr=1e-6)
    
    # ������ѵ����������
    total_iters = num_epochs * len(train_loader)
    current_iter = 0
    
    # ��ʼ����ͣ����
    best_val_metric = -float('inf')
    best_epoch = 0
    no_improve_count = 0
    best_model_path = os.path.join(results_dir, 'best_model.pth')
    
    # ��ʼ��ѵ����ʷ��¼
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_r2': [], 'val_r2': [], 'test_r2': [],
        'train_rmse': [], 'val_rmse': [], 'test_rmse': [],
        'train_rpd': [], 'val_rpd': [], 'test_rpd': [],
        'lr': [], 'sigma': [],  # ���sigma��¼
        'coarse_loss': [], 'fine_loss': [],  # ��Ӵ�ϸ��ʧ��¼
        # �����ض�ָ��
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
    
    # �����ݶ����������ڻ�Ͼ���ѵ��
    scaler = GradScaler()
    
    # ��ѵ��ѭ��
    logger.info("��ʼѵ��...")
    start_time = time.time()

    # ����ѧϰ��Ԥ������
    warmup_epochs = 10
    
    # ��ӵ�����Ϣ
    logger.info("ģ��ѵ������:")
    logger.info(f"ѧϰ��: {learning_rate}, Ȩ��˥��: {weight_decay}")
    logger.info(f"Ԥ������: {warmup_epochs}, ��ѵ������: {num_epochs}")
    logger.info(f"������С: {train_loader.batch_size}, ��ѵ������: {len(train_loader.dataset)}")
    logger.info(f"�����ξ����Ȩ�أ�ʹ�ôֵ�ϸ��ʧ����(CFL)����ѵ��")
    logger.info(f"ѡ�񲨶�����: {model.band_selection.k_bands}")
    
    for epoch in range(num_epochs):
        # ѵ���׶�
        model.train()
        train_losses = []
        coarse_losses = []
        fine_losses = []
        sigma_values = []
        
        # ѧϰ��Ԥ��
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * ((epoch + 1) / warmup_epochs)
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            # ���µ�ǰ��������
            current_iter += 1
            
            # ����ģ�͵ĵ�������
            model.set_iteration_params(current_iter, total_iters)
            
            # ȷ����������ȷ���豸��
            if features.device != device:
                features = features.to(device)
            if targets.device != device:
                targets = targets.to(device)
                
            optimizer.zero_grad()
            
            # ʹ�û�Ͼ��Ȳ�����ݶȲü�
            with autocast():
                # ǰ�򴫲� - ��ȡ˫��֧���
                # fine_output - ѡ�񲨶κ�����
                # coarse_output - ȫ�������
                fine_output, coarse_output = model(features, return_extra=True)
                
                # ����CFL��ʧ
                loss, coarse_loss, fine_loss, sigma = cfl_criterion(
                    fine_output, coarse_output, targets, current_iter, total_iters)
            
            # ���򴫲�
            scaler.scale(loss).backward()
            
            # ����ݶȲü�����ֹ�ݶȱ�ը
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # ��¼��ʧֵ
            train_losses.append(loss.item())
            coarse_losses.append(coarse_loss.item())
            fine_losses.append(fine_loss.item())
            sigma_values.append(sigma)
            
        # ����ƽ����ʧ
        train_loss = np.mean(train_losses)
        avg_coarse_loss = np.mean(coarse_losses)
        avg_fine_loss = np.mean(fine_losses)
        avg_sigma = np.mean(sigma_values)
        
        # ��֤�׶�
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                if features.device != device:
                    features = features.to(device)
                if targets.device != device:
                    targets = targets.to(device)
                    
                # ����ģʽֻʹ��fine���
                outputs = model(features, return_extra=False)
                loss = base_criterion(outputs, targets)
                val_losses.append(loss.item())
        
        # ������֤��ʧ
        val_loss = np.mean(val_losses)
        
        # �������ָ��
        train_r2, train_rmse, train_rpd = evaluate_model(model, train_loader, device, len(task_names))
        val_r2, val_rmse, val_rpd = evaluate_model(model, val_loader, device, len(task_names))
        test_r2, test_rmse, test_rpd = evaluate_model(model, test_loader, device, len(task_names))
        
        # ����ѧϰ��
        current_val_metric = np.mean(val_r2) if metric_monitor == 'val_r2' else -np.mean(val_rmse)
        scheduler.step(current_val_metric)
        
        # ��¼��ʷ
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
        
        # ��¼ÿ�������ָ��
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
        
        # ��ӡ����
        if (epoch+1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                      f"ѵ����ʧ: {train_loss:.4f}, ��֤��ʧ: {val_loss:.4f}, "
                      f"����ʧ: {avg_coarse_loss:.4f}, ϸ��ʧ: {avg_fine_loss:.4f}, ��: {avg_sigma:.4f}, "
                      f"ѵ��R�0�5: {np.mean(train_r2):.4f}, ��֤R�0�5: {np.mean(val_r2):.4f}")
        
        # ��ʽ�������ָ��
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"ѵ����ʧ: {train_loss:.4f}, ��֤��ʧ: {val_loss:.4f}")

        logger.info("ѵ����ָ��:")
        logger.info(format_metrics_table(task_names, train_r2, train_rmse, train_rpd))
        
        logger.info("��֤��ָ��:")
        logger.info(format_metrics_table(task_names, val_r2, val_rmse, val_rpd))
        
        # �������ģ��
        if current_val_metric > best_val_metric + min_delta:
            best_val_metric = current_val_metric
            best_epoch = epoch
            no_improve_count = 0
            
            # �������ģ��
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
            
            logger.info(f"Epoch {epoch+1}: �������ģ�ͣ���֤ {metric_monitor}: {best_val_metric:.4f}")
        else:
            no_improve_count += 1
        
        # ��ͣ���
        if early_stopping and no_improve_count >= patience:
            logger.info(f"��ͣ������{patience}���ִ����޸��ơ�")
            break
    
    # �������ģ��
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # �������ģ�͵�����ָ��
    train_r2, train_rmse, train_rpd = evaluate_model(model, train_loader, device, len(task_names))
    val_r2, val_rmse, val_rpd = evaluate_model(model, val_loader, device, len(task_names))
    test_r2, test_rmse, test_rpd = evaluate_model(model, test_loader, device, len(task_names))
    
    # ����ѵ��ʱ��
    total_time = time.time() - start_time
    logger.info(f"ѵ����ɣ�����ʱ: {total_time:.2f}�� ({total_time/60:.2f}����)")
    logger.info(f"���ģ�����Ե�{best_epoch+1}�֣���֤R�0�5: {np.mean(val_r2):.4f}������R�0�5: {np.mean(test_r2):.4f}")
    
    # ��ȡ����ѡ����
    band_importance = checkpoint['band_importance']
    selected_bands = checkpoint['selected_bands']
    
    logger.info(f"����ѡ������ѡ����{len(selected_bands)}������")
    
    # ���沨����Ҫ������
    importance_df = pd.DataFrame({
        'Band_Index': np.arange(len(band_importance)),
        'Importance': band_importance
    })
    importance_df.to_csv(os.path.join(results_dir, 'band_importance.csv'), index=False)
    
    # ����ѡ��Ĳ���
    selected_df = pd.DataFrame({
        'Selected_Band_Index': selected_bands
    })
    selected_df.to_csv(os.path.join(results_dir, 'selected_bands.csv'), index=False)
    
    # ����ѵ����ʷ
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
    
    # ����ʹ��һ�����figure������Ϊÿ����ͼ����������figure�����ø�ʽ
    
    # 1. R�0�5 ����ͼ
    setfig(column=1, x=2.5, y=2.8)
    plt.plot(history['epoch'], history['train_r2'], label='Train R�0�5')
    plt.plot(history['epoch'], history['val_r2'], label='Val R�0�5')
    plt.plot(history['epoch'], history['test_r2'], label='Test R�0�5')
    plt.xlabel('Epoch')
    plt.ylabel('R�0�5')
    plt.legend(prop={'size':6, 'family': 'Arial'}, frameon=False) 
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'r2_history.pdf'), format='PDF', transparent=True, bbox_inches='tight')
    plt.close()
    
    # 2. RMSE ����ͼ
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
    
    # 3. ��ʧ����ͼ
    setfig(column=1, x=2.5, y=2.8)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(prop={'size':6, 'family': 'Arial'}, frameon=False) 
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'loss_history.pdf'), format='PDF', transparent=True, bbox_inches='tight')
    plt.close()
    
    # 4. ������Ҫ������ͼ
    setfig(column=1, x=2.5, y=2.8)
    # ʹ��ģ�͵�k_bands���Ի�ֱ��ʹ��selected_bands�ĳ���
    k_bands = model.band_selection.k_bands
    selected_bands_set = set(selected_bands[:k_bands])
    
    # ������ɫ�б���ѡ�еĲ����ú�ɫ��δѡ�е�����ɫ
    colors = ['red' if i in selected_bands_set else 'blue' for i in range(len(band_importance))]
    
    # ���Ʋ�����Ҫ������ͼ
    bars = plt.bar(np.arange(len(band_importance)), band_importance, color=colors)
    
    # ���ͼ��˵��
    red_patch = plt.Rectangle((0, 0), 1, 1, fc="red")
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc="blue")
    plt.legend([red_patch, blue_patch], ['Selected', 'Not Selected'], prop={'size':6, 'family': 'Arial'}, frameon=False)

    plt.xlabel('Band Index')
    plt.ylabel('Importance')
    plt.title(f'Band Importance (Selected: {len(selected_bands_set)} bands)')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'band_importance.pdf'), format='PDF', transparent=True, bbox_inches='tight')
    plt.close()
    
    # 5. �ֵ�ϸ��ʧ����ͼ
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
    
    # ���ؽ��
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
    
    # ����������ݼ��ĶԱȱ��
    logger.info("\n========= �������ݼ����ܶԱ� =========")
    logger.info("���ݼ�        ƽ��R�0�5      ƽ��RMSE    ƽ��RPD")
    logger.info(f"ѵ����      {np.mean(train_r2):.4f}     {np.mean(train_rmse):.4f}     {np.mean(train_rpd):.4f}")
    logger.info(f"��֤��      {np.mean(val_r2):.4f}     {np.mean(val_rmse):.4f}     {np.mean(val_rpd):.4f}")
    logger.info(f"���Լ�      {np.mean(test_r2):.4f}     {np.mean(test_rmse):.4f}     {np.mean(test_rpd):.4f}")
    logger.info("========================================")
    
    # �����ϸ����ָ����� - �޸ı�ͷ����MAPE��
    logger.info("\n========= ��ϸ��������ָ�� =========")
    logger.info(f"{'����':<10}{'���ݼ�':<12}{'R�0�5':<10}{'RMSE':<10}{'MAE':<10}{'MAPE(%)':<10}{'���ͷ���':<12}")
    logger.info(f"{'-'*70}")  # �ӳ��ָ�������Ӧ������MAPE��
    
    # ��ȡ�������ݼ���Ԥ��ֵ��Ŀ��ֵ��ȷ������ƥ��
    train_preds, train_targets_np = get_predictions_and_targets(model, train_loader)
    val_preds, val_targets_np = get_predictions_and_targets(model, val_loader)
    test_preds, test_targets_np = get_predictions_and_targets(model, test_loader)
    
    # ���һ��С��epsilonֵ�������ڼ���MAPEʱ����0
    epsilon = 1e-10
    
    for i, target in enumerate(task_names):
        # ѵ���� - ʹ����Ե�Ԥ��ֵ��Ŀ��ֵ
        train_r2_i = train_r2[i]
        train_rmse_i = train_rmse[i]
        train_mae = np.mean(np.abs(train_targets_np[:,i] - train_preds[:,i]))
        # ����ѵ����MAPE
        # train_mape = np.mean(np.abs((train_targets_np[:,i] - train_preds[:,i]) / (np.abs(train_targets_np[:,i]) + epsilon))) * 100
        absolute_percentage_errors = np.abs((train_targets_np[:,i] - train_preds[:,i]) / (train_targets_np[:,i] + epsilon))
        train_mape = np.mean(absolute_percentage_errors) * 100
        train_ev = 1 - np.var(train_targets_np[:,i] - train_preds[:,i]) / np.var(train_targets_np[:,i])
        
        # ��֤��
        val_r2_i = val_r2[i]
        val_rmse_i = val_rmse[i]
        val_mae = np.mean(np.abs(val_targets_np[:,i] - val_preds[:,i]))
        # ������֤��MAPE
        # val_mape = np.mean(np.abs((val_targets_np[:,i] - val_preds[:,i]) / (np.abs(val_targets_np[:,i]) + epsilon))) * 100
        absolute_percentage_errors = np.abs((val_targets_np[:,i] - val_preds[:,i]) / (val_targets_np[:,i] + epsilon))
        val_mape = np.mean(absolute_percentage_errors) * 100
        val_ev = 1 - np.var(val_targets_np[:,i] - val_preds[:,i]) / np.var(val_targets_np[:,i])
        
        # ���Լ�
        test_r2_i = test_r2[i]
        test_rmse_i = test_rmse[i]
        test_mae = np.mean(np.abs(test_targets_np[:,i] - test_preds[:,i]))
        # ������Լ�MAPE
        # test_mape = np.mean(np.abs((test_targets_np[:,i] - test_preds[:,i]) / (np.abs(test_targets_np[:,i]) + epsilon))) * 100
        absolute_percentage_errors = np.abs((test_targets_np[:,i] - test_preds[:,i]) / (test_targets_np[:,i] + epsilon))
        test_mape = np.mean(absolute_percentage_errors) * 100
        test_ev = 1 - np.var(test_targets_np[:,i] - test_preds[:,i]) / np.var(test_targets_np[:,i])
        
        # ��¼����־ - ����MAPE��
        logger.info(f"{target:<10}{'ѵ����':<12}{train_r2_i:<10.4f}{train_rmse_i:<10.4f}{train_mae:<10.4f}{train_mape:<10.2f}{train_ev:<12.4f}")
        logger.info(f"{'':<10}{'��֤��':<12}{val_r2_i:<10.4f}{val_rmse_i:<10.4f}{val_mae:<10.4f}{val_mape:<10.2f}{val_ev:<12.4f}")
        logger.info(f"{'':<10}{'���Լ�':<12}{test_r2_i:<10.4f}{test_rmse_i:<10.4f}{test_mae:<10.4f}{test_mape:<10.2f}{test_ev:<12.4f}")
        logger.info(f"{'-'*70}")
        
        # ����Щ����ָ����ӵ������
        results['metrics'][f'{target}_train_mae'] = train_mae
        results['metrics'][f'{target}_train_ev'] = train_ev
        results['metrics'][f'{target}_train_mape'] = train_mape  # ���MAPE�������
        results['metrics'][f'{target}_val_mae'] = val_mae
        results['metrics'][f'{target}_val_ev'] = val_ev
        results['metrics'][f'{target}_val_mape'] = val_mape  # ���MAPE�������
        results['metrics'][f'{target}_test_mae'] = test_mae
        results['metrics'][f'{target}_test_ev'] = test_ev
        results['metrics'][f'{target}_test_mape'] = test_mape  # ���MAPE�������
    
    # ����ƽ��MAPE����ӵ������
    results['metrics']['train_mean_mape'] = np.mean([results['metrics'][f'{target}_train_mape'] for target in task_names])
    results['metrics']['val_mean_mape'] = np.mean([results['metrics'][f'{target}_val_mape'] for target in task_names])
    results['metrics']['test_mean_mape'] = np.mean([results['metrics'][f'{target}_test_mape'] for target in task_names])
    
    return results

def main():
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        
        # ����
        'csv_path': 'data/processed/subsets/dataset_max15_per_class.csv',
        'target_cols': ['RUE', 'Pn', 'Gs', 'Ci', 'Tr', 'Ci-Ca', 'WUE', 'iWUE', 'Pmax', 'Rd', 'Ic', 'SPAD', 'LAW'],
        'results_dir': os.path.join('results', 'MT_bandwise_cfl_selection'),  # �޸�Ŀ¼��
        
        # ˮ��
        # 'csv_path': 'data/processed/Rice subsets/rice dataset_all_per_class.csv',
        # 'target_cols': ['SPAD','Pn', 'LNC', 'Chl-a', 'Chl-b','LAW', 'Cx', 'Chl'],  # Rice 8��ˮ������
        # 'results_dir': os.path.join('results', 'Rice_bandwise_cfl_selection'),  # �޸�Ŀ¼��
        'feature_cols': list(range(3, 276)),  # 273������
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
        # ��ͣ����
        'early_stopping': True,
        'patience': 20,               
        'min_delta': 0.0005,
        'metric_monitor': 'val_r2',
    }
    
    # �������Ŀ¼
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config['results_dir'] = os.path.join(config['results_dir'], timestamp)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # ���ü�¼�����������
    logger = setup_logger(config)
    set_seed(config['seed'])

     # ����CSV����
    logger.info(f"�� {config['csv_path']} ��������...")
    try:
        data = pd.read_csv(config['csv_path'])
        logger.info(f"�ɹ��������ݣ��� {len(data)} ����¼��{len(config['feature_cols'])} ������")
    except Exception as e:
        logger.error(f"��������ʧ��: {str(e)}")
        return
    
    # ���Ŀ�����Ƿ����
    missing_cols = [col for col in config['target_cols'] if col not in data.columns]
    if missing_cols:
        logger.error(f"����Ŀ���в�������������: {missing_cols}")
        return
    
    # ��ʾ���еĻ���ͳ����Ϣ
    logger.info("Ŀ�����ͳ����Ϣ:")
    for col in config['target_cols']:
        logger.info(f"{col}: ��ֵ={data[col].mean():.4f}, ��׼��={data[col].std():.4f}, ��Χ=[{data[col].min():.4f}, {data[col].max():.4f}]")
    
    # �������ݼ�
    logger.info(f"ʹ�� {config['split_method']} ���������ݼ�����Ϊѵ��������֤���Ͳ��Լ�")
    # ʹ��sklearn�����������ݼ�����
    train_indices, temp_indices = train_test_split(
        np.arange(len(data)), 
        test_size=config['val_size'] + config['test_size'], 
        random_state=config['seed']
    )
    
    # ȷ����֤���Ͳ��Լ��ı���
    val_ratio = config['val_size'] / (config['val_size'] + config['test_size'])
    
    # Ȼ����ʱ���ֳ���֤���Ͳ��Լ�
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=1-val_ratio,
        random_state=config['seed']
    )
    
    logger.info(f"���ݼ�����: ѵ����{len(train_indices)}����, ��֤��{len(val_indices)}����, ���Լ�{len(test_indices)}����")
    
    # �������ݼ�
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
    
    # ���ݼ�����
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    # ��֤�Ͳ��Լ���ȷ����shuffle=False��ȷ��Ԥ��ֵ��Ŀ��ֵ˳��һ��
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # ����������ѡ��� ViT ģ��
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
    
    # ʹ��CFLѵ������
    training_func = train_cfl_model
    logger.info("ʹ��CFL˫��֧ģ�ͼܹ�����ѵ��")
    
    # ���ģ�ͽṹ
    num_params = count_parameters(model)
    logger.info(f"ģ�ͽṹ:\n{model}")
    logger.info(f"ģ�Ͳ�����: {num_params}")
    
    # ѵ��ģ��
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
    
    # ���ÿ�����������ָ��
    logger.info("\n=================== ���������Լ����� ===================")
    logger.info("����         R�0�5          RMSE        RPD")
    logger.info("-------------------------------------------------")
    metrics = results['metrics']
    for i, task in enumerate(config['target_cols']):
        logger.info(f"{task:<12} {metrics['test_r2'][i]:.4f}     {metrics['test_rmse'][i]:.4f}     {metrics['test_rpd'][i]:.4f}")
    logger.info("-------------------------------------------------")
    logger.info(f"ƽ��        {metrics['test_mean_r2']:.4f}     {metrics['test_mean_rmse']:.4f}     {metrics['test_mean_rpd']:.4f}")
    logger.info("=================================================")
    
    # �������ѡ����
    logger.info("\n=================== ����ѡ���� ===================")
    
    # ��ȡ��Ҫ�Ĳ���������ʵ����������
    selected_bands = results['selected_bands']
    actual_indices = [config['feature_cols'][idx] for idx in selected_bands]
    
    logger.info(f"ѡ����{len(selected_bands)}������")
    logger.info(f"ѡ��Ĳ�������: {actual_indices}")
    logger.info("=================================================")
    
    logger.info(f"ѵ����ɣ������֤R�0�5: {metrics['val_mean_r2']:.4f}, ����R�0�5: {metrics['test_mean_r2']:.4f}")
    print(f"���н���ѱ��浽: {config['results_dir']}")

if __name__ == "__main__":
    main()