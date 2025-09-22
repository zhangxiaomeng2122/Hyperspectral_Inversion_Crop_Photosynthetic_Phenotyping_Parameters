import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import spectral
import torch.nn.functional as F

class HSIDataset(Dataset):
    """高光谱数据集类，用于加载和预处理高光谱数据"""
    
    def __init__(self, X, y, x_mean=None, x_std=None, y_mean=None, y_std=None):
        self.X = X
        self.y = y
        
        # 标准化数据
        if x_mean is None or x_std is None:
            self.x_mean = np.mean(X, axis=0)
            self.x_std = np.std(X, axis=0)
        else:
            self.x_mean = x_mean
            self.x_std = x_std
            
        if y_mean is None or y_std is None:
            self.y_mean = np.mean(y, axis=0)
            self.y_std = np.std(y, axis=0)
        else:
            self.y_mean = y_mean
            self.y_std = y_std
        
        # 应用标准化
        self.X_normalized = (X - self.x_mean) / (self.x_std + 1e-8)
        self.y_normalized = (y - self.y_mean) / (self.y_std + 1e-8)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_normalized[idx], dtype=torch.float32), 
            torch.tensor(self.y_normalized[idx], dtype=torch.float32)
        )


class HSIDataset_cube_1(Dataset):
    """高光谱立方体数据集类，用于加载和预处理高光谱数据
    
    Args:
        Dataset: PyTorch数据集基类
        img_folder (str): 图像文件夹路径
        label_path (str): 标签文件路径
        task_names (list): 任务名称列表
        indices (list, optional): 选择的样本索引列表. Defaults to None.
        device (str, optional): 设备类型（如'cpu'或'cuda'）. Defaults to None.
    """
    def __init__(self, img_folder, label_path, task_names, indices=None, device=None):
        self.img_folder = img_folder
        self.label_df = pd.read_excel(label_path)
        
        if indices is not None:
            self.label_df = self.label_df.iloc[indices]
            
        self.task_names = task_names
        self.device = device
        
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        img_name = self.label_df.iloc[idx]['name']
        
        # 直接在img_folder下查找文件
        img_path = os.path.join(self.img_folder, f'{img_name}.hdr')
        
        try:
            # 加载高光谱图像
            img = spectral.open_image(img_path).load()
            img = np.array(img)  # [H, W, C]
            
            # 首先裁剪或填充到64x64
            H, W, C = img.shape
            new_img = np.zeros((64, 64, C), dtype=np.float32)
            
            if H > 64 or W > 64:
                # 进行中心裁剪
                start_h = max(0, (H - 64) // 2)
                start_w = max(0, (W - 64) // 2)
                img_cropped = img[start_h:min(start_h+64, H), start_w:min(start_w+64, W), :]
                h, w = img_cropped.shape[:2]
                new_img[:h, :w, :] = img_cropped
            else:
                # 居中填充
                start_h = (64 - H) // 2
                start_w = (64 - W) // 2
                new_img[start_h:start_h+H, start_w:start_w+W, :] = img
            
            img = new_img
            
        except Exception as e:
            print(f"加载图像出错 {img_name}: {e}")
            # 创建空图像
            img = np.zeros((64, 64, 100), dtype=np.float32)
        
        # 重要：转置通道维度到第一位，符合PyTorch的NCHW格式
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()  # [C, H, W]
        
        if self.device:
            img_tensor = img_tensor.to(self.device)
        
        # 提取目标变量
        try:
            targets = self.label_df.iloc[idx][self.task_names].values
            targets = np.array(targets, dtype=np.float32)
        except Exception as e:
            print(f"处理目标变量出错: {e}")
            targets = np.zeros(len(self.task_names), dtype=np.float32)
        
        targets = torch.tensor(targets, dtype=torch.float32)
        if self.device:
            targets = targets.to(self.device)
        
        return img_tensor, targets
    
class HSIDataset_cube(Dataset):
    def __init__(self, img_folder, label_path, task_namewobus, indices=None, target_size=64, zscore_norm=True, device=None):
        self.img_folder = img_folder
        self.task_names = task_namewobus
        self.samples = pd.read_excel(label_path)
        self.target_size = target_size
        self.zscore_norm = zscore_norm
        self.device = device

        if indices is not None:
            self.samples = self.samples.iloc[indices]

        if self.zscore_norm:
            self.means = self.samples[self.task_names].mean()
            self.stds = self.samples[self.task_names].std()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples.iloc[idx]
        img_name = sample_info['name']
        img_path = os.path.join(self.img_folder, f'{img_name}.hdr')
        
        try:
            # Load and process image on CPU first
            img = spectral.open_image(img_path).load()
            img = np.array(img)  # [H, W, C]
            
            # Resize to 16x8 (HxW) to match model expectations
            H, W, C = img.shape
            target_h, target_w = 16, 8
            new_img = np.zeros((target_h, target_w, C), dtype=np.float32)
            
            if H > target_h or W > target_w:
                # Center crop
                start_h = max(0, (H - target_h) // 2)
                start_w = max(0, (W - target_w) // 2)
                img_cropped = img[start_h:start_h+target_h, 
                                start_w:start_w+target_w, :]
                new_img = img_cropped
            else:
                # Center pad
                start_h = (target_h - H) // 2
                start_w = (target_w - W) // 2
                new_img[start_h:start_h+H, start_w:start_w+W, :] = img
            
            # Convert to tensor and move to device
            img = np.transpose(new_img, (2, 0, 1))  # [C, H, W]
            img = torch.tensor(img, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            img = torch.zeros((278, 16, 8), dtype=torch.float32)
        
        # Process targets
        target = torch.tensor(sample_info[self.task_names].astype(float).values, 
                            dtype=torch.float32)
        
        if self.zscore_norm:
            target = (target - torch.tensor(self.means.values, dtype=torch.float32)) / \
                    torch.tensor(self.stds.values, dtype=torch.float32)
        
        # Move to device if specified
        if self.device:
            img = img.to(self.device)
            target = target.to(self.device)
            
        return img, target
