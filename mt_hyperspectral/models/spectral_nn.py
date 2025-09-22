import torch
import torch.nn as nn

class AdvancedSpectralNN(nn.Module):
    """改进的卷积神经网络模型，用于高光谱数据分析"""
    
    def __init__(self, input_size=277, output_size=15):
        super(AdvancedSpectralNN, self).__init__()
        
        # 特征提取模块 - 多尺度卷积
        self.conv_small = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
        )
        
        self.conv_medium = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
        )
        
        self.conv_large = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
        )
        
        # 共享特征处理
        self.shared_conv = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2)
        )
        
        # 计算卷积后的特征大小
        conv_output_size = (input_size // 4) * 32
        
        # 多个独立的回归器，每个用于一个目标变量
        self.regressors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(conv_output_size, 128),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            ) for _ in range(output_size)
        ])
        
        # 注意力机制
        self.key_layer = nn.Linear(conv_output_size, 64)
        self.query_layer = nn.Linear(conv_output_size, 64)
        self.value_layer = nn.Linear(conv_output_size, conv_output_size)
        
        # 特征增强层 - 为每个目标添加特定的处理
        self.feature_enhancers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(conv_output_size, conv_output_size),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(conv_output_size),
            ) for _ in range(output_size)
        ])
    
    def forward(self, x):
        # 添加通道维度 [batch, features] -> [batch, 1, features]
        x_unsqueezed = x.unsqueeze(1)
        
        # 多尺度特征提取
        feat_small = self.conv_small(x_unsqueezed)
        feat_medium = self.conv_medium(x_unsqueezed)
        feat_large = self.conv_large(x_unsqueezed)
        
        # 合并多尺度特征
        x_concat = torch.cat([feat_small, feat_medium, feat_large], dim=1)
        
        # 共享特征处理
        x_shared = self.shared_conv(x_concat)
        x_flat = x_shared.view(x_shared.size(0), -1)
        
        # 改进的自注意力机制
        keys = self.key_layer(x_flat)
        queries = self.query_layer(x_flat)
        values = self.value_layer(x_flat)
        
        # 计算注意力分数 (批量内自注意力)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (64 ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 应用注意力
        attended_features = torch.matmul(attention_weights, values)
        
        # 通过每个独立的回归器，使用增强的特征
        outputs = []
        for i, (enhancer, regressor) in enumerate(zip(self.feature_enhancers, self.regressors)):
            # 特征增强 - 针对每个目标变量单独处理
            enhanced_features = enhancer(x_flat + attended_features)
            out = regressor(enhanced_features)
            outputs.append(out)
        
        # 合并结果 [batch, output_size]
        output_tensor = torch.cat(outputs, dim=1)
        
        return output_tensor

class GSModel(nn.Module):
    """专门为Gs（气孔导度）设计的自定义模型"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(GSModel, self).__init__()
        
        # 保存模型参数
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # 创建网络层
        layers = []
        
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(0.5))
        
        # 隐藏层
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.Dropout(0.3))
        
        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        # 创建顺序模型
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
    def get_architecture_info(self):
        """返回模型架构信息，用于保存"""
        return {
            'type': 'GSModel',
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims
        }
