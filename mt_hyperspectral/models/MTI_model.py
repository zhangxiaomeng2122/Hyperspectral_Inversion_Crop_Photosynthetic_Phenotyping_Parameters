import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

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