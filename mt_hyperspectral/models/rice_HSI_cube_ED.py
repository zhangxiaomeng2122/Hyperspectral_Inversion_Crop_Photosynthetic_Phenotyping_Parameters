import torch
import torch.nn as nn


# class ViTEncoderBlock(nn.Module):
#     def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop=0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
#         self.drop1 = nn.Dropout(drop)

#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Dropout(drop),
#             nn.Linear(int(dim * mlp_ratio), dim),
#             nn.Dropout(drop)
#         )

#     def forward(self, x):  # x: [B, N, C]
#         # Multi-head self-attention with residual
#         x_res = x
#         x = self.norm1(x)
#         attn_output, _ = self.attn(x, x, x)
#         x = x_res + self.drop1(attn_output)

#         # MLP with residual
#         x_res = x
#         x = self.norm2(x)
#         x = x_res + self.mlp(x)
#         return x
    

# class Decoder(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Decoder, self).__init__()
#         self.fc1 = nn.Linear(input_size, input_size)
#         self.fc2 = nn.Linear(input_size, input_size)
#         self.fc3 = nn.Linear(input_size, output_size)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # VIT
# class HSIModel(nn.Module):
#     def __init__(self, band_num=278, num_tasks=5, num_layers=3, max_h=64, max_w=64, num_heads=1, mlp_ratio=4.0, drop=0.0):
#         super().__init__()

#         self.max_h = max_h
#         self.max_w = max_w

#         # Learnable 2D positional embeddings
#         self.pos_embedding = nn.Parameter(torch.randn(1, max_h * max_w, band_num))

#         self.encoders = nn.Sequential(
#             *[ViTEncoderBlock(dim=band_num, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop) for _ in range(num_layers)]
#         )

#         self.decoder = Decoder(input_size=band_num, output_size=num_tasks)

#     def forward(self, x):  # x: [B, C, H, W]
#         B, C, H, W = x.shape

#         if H > self.max_h or W > self.max_w:
#             raise ValueError(f"Input image size {H}x{W} exceeds the max supported size of {self.max_h}x{self.max_w}")

#         pos_embed = self.pos_embedding[:, :H * W, :]  # [1, H*W, C]

#         x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
#         x = x.view(B, H * W, C)  # [B, N, C]
#         x = x + pos_embed

#         x = self.encoders(x)  # [B, N, C]
#         x = x.mean(dim=1)     # Global average pooling over tokens

#         # print(x.shape)
#         out = self.decoder(x)  # [B, num_tasks]
#         return out

class ViTEncoderBlock(nn.Module):
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

    def forward(self, x):  # x: [B, N, C]
        # Multi-head self-attention with residual
        x_res = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x_res + self.drop1(attn_output)

        # MLP with residual
        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)
        return x
    



# VIT
class HSIModel(nn.Module):
    def __init__(self, band_num=278, num_tasks=5, num_layers=3, max_h=64, max_w=64, num_heads=1, mlp_ratio=4.0, drop=0.0):
        super().__init__()

        self.max_h = max_h
        self.max_w = max_w

        # Initialize with maximum expected size (64x64=4096)
        self.pos_embedding = nn.Parameter(torch.randn(1, 4096, band_num))

        self.encoders = nn.Sequential(
            *[ViTEncoderBlock(dim=band_num, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop) for _ in range(num_layers)]
        )

        self.decoder = Decoder(input_size=band_num, output_size=num_tasks)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Dynamic positional embeddings based on actual input size
        pos_embed = self.pos_embedding[:, :H * W, :]  # [1, H*W, C]

        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = x.view(B, H * W, C)  # [B, N, C]
        x = x + pos_embed

        x = self.encoders(x)  # [B, N, C]
        x = x.mean(dim=1)     # Global average pooling over tokens

        # print(x.shape)
        out = self.decoder(x)  # [B, num_tasks]
        return out



class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
