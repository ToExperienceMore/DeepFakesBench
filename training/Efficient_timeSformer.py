import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, T, P):
        # x: (B, T*P, D) → (B*P, T, D)
        B, TP, D = x.shape
        x = x.view(B, T, P, D).permute(0, 2, 1, 3).contiguous()  # (B, P, T, D)
        x = x.view(B * P, T, D)
        out, _ = self.attn(x, x, x)
        out = out.view(B, P, T, D).permute(0, 2, 1, 3).contiguous()  # (B, T, P, D)
        return out.view(B, T * P, D)


class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, T, P):
        # x: (B, T*P, D) → (B*T, P, D)
        B, TP, D = x.shape
        x = x.view(B, T, P, D).view(B * T, P, D)
        out, _ = self.attn(x, x, x)
        out = out.view(B, T, P, D).view(B, T * P, D)
        return out


class EffNet_TimeSformer_Decomposed(nn.Module):
    def __init__(self, num_frames=8, embed_dim=1280, num_heads=8, num_layers=6, num_classes=2):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.patch_per_frame = 49

        self.effnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.effnet._fc = nn.Identity()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_frames * self.patch_per_frame, embed_dim))

        self.temporal_attn = TemporalAttention(embed_dim, num_heads)
        self.spatial_attn = SpatialAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def extract_patch_tokens(self, x):
        feat = self.effnet.extract_features(x)
        B_T, C, H, W = feat.shape
        feat = feat.view(B_T, C, H * W).permute(0, 2, 1)  # (B*T, 49, 1280)
        return feat

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        patch_tokens = self.extract_patch_tokens(x)  # (B*T, 49, 1280)
        patch_tokens = patch_tokens.view(B, T * self.patch_per_frame, self.embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, patch_tokens], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)

        cls, tokens = x[:, :1, :], x[:, 1:, :]
        tokens = self.temporal_attn(tokens, T, self.patch_per_frame)
        tokens = self.spatial_attn(tokens, T, self.patch_per_frame)

        x = torch.cat([cls, tokens], dim=1)
        cls_token = self.norm(x[:, 0, :])
        return self.fc(cls_token)