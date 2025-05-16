"""
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the TimesformerDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{bertasius2021space,
  title={Is space-time attention all you need for video understanding?},
  author={Bertasius, Gedas and Wang, Heng and Torresani, Lorenzo},
  booktitle={ICML},
  volume={2},
  number={3},
  pages={4},
  year={2021}
}
"""

import logging
import torch
import torch.nn as nn
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)

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
        x = x.reshape(B, T, P, D).reshape(B * T, P, D)  # 使用reshape替代view
        out, _ = self.attn(x, x, x)
        out = out.reshape(B, T, P, D).reshape(B, T * P, D)  # 使用reshape替代view
        return out

@DETECTOR.register_module(module_name='timesformer')
class TimeSformerDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.efficientnet = self.build_efficientnet(config)
        self.num_frames = 8
        self.embed_dim = 1280
        self.patch_per_frame = 49
        self.num_heads = 8

        # 可分离的时空注意力模块
        self.temporal_attn = TemporalAttention(self.embed_dim, self.num_heads)
        self.spatial_attn = SpatialAttention(self.embed_dim, self.num_heads)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        # 分类头
        self.fc = nn.Linear(self.embed_dim, 2)
        self.loss_func = self.build_loss(config)

    def build_efficientnet(self, config):
        from efficientnet_pytorch import EfficientNet
        efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        efficientnet._avg_pooling = nn.Identity()
        efficientnet._fc = nn.Identity()
        return efficientnet

    def build_backbone(self, config):
        # 为了满足抽象类的要求，返回一个空模块
        return nn.Identity()

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        video_faces = data_dict['image']
        b, t, c, h, w = video_faces.shape
        #print("video_faces shape:", video_faces.shape)
        video_faces = video_faces.view(-1, c, h, w)
        #print("video_faces shape:", video_faces.shape)
        
        # 1. 通过EfficientNet提取空间特征
        efficientnet_features = self.efficientnet(video_faces)
        #print("efficientnet_features shape:", efficientnet_features.shape)
        eff_features = efficientnet_features.view(b, t, 1280, 7, 7)  # (B, T, 1280, 7, 7)
        #print("eff_features shape:", eff_features.shape)

        # Step 1: reshape 每帧 7x7 patch → 49 patch
        patch_tokens = eff_features.view(b, t, 1280, 49)        # (B, T, 1280, 49)
        #print("patch_tokens shape:", patch_tokens.shape)

        # Step 2: 转置 → patch 在时间维度之前
        patch_tokens = patch_tokens.permute(0, 1, 3, 2)         # (B, T, 49, 1280)
        #print("patch_tokens permuted shape:", patch_tokens.shape)

        # Step 3: 合并 T × 49 作为 token 序列
        patch_tokens = patch_tokens.reshape(b, t * 49, 1280)    # (B, 392, 1280)
        #print("patch_tokens reshaped shape:", patch_tokens.shape)

        # 4. 应用可分离的时空注意力
        # 首先应用时间注意力
        tokens = self.temporal_attn(patch_tokens, t, self.patch_per_frame)
        #print("after temporal attention shape:", tokens.shape)
        
        # 然后应用空间注意力
        tokens = self.spatial_attn(tokens, t, self.patch_per_frame)
        #print("after spatial attention shape:", tokens.shape)
        
        # 5. 取第一个token作为序列表示
        features = self.norm(tokens[:, 0, :])  # (B, 1280)
        #print("final features shape:", features.shape)
        
        return features

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.fc(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict
