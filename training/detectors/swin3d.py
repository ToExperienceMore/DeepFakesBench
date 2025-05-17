"""
# author: [Your Name]
# email: [Your Email]
# date: 2024-03-21
# description: Class for the Swin3DDeepfakeDetector

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
"""

import logging
import torch
import torch.nn as nn
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train
from models.swin_transformer_3d import SwinTransformer3D

from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='swin3d')
class Swin3DDeepfakeDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        model = SwinTransformer3D(
            pretrained=None,
            pretrained2d=False,
            patch_size=(2,4,4),  # T,H,W
            window_size=(8,7,7),
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            in_channels=3,
            num_classes=None  # 去掉默认分类头
        )
        return model

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        video_faces = data_dict['image']  # B x T x C x H x W
        b, t, c, h, w = video_faces.shape
        
        # 调整输入格式为 Swin3D 所需的格式
        video_faces = video_faces.permute(0, 2, 1, 3, 4)  # B x C x T x H x W
        
        # 通过 Swin3D 提取特征
        features = self.model(video_faces)  # B x C x T' x H' x W'
        
        # 全局平均池化
        features = torch.mean(features, dim=(2, 3, 4))  # B x C
        
        return features

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.model.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label'].long()
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


if __name__ == '__main__':
    model = Swin3DDeepfakeDetector()
    dummy_input = torch.randn(2, 3, 8, 224, 224)  # Batch = 2, T = 8
    out = model(dummy_input)
    print("Output:", out.shape)  # 应为 (2, 2)
