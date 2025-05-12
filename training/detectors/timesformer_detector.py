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


@DETECTOR.register_module(module_name='timesformer')
class TimeSformerDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.efficientnet = self.build_efficientnet(config)
        self.backbone = self.build_backbone(config)
        self.fc = nn.Linear(1280 + 768, 2)
        self.loss_func = self.build_loss(config)

    def build_efficientnet(self, config):
        from efficientnet_pytorch import EfficientNet
        efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        efficientnet._fc = nn.Identity()  # remove classification head to get 1280-dim features
        return efficientnet

    def build_backbone(self, config):
        from transformers import TimesformerModel
        backbone = TimesformerModel.from_pretrained("/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/")
        return backbone

    def build_temporal_module(self, config):
        return nn.LSTM(input_size=2048, hidden_size=512, num_layers=3, batch_first=True)

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()

        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        #video_faces = data_dict['video_faces']
        video_faces = data_dict['image']
        b, t, c, h, w = video_faces.shape
        video_faces = video_faces.view(-1, c, h, w)
        #print("video_faces shape:", video_faces.shape)  # 打印维度
        efficientnet_features = self.efficientnet(video_faces)
        #print("efficientnet_features shape:", efficientnet_features.shape)  # 打印维度
        eff_features = efficientnet_features.view(b, t, -1)  # (B, T, 1280)
        eff_pooled = eff_features.mean(dim=1)  # (B, 1280)
        #print("eff_pooled shape:", eff_pooled.shape)  # 打印维度
        
        outputs = self.backbone(data_dict['image'], output_hidden_states=True)
        timesformer_features = outputs[0][:, 0]
        #print("timesformer_features shape:", timesformer_features.shape)  # 打印维度

        combined_features = torch.cat((eff_pooled, timesformer_features), dim=1)
        #print("combined_features shape:", combined_features.shape)  # 打印维度
        return combined_features

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
