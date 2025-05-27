from efficientnet_pytorch import EfficientNet
from transformers import TimesformerModel
import torch
import torch.nn as nn
from ..registry import DETECTOR
from ..builder import AbstractDetector, LOSSFUNC
from ..losses.metrics import calculate_metrics_for_train

@DETECTOR.register_module(module_name='timesformer')
class TimeSformerDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.efficientnet = self.build_efficientnet(config)
        self.backbone = self.build_backbone(config)
        self.fc = nn.Linear(1280 + 768, 1)
        self.loss_func = self.build_loss(config)

    def build_efficientnet(self, config):
        efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        efficientnet._fc = nn.Identity()  # remove classification head to get 1280-dim features
        return efficientnet

    def build_backbone(self, config):
        return TimesformerModel.from_pretrained(config["timesformer_ckpt"])

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    def features(self, data_dict: dict) -> torch.tensor:
        video_faces = data_dict['image']  # shape: (B, T, 3, 224, 224)
        b, t, c, h, w = video_faces.shape
        video_faces = video_faces.view(-1, c, h, w)  # (B*T, 3, 224, 224)
        eff_features = self.efficientnet(video_faces)  # (B*T, 1280)
        eff_features = eff_features.view(b, t, -1)  # (B, T, 1280)
        eff_pooled = eff_features.mean(dim=1)  # (B, 1280)

        ts_output = self.backbone(data_dict['image'], output_hidden_states=True)
        ts_feat = ts_output.pooler_output  # (B, 768) — using pooler_output instead of CLS token

        combined = torch.cat((eff_pooled, ts_feat), dim=1)  # (B, 2048)
        return combined

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.fc(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        return {'overall': loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.sigmoid(pred).squeeze(1)
        return {'cls': pred, 'prob': prob, 'feat': features}
