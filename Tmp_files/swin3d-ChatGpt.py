import torch
import torch.nn as nn
from einops import rearrange

from training.registry import DETECTOR
from training.models.backbone.swin_transformer_3d import SwinTransformer3D  # 确保模块路径正确

@DETECTOR.register_module(module_name='swin3d')
class Swin3DDeepfakeDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = SwinTransformer3D(
            pretrained=None,
            pretrained2d=False,
            patch_size=(2, 4, 4),
            window_size=(8, 7, 7),
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            in_channels=3,
            num_classes=None
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(768, 2)
        )

        if config.get("pretrained_weights"):
            checkpoint = torch.load(config.pretrained_weights, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.backbone.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.backbone.load_state_dict(checkpoint, strict=False)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, batch):
        x = batch['video']  # B x 3 x T x H x W
        y = batch['label']  # B

        feat = self.backbone(x)           # B x C x t x h x w
        logits = self.classifier(feat)    # B x 2

        if self.training:
            loss = self.loss_func(logits, y)
            return {"loss": loss}
        else:
            return {"logits": logits, "probs": torch.softmax(logits, dim=1)}
