import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import timm

class EfficientViT(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientViT, self).__init__()
        
        # EfficientNet-B7 (原仓库使用B7而不是B0)
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7')
        self.efficient_features = self.efficient_net._fc.in_features
        
        # ViT
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit_features = self.vit.head.in_features
        self.vit.head = nn.Identity()
        
        # 按照原仓库的组合层结构
        self.classifier = nn.Sequential(
            nn.Linear(self.efficient_features + self.vit_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # EfficientNet forward
        efficient_features = self.efficient_net.extract_features(x)
        efficient_features = self.efficient_net._avg_pooling(efficient_features)
        efficient_features = efficient_features.flatten(start_dim=1)
        
        # ViT forward
        vit_features = self.vit(x)
        
        # 特征融合
        combined_features = torch.cat([efficient_features, vit_features], dim=1)
        output = self.classifier(combined_features)
        
        return output 