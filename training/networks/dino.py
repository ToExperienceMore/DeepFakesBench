import torch
import torch.nn as nn
from transformers import AutoModel
from metrics.registry import BACKBONE

@BACKBONE.register_module(module_name="dinov2")
class DINOv2(nn.Module):
    def __init__(self, dino_config):
        """
        DINOv2 ViT-L/14 model for image deepfake detection
        Args:
            dino_config (dict): Configuration dictionary containing:
                - pretrained (bool): Whether to use pretrained weights
                - num_classes (int): Number of output classes
                - dropout (float, optional): Dropout rate for classifier
        """
        super().__init__()
        
        self.num_classes = dino_config.get("num_classes", 2)
        pretrained_path = dino_config.get("pretrained", None)
        dropout = dino_config.get("dropout", 0.1)
        
        # Load pretrained DINOv2 ViT-L/14
        #self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=False)
        if pretrained_path:
            state_dict = torch.load(pretrained_path)
            self.model.load_state_dict(state_dict)
        
        # Freeze DINOv2 parameters if needed
        if pretrained_path:
            # 1. 冻结所有参数
            for param in self.model.parameters():
                param.requires_grad = False

            # 2. 解冻最后8层 Transformer blocks
            for i in range(16, 24):  # ViT-L has 24 blocks
                for param in self.model.blocks[i].parameters():
                    param.requires_grad = True

            # 可选：也可解冻 cls_token 或 pos_embed（可试验效果）
            self.model.cls_token.requires_grad = True
            self.model.pos_embed.requires_grad = True
        
        # Feature dimension from DINOv2-L
        self.feature_dim = 1024
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Get image features
        features = self.model(x).last_hidden_state[:, 0]  # [B, D]
        
        # Classification
        logits = self.classifier(features)
        return logits
