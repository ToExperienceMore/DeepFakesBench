"""
Video deepfake detection using DINOv2 and layout-aware transformer
"""

import logging
import torch
import torch.nn as nn
from detectors import DETECTOR
from metrics.base_metrics_class import calculate_metrics_for_train
from loss import LOSSFUNC
from .base_detector import AbstractDetector
from networks import BACKBONE

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='tall_dinov2')
class TALLVideoDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.layout_transformer = self.build_transformer(config)
        self.head = self.build_classifier(config)
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        """Build DINOv2 backbone"""
        # Get backbone class and config
        backbone_class = BACKBONE[config['backbone_name']]
        backbone_config = config['backbone_config']
        
        # Build backbone
        model = backbone_class(backbone_config)
        
        # Freeze DINOv2 parameters if specified
        if config['dinov2'].get('freeze_backbone', True):
            for param in model.parameters():
                param.requires_grad = False
                
        return model
    
    def build_transformer(self, config):
        """Build layout-aware transformer"""
        transformer_config = config['transformer']
        return LayoutAwareTransformer(
            input_dim=transformer_config['input_dim'],
            hidden_dim=transformer_config['hidden_dim'],
            num_layers=transformer_config['num_layers'],
            num_heads=transformer_config['num_heads'],
            dropout=transformer_config['dropout']
        )
    
    def build_classifier(self, config):
        """Build classification head"""
        classifier_config = config['classifier']
        transformer_config = config['transformer']
        return nn.Sequential(
            nn.Linear(transformer_config['input_dim'], classifier_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(classifier_config['dropout']),
            nn.Linear(classifier_config['hidden_dim'], classifier_config['num_classes'])
        )
    
    def build_loss(self, config):
        """Build loss function"""
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()
    
    def features(self, data_dict: dict) -> torch.tensor:
        """
        Extract features from video frames
        Args:
            data_dict: Dictionary containing video frames [B, T, C, H, W]
        Returns:
            features: Extracted features [B, D]
        """
        x = data_dict['image']  # [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # Reshape to process all frames
        x = x.view(B * T, C, H, W)
        
        # Extract features using DINOv2
        features = self.backbone.model(x).last_hidden_state[:, 0]  # [B*T, D]
        
        # Reshape back to video format
        features = features.view(B, T, -1)  # [B, T, D]
        
        # Process through layout transformer
        features = self.layout_transformer(features)  # [B, T, D]
        
        # Average pooling over temporal dimension
        features = features.mean(dim=1)  # [B, D]
        
        return features
    
    def classifier(self, features: torch.tensor) -> torch.tensor:
        """Classification head"""
        return self.head(features)
    
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


class LayoutAwareTransformer(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Position encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, input_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input features [B, T, D]
            mask: Attention mask [B, T]
        Returns:
            output: Processed features [B, T, D]
        """
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Transformer processing
        output = self.transformer(x, src_key_padding_mask=mask)
        
        # Output projection
        output = self.output_proj(output)
        
        return output