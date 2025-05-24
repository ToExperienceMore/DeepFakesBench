import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
from typing import Optional, Tuple
from metrics.registry import DETECTOR

@DETECTOR.register_module(module_name='clip_enhanced')
class CLIPEnhanced(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load CLIP vision model
        self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        
        # Freeze backbone if specified
        if config.get('freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # PEFT: Layer Normalization tuning
        if config.get('use_peft', False):
            self._setup_peft()
        
        # Classifier
        self.classifier = nn.Linear(
            config['classifier']['in_features'],
            config['classifier']['out_features'],
            bias=config['classifier'].get('bias', True)
        )
        
        # L2 normalization
        self.use_l2_norm = config.get('use_l2_norm', False)
        
        # Metric learning parameters
        self.use_metric_learning = config.get('use_metric_learning', False)
        if self.use_metric_learning:
            self.metric_margin = config.get('metric_margin', 0.3)
        
        # Contrastive learning parameters
        self.use_contrastive_loss = config.get('use_contrastive_loss', False)
        if self.use_contrastive_loss:
            self.temperature = config.get('temperature', 0.07)
    
    def _setup_peft(self):
        """Setup Parameter Efficient Fine-Tuning"""
        if self.config.get('peft_type') == 'ln_tuning':
            # Only train LayerNorm parameters
            for name, param in self.backbone.named_parameters():
                if 'layer_norm' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Get features from backbone
        features = self.backbone(x).last_hidden_state[:, 0, :]  # [B, 768]
        
        # Apply L2 normalization if enabled
        if self.use_l2_norm:
            features = F.normalize(features, p=2, dim=1)
        
        # Get classification logits
        logits = self.classifier(features)
        
        # Return features for contrastive loss if enabled
        if self.use_contrastive_loss:
            return logits, features
        return logits, None
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                    features: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Cross entropy loss
        ce_loss = F.cross_entropy(logits, targets)
        
        if not (self.use_contrastive_loss and features is not None):
            return ce_loss
        
        # Contrastive loss
        batch_size = features.size(0)
        labels = targets.unsqueeze(0).eq(targets.unsqueeze(1)).float()
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.t()) / self.temperature
        
        # Remove diagonal
        mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        similarity = similarity[~mask].view(batch_size, -1)
        labels = labels[~mask].view(batch_size, -1)
        
        # Compute contrastive loss
        pos_sim = similarity[labels.bool()].view(batch_size, -1)
        neg_sim = similarity[~labels.bool()].view(batch_size, -1)
        
        # Compute metric learning loss if enabled
        if self.use_metric_learning:
            pos_loss = torch.clamp(self.metric_margin - pos_sim, min=0).mean()
            neg_loss = torch.clamp(neg_sim, min=0).mean()
            metric_loss = pos_loss + neg_loss
        else:
            metric_loss = 0
        
        # Combine losses
        contrastive_loss = -torch.log(
            torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim).sum(dim=1, keepdim=True))
        ).mean()
        
        # Weighted combination
        ce_weight = self.config['loss'].get('ce_weight', 0.7)
        contrastive_weight = self.config['loss'].get('contrastive_weight', 0.3)
        
        total_loss = (ce_weight * ce_loss + 
                     contrastive_weight * (contrastive_loss + metric_loss))
        
        return total_loss 