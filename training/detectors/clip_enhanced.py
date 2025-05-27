import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
from typing import Optional, Tuple, Dict, Any
from metrics.registry import DETECTOR
from dataclasses import dataclass
from .base_detector import AbstractDetector
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train
import os

@dataclass
class Batch:
    images: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]
    identity: Optional[torch.Tensor]
    source: Optional[torch.Tensor]
    idx: Optional[torch.Tensor]
    paths: Optional[list[str]]

    def __getitem__(self, key):
        return getattr(self, key)

    @staticmethod
    def from_dict(batch: dict):
        return Batch(
            images=batch.get("image"),
            labels=batch.get("label"),
            identity=batch.get("identity"),
            source=batch.get("source"),
            idx=batch.get("idx"),
            paths=batch.get("path"),
        )

class LayerNormWithTuning(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.base_layer = nn.LayerNorm(hidden_size)
        self.ln_tuning_layers = nn.ModuleDict({
            'default': nn.LayerNorm(hidden_size)
        })

    def forward(self, x):
        #return self.base_layer(x) + self.ln_tuning_layers['default'](x)
        x = self.base_layer(x)
        return self.ln_tuning_layers['default'](x)

@DETECTOR.register_module(module_name='clip_enhanced')
class CLIPEnhanced(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize feature extractor (CLIP)
        self.clip_path = "../deepfake-detection/weights/clip-vit-large-patch14"
        if not os.path.exists(self.clip_path):
            raise ValueError(f"本地模型文件不存在: {self.clip_path}，请确保模型文件已下载到正确位置")
        
        self.feature_extractor = CLIPVisionModel.from_pretrained(self.clip_path)
        
        # Replace layer norm layers with LayerNormWithTuning
        self.feature_extractor.vision_model.pre_layrnorm = LayerNormWithTuning(self.feature_extractor.config.hidden_size)
        self.feature_extractor.vision_model.post_layernorm = LayerNormWithTuning(self.feature_extractor.config.hidden_size)
        
        for layer in self.feature_extractor.vision_model.encoder.layers:
            layer.layer_norm1 = LayerNormWithTuning(self.feature_extractor.config.hidden_size)
            layer.layer_norm2 = LayerNormWithTuning(self.feature_extractor.config.hidden_size)
        
        # Initialize head (classifier)
        features_dim = self.feature_extractor.config.hidden_size
        self.model = nn.Module()
        self.model.linear = nn.Linear(features_dim, 2, bias=True)
        
        # Initialize loss function
        self.loss_func = self.build_loss(config)
    
    def build_backbone(self, config):
        """Build the backbone network"""
        return self.feature_extractor
    
    def build_loss(self, config):
        """Build the loss function"""
        loss_name = config.get('loss_func', 'CrossEntropyLoss')
        loss_class = LOSSFUNC[loss_name]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        """Extract features from the input data"""
        x = data_dict['image']
        features = self.feature_extractor(x).last_hidden_state[:, 0, :]  # [B, 768]
        #features = self.feature_extractor(x).pooler_output  # [B, 768]
        return features
    
    def classifier(self, features: torch.tensor) -> torch.tensor:
        """Classify the features"""
        return self.model.linear(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """Compute the losses"""
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        return {'overall': loss}
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        """Compute training metrics"""
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        """Forward pass through the model"""
        features = self.features(data_dict)
        features = F.normalize(features, dim=1)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        
        pred_dict = {
            'cls': pred,
            'prob': prob
        }
        return pred_dict
    
    def get_preprocessing(self):
        """Get preprocessing function for inference"""
        processor = CLIPImageProcessor.from_pretrained(self.clip_path)
        def preprocess(image):
            return processor(images=image, return_tensors="pt")["pixel_values"][0]
        return preprocess 