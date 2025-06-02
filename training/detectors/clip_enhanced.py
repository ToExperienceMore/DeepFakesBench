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
from peft import get_peft_model
from peft import LNTuningConfig

@DETECTOR.register_module(module_name='clip_enhanced')
class CLIPEnhanced(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize feature extractor (CLIP)
        self.clip_path = config.get('clip_path', "weights/clip-vit-base-patch16")
        print(f"clip_path: {self.clip_path}")

        if not os.path.exists(self.clip_path):
            raise ValueError(f"本地模型文件不存在: {self.clip_path}，请确保模型文件已下载到正确位置")
        
        self.feature_extractor = CLIPVisionModel.from_pretrained(self.clip_path)

        # 首先冻结所有参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        target_modules_list=['pre_layrnorm', 'layer_norm1', 'layer_norm2', 'post_layernorm', 'layernorm']
        peft_config = LNTuningConfig(target_modules=target_modules_list)

        backbone = self.feature_extractor
        training_parameters = {name for name, param in backbone.named_parameters() if param.requires_grad}

        self.feature_extractor = get_peft_model(self.feature_extractor, peft_config)

        for name, param in backbone.named_parameters():
            if name in training_parameters:
                param.requires_grad = True
        
        # Initialize head (classifier)
        features_dim = self.feature_extractor.config.hidden_size
        print(f"features_dim: {features_dim}")
        self.model = nn.Module()
        #hidden_dim = features_dim // 2  # Using half of the input dimension as hidden size
        print(f"mlp_layer: {self.config.get('mlp_layer', 1)}")
        if self.config.get('mlp_layer', 1) == 2:
            hidden_dim = 512
            self.model.mlp = nn.Sequential(
                nn.Linear(features_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )
        else:
            self.model.linear = nn.Linear(features_dim, 2, bias=True)
        
        # Initialize loss function
        self.loss_func = self.build_loss(config)
        
        self.print_trainable_parameters()
    
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
        #features = self.feature_extractor(x).last_hidden_state[:, 0, :]  # [B, 768]
        features = self.feature_extractor(x).pooler_output  # [B, 768]
        return features
    
    def classifier(self, features: torch.tensor) -> torch.tensor:
        """Classify the features"""

        if self.config.get('mlp_layer', 1) == 2:
            return self.model.mlp(features)
        else:
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
        #pred = pred_dict['prob']  # 使用已经计算好的概率值
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
            'prob': prob,
            'feat': features
        }
        return pred_dict
    
    def get_preprocessing(self):
        """Get preprocessing function for inference"""
        processor = CLIPImageProcessor.from_pretrained(self.clip_path)
        def preprocess(image):
            return processor(images=image, return_tensors="pt")["pixel_values"][0]
        return preprocess 

    def print_trainable_parameters(self):
        print("\n🔥 Trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name} shape = {tuple(param.shape)}")

        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"Total parameters: {all_params}, trainable: {trainable_params}, %: {trainable_params / all_params * 100:.4f}"
        )