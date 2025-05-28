import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
from typing import Optional, Tuple, Dict, Any
from metrics.registry import DETECTOR
from dataclasses import dataclass
from .clip_enhanced import CLIPEnhanced
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train
import os
from peft import get_peft_model
from peft import LNTuningConfig
from einops import rearrange, repeat

class SpatialTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, dim)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

@DETECTOR.register_module(module_name='clip_spatial')
class CLIPSpatialDetector(CLIPEnhanced):
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize Spatial Transformer
        self.spatial_transformer = SpatialTransformer(
            dim=self.feature_extractor.config.hidden_size,
            num_heads=config['video_processing']['spatial_transformer']['num_heads'],
            num_layers=config['video_processing']['spatial_transformer']['num_layers'],
            dim_feedforward=config['video_processing']['spatial_transformer']['dim_feedforward'],
            dropout=config['video_processing']['spatial_transformer']['dropout']
        )
        
    def features(self, data_dict: dict) -> torch.tensor:
        # Get video frames
        x = data_dict['image']  # shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, c, h, w = x.shape
        
        # Process each frame through CLIP
        frame_features = []
        for i in range(num_frames):
            frame = x[:, i]  # Get single frame
            frame_feat = super().features({'image': frame})
            frame_features.append(frame_feat)
        frame_features = torch.stack(frame_features, dim=1)  # (batch_size, num_frames, dim)
        
        # Process through Spatial Transformer
        video_features = self.spatial_transformer(frame_features)
        
        # Global average pooling over temporal dimension
        video_features = video_features.mean(dim=1)
        
        return video_features
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        
        pred_dict = {
            'cls': pred,
            'prob': prob,
            'feat': features
        }
        return pred_dict
    
    def get_preprocessing(self):
        processor = CLIPImageProcessor.from_pretrained(self.clip_path)
        def preprocess(image):
            return processor(images=image, return_tensors="pt")["pixel_values"][0]
        return preprocess 