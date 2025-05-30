"""
# author: [Your Name]
# email: [Your Email]
# date: 2024-03-21
# description: Class for the CLIPSTANDetector

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
@inproceedings{stan2024revisiting,
  title={Revisiting Temporal Modeling for CLIP-based Image-to-Video Knowledge Transferring},
  author={[Authors]},
  booktitle={[Conference]},
  year={2024}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPConfig
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPMLP
from typing import Optional, Tuple, Dict, Any
from metrics.registry import DETECTOR
from dataclasses import dataclass
from .base_detector import AbstractDetector
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train
import os
from peft import get_peft_model
from peft import LNTuningConfig
from einops import rearrange

@DETECTOR.register_module(module_name='clip_stan')
class CLIPSTANDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize feature extractor (CLIP)
        self.clip_path = config.get('clip_path', "weights/clip-vit-base-patch16")
        print(f"clip_path: {self.clip_path}")

        if not os.path.exists(self.clip_path):
            raise ValueError(f"Local model file does not exist: {self.clip_path}, please ensure the model file is downloaded to the correct location")
        
        self.feature_extractor = CLIPVisionModel.from_pretrained(self.clip_path)

        """
        # First freeze all parameters if specified
        if config.get('freeze_backbone', False):
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Apply PEFT for layer norm tuning if specified
        if config.get('use_peft', True):
            target_modules_list=['pre_layrnorm', 'layer_norm1', 'layer_norm2', 'post_layernorm', 'layernorm']
            peft_config = LNTuningConfig(target_modules=target_modules_list)

            backbone = self.feature_extractor
            training_parameters = {name for name, param in backbone.named_parameters() if param.requires_grad}

            self.feature_extractor = get_peft_model(self.feature_extractor, peft_config)

            for name, param in backbone.named_parameters():
                if name in training_parameters:
                    param.requires_grad = True
        """
        
        # Initialize STAN components from config
        self.depth = config.get('depth', 4)
        self.time_module = config.get('time_module', 'selfattn')
        self.cls_residue = config.get('cls_residue', False)
        self.gradient_checkpointing = config.get('gradient_checkpointing', True)
        
        # Calculate dimensions
        self.num_patches = (self.feature_extractor.vision_model.config.image_size // self.feature_extractor.vision_model.config.patch_size) ** 2
        self.embed_dim = self.feature_extractor.vision_model.config.hidden_size
        
        # Initialize STAN layers
        dpr = torch.linspace(0, 0.1, self.depth)
        self.STAN_S_layers = nn.ModuleList([
            CLIPLayer_Spatial(self.feature_extractor.vision_model.config, 8, dpr[i]) 
            for i in range(self.depth)
        ])
        
        if self.time_module == "selfattn":
            self.STAN_T_layers = nn.ModuleList([
                CLIPLayer_AttnTime(self.feature_extractor.vision_model.config, 8, dpr[i])
                for i in range(self.depth)
            ])
        elif self.time_module == "conv":
            self.STAN_T_layers = nn.ModuleList([
                CLIPLayer_ConvTime(self.feature_extractor.vision_model.config.hidden_size, 8, dpr[i])
                for i in range(self.depth)
            ])
            
        # Initialize embeddings
        self.time_embed = nn.Embedding(64, self.embed_dim)
        self.drop_after_pos = nn.Dropout(p=0)
        self.drop_after_time = nn.Dropout(p=0)

        self.patch_embeds = self.feature_extractor.vision_model.embeddings.patch_embedding
        self.pos_embed = self.feature_extractor.vision_model.embeddings.position_embedding
        self.class_embedding = self.feature_extractor.vision_model.embeddings.class_embedding

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize head (classifier) from config
        classifier_config = config.get('classifier', {})
        features_dim = classifier_config.get('in_features', self.embed_dim)
        print(f"features_dim: {features_dim}")
        self.model = nn.Module()
        self.model.linear = nn.Linear(
            features_dim, 
            classifier_config.get('out_features', 2),
            bias=classifier_config.get('bias', True)
        )
        
        # Initialize loss function
        self.loss_func = self.build_loss(config)
        
        # Apply freezing strategy
        self.frozen_layers = config.get('frozen_layers', True)
        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Freeze specific stages of the model based on the freezing strategy."""
        if self.frozen_layers:
            for name, param in self.named_parameters():
                # 1. 保留 STAN 相关参数
                if "STAN" in name:
                    param.requires_grad = True
                    continue
                    
                # 2. 冻结文本编码器
                elif 'text_backbone' in name:
                    param.requires_grad = False
                    
                # 3. 保留额外投影层
                elif 'extra_proj' in name:
                    param.requires_grad = True
                    continue
                    
                # 4. 保留平衡参数
                elif 'balance' in name:
                    param.requires_grad = True
                    continue
                    
                # 5. 处理视觉编码器层
                elif 'backbone.layers' in name:
                    layer_n = int(name.split('.')[2])
                    if layer_n >= 20:  # 只冻结前20层
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                    
                # 6. 确保分类器参数可训练
                elif 'model.linear' in name:
                    param.requires_grad = True
                    
                # 7. 冻结其他所有参数
                else:
                    param.requires_grad = False

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
        print("x.shape:", x.shape)
        print("x.ndim:", x.ndim)
        
        # Handle video input
        if x.ndim == 5:
            if x.shape[1] == 3:
                B, D, T, H, W = x.shape             
                x = x.permute(0, 2, 1, 3, 4)
            else:
                B, T, D, H, W = x.shape   
            x = x.reshape((-1,) + x.shape[2:])
        else:
            B, _, _, _ = x.shape
            T = 1
            
        self.T = T
        
        # Get embeddings
        #(B*T, num_patches + 1, embed_dim)
        #x.shape: torch.Size([256, 3, 224, 224])
        print("x.shape:", x.shape)
        embeddings = self.forward_embedding(x)
        print("embeddings.shape:", embeddings.shape)
        
        # Forward through STAN layers
        x = self.feature_extractor.vision_model.pre_layrnorm(embeddings)
        x2 = None
        
        for idx, encoder_layer in enumerate(self.feature_extractor.vision_model.encoder.layers):
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(encoder_layer, x, None, None)
            else: 
                layer_outputs = encoder_layer(x, attention_mask=None, causal_attention_mask=None)
            x = layer_outputs[0]

            if idx >= len(self.feature_extractor.vision_model.encoder.layers) - self.depth: 
                num_layer = idx + self.depth - len(self.feature_extractor.vision_model.encoder.layers)
                x2 = self.forward_time_module(x, x2, num_layer, self.T)
        
        # x: (B*T, num_patches + 1, embed_dim)
        # x2: (B*T, num_patches + 1, embed_dim)
        print("x.shape:", x.shape)
        print("x2.shape:", x2.shape)
        #x.shape: torch.Size([256, 197, 768])
        #x2.shape: torch.Size([32, 1569, 768]) #TODO: x2.shape is wroing

        """
        # Get final features
        cls_token = x[:, 0].reshape(B, T, -1).mean(dim=1) + x2[:, 0]
        print("cls_token:", cls_token.shape)
        features = self.feature_extractor.vision_model.post_layernorm(cls_token)
        # TODO, encoder_states is missing
        """

        spatial_cls = x[:, 0]  # (B*T, embed_dim)
        
        # 4.2 时间CLS token
        temporal_cls = x2[:, 0]  # (B*T, embed_dim)
        print("temporal.shape:", temporal_cls.shape)
        temporal_cls = temporal_cls.repeat(1, self.T)  # (B*T, T, embed_dim)
        temporal_cls = temporal_cls.view(x2.size(0) * self.T, -1)  # (B*T, embed_dim)
        
        # 4.3 融合CLS token
        cls_token = spatial_cls + temporal_cls  # (B*T, embed_dim)
        
        # 4.4 后处理
        #cls_token = self.post_layernorm(cls_token)  # (B*T, embed_dim)
        cls_token = self.feature_extractor.vision_model.post_layernorm(cls_token)
        print("cls_token.shape:", cls_token.shape)

        cls_token = rearrange(cls_token, '(b t) d -> b t d', t=self.T)
        cls_token = cls_token.mean(1) #b, d
        print("### cls_token.shape:", cls_token.shape)
        
        return cls_token
    
    def forward_embedding(self, x):
        """Forward pass through embedding layers"""
        batch_size = x.shape[0] #should be B*T
        print("batch_size:", batch_size)
        patch_embeds = self.patch_embeds(x)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        #class_embeds = self.cls_token.expand(batch_size, 1, -1)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        
        position_ids = torch.arange(self.num_patches + 1, dtype=torch.long, device=x.device).expand((1, -1))
        embeddings = embeddings + self.pos_embed(position_ids)

        return embeddings
    
    def forward_time_module(self, x1, x2, num_layer, T):
        """Forward pass through time module"""
        B = x1.size(0) // T
        # 输入: (B*T, num_patches + 1, embed_dim)
        # 输出: (B, T, num_patches + 1, embed_dim)
        x1 = x1.reshape(B, T, x1.size(1), x1.size(2))

        if x2 is not None:
            cls_token_ori = x1[:, :, 0, :]
            cls_token = cls_token_ori.mean(dim=1).unsqueeze(1)
            x1 = x1[:, :, 1:, :]
            # 重组维度
            # 输入: (B, T, num_patches, embed_dim)
            # 输出: (B, T*num_patches, embed_dim)
            x1 = x1.reshape(x1.size(0), -1, x1.size(-1))
            print("(B, T*num_patches, embed_dim):", x1.shape)
            x1 = torch.cat((cls_token, x1), dim=1)

            if not self.cls_residue:
                x = x2 + x1
            else:
                if self.training:
                    cls_token1 = cls_token_ori[:,0::2,:].mean(dim=1).unsqueeze(1)
                else:
                    cls_token1 = cls_token_ori.mean(dim=1).unsqueeze(1)
                
                x1 = torch.cat((cls_token1.repeat(1,1,1), x1[:, 1:, :]), dim=1)
                x = x2 + x1
        else:
            x = x1
        
        if num_layer == 0:
            x = self.input_ini(x)

        if self.gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.STAN_T_layers[num_layer], x)
            x = torch.utils.checkpoint.checkpoint(self.STAN_S_layers[num_layer], x, None, None)
        else: 
            x = self.STAN_T_layers[num_layer](x)
            x = self.STAN_S_layers[num_layer](x, None, None)
        return x
    
    def input_ini(self, x):
        """Initialize input for first layer"""
        cls_old = x[:, :, 0, :].mean(dim=1).unsqueeze(1)
        x = x[:,:,1:,:]
        B,T,L,D = x.size()
        print(f"input_ini - x shape before view: {x.shape}")
        
        # 修改维度变换顺序
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, L, T, D]
        x = x.reshape(-1, T, D)    # [B*L, T, D]
        print(f"input_ini - x shape after first reshape: {x.shape}")
        
        cls_tokens = self.cls_token.expand(x.size(0), 1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        print(f"input_ini - x shape after cat: {x.shape}")
        
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pos_embed = self.pos_embed(position_ids)
        x = x + pos_embed
        x = self.drop_after_pos(x)
        
        cls = x[:B, 0, :].unsqueeze(1)
        x = x[:, 1:, :]
        print(f"input_ini - x shape before final reshape: {x.shape}")
        
        # 修改维度变换顺序
        x = x.reshape(B, L, T, D)  # [B, L, T, D]
        print(f"input_ini - x shape after final reshape: {x.shape}")
        
        position_ids = torch.arange(T, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, -1)
        time_embed = self.time_embed(position_ids)  # [B, T, D]
        time_embed = time_embed.unsqueeze(2)  # [B, T, 1, D]
        time_embed = time_embed.expand(-1, -1, L, -1)  # [B, T, L, D]
        time_embed = time_embed.permute(0, 2, 1, 3)  # [B, L, T, D]
        x = x + time_embed
        x = self.drop_after_time(x)
        x = x.reshape(B, -1, D)
        print(f"input_ini - x shape at end: {x.shape}")
        
        cls = (cls_old + cls) / 2
        x = torch.cat((cls, x), dim=1)
        return x
    
    def classifier(self, features: torch.tensor) -> torch.tensor:
        """Classify the features"""
        return self.model.linear(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """Compute the losses"""
        label = data_dict['label']
        pred = pred_dict['cls']
        print("label.shape:", label.shape)
        print("pred.shape:", pred.shape)
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
        print("features.shape:", features.shape)
        pred = self.classifier(features)
        print("pred.shape:", pred.shape)
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

    def inflate_weight(self, weight_2d, time_dim, center=True):
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d

class CLIPLayer_Spatial(nn.Module):
    def __init__(self, config, T, layer_num=0.1):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = nn.Dropout(layer_num) if layer_num > 0. else nn.Identity()
        self.t = T

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None):
        residual = hidden_states

        init_cls_token = hidden_states[:, :1, :]
        query_s = hidden_states[:, 1:, :]

        b, pt, m = query_s.size()
        p, t = pt // self.t, self.t
        cls_token = init_cls_token.unsqueeze(1).repeat(1, t, 1, 1).reshape(b * t, 1, m)
        query_s = query_s.reshape(-1, p, m) # not the same
        hidden_states = torch.cat((cls_token, query_s), 1)

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
        )

        res_spatial = self.dropout_layer(
            self.proj_drop(hidden_states.contiguous()))
        cls_token = res_spatial[:, :1, :].reshape(b, self.t, 1, m)
        cls_token = torch.mean(cls_token, 1)
        res_spatial = res_spatial[:, 1:, :].reshape(b, p * self.t, m) #not the same
        hidden_states = torch.cat((cls_token, res_spatial), 1)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class CLIPLayer_AttnTime(nn.Module):
    def __init__(self, config, T, layer_num=0.1):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)

        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = nn.Dropout(layer_num) if layer_num > 0. else nn.Identity()
        self.temporal_fc = nn.Linear(self.embed_dim, self.embed_dim)
        nn.init.constant_(self.temporal_fc.weight, 0)
        nn.init.constant_(self.temporal_fc.bias, 0)
        self.t = T

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None):
        residual = hidden_states[:, 1:, :]

        init_cls_token = hidden_states[:, :1, :]
        query_t = hidden_states[:, 1:, :]
        b, pt, m = query_t.size()
        p, t = pt // self.t, self.t
        hidden_states = query_t.reshape(b * p, t, m)

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
        )

        res_temporal = self.dropout_layer(
            self.proj_drop(hidden_states.contiguous()))
        res_temporal = self.temporal_fc(res_temporal)
        hidden_states = res_temporal.reshape(b, p * self.t, m)
        hidden_states = residual + hidden_states
        hidden_states = torch.cat((init_cls_token, hidden_states), 1)

        return hidden_states

class CLIPLayer_ConvTime(nn.Module):
    def __init__(self, embed_dim, T, layer_num=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = nn.Dropout(layer_num) if layer_num > 0. else nn.Identity()
        self.t = T

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None):
        residual = hidden_states[:, 1:, :]
        init_cls_token = hidden_states[:, :1, :]
        query_t = hidden_states[:, 1:, :]
        
        b, pt, m = query_t.size()
        p, t = pt // self.t, self.t
        hidden_states = query_t.reshape(b * p, t, m)
        
        # Apply temporal convolution
        hidden_states = hidden_states.transpose(1, 2)  # [B*P, D, T]
        hidden_states = self.conv(hidden_states)  # [B*P, D, T]
        hidden_states = hidden_states.transpose(1, 2)  # [B*P, T, D]
        
        hidden_states = self.layer_norm(hidden_states)
        res_temporal = self.dropout_layer(self.proj_drop(hidden_states))
        
        # Reshape back
        hidden_states = res_temporal.reshape(b, p * self.t, m)
        hidden_states = residual + hidden_states
        hidden_states = torch.cat((init_cls_token, hidden_states), 1)
        
        return hidden_states 