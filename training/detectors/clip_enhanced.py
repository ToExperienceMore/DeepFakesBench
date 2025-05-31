import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from typing import Optional, Tuple, Dict, Any, List
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
        
        self.clip_path = config.get('clip_path', "weights/clip-vit-base-patch16")
        print(f"clip_path: {self.clip_path}")
        if not os.path.exists(self.clip_path):
            raise ValueError(f"本地模型文件不存在: {self.clip_path}，请确保模型文件已下载到正确位置")
        
        self.vision_encoder = CLIPVisionModel.from_pretrained(self.clip_path)
        self.text_encoder = CLIPTextModel.from_pretrained(self.clip_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.clip_path)

        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        target_modules_list=['pre_layrnorm', 'layer_norm1', 'layer_norm2', 'post_layernorm', 'layernorm']
        peft_config = LNTuningConfig(target_modules=target_modules_list)
        self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)

        self.n_ctx = config.get('n_ctx', 5)
        self.ctx_dim = self.text_encoder.config.hidden_size
        self.classnames = ["a photo of a real face", "a photo of a fake face"]
        self.n_classes = len(self.classnames)
        
        self.ctx = nn.Parameter(torch.randn(self.n_ctx, self.ctx_dim))
        self.ctx.requires_grad = True

        self.prompts = [" ".join(["X"] * self.n_ctx) + " " + name for name in self.classnames]
        tokenized = self.tokenizer(self.prompts, return_tensors="pt", padding=True, truncation=True)
        self.prompt_ids = tokenized["input_ids"]
        self.prompt_attention_mask = tokenized["attention_mask"]

        self.projection = nn.Linear(self.vision_encoder.config.hidden_size, self.text_encoder.config.hidden_size)
        self.loss_func = self.build_loss(config)

        self.print_trainable_parameters()
        self._debug_step = 0

    def encode_text_with_prompt(self):
        tokenized = self.tokenizer(self.prompts, return_tensors="pt", padding=True, truncation=True)
        tokenized = {k: v.to(self.ctx.device) for k, v in tokenized.items()}
        inputs = self.text_encoder.get_input_embeddings()(tokenized["input_ids"]).clone()

        x_token_id = self.tokenizer.encode("X", add_special_tokens=False)[0]
        for i in range(self.n_classes):
            x_positions = (tokenized["input_ids"][i] == x_token_id).nonzero(as_tuple=True)[0]
            if len(x_positions) >= self.n_ctx:
                start = x_positions[0]
                inputs[i, start:start+self.n_ctx] = self.ctx

        outputs = self.text_encoder(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"]
        )
        return F.normalize(outputs.last_hidden_state[:, 0, :], dim=-1)

    def features(self, data_dict: dict) -> torch.tensor:
        x = data_dict['image']
        device = next(self.parameters()).device
        x = x.to(device)
        
        image_output = self.vision_encoder(x)
        image_features = image_output.last_hidden_state[:, 0, :]
        image_features = self.projection(image_features)
        return F.normalize(image_features, dim=-1)

    def classifier(self, features: torch.tensor) -> torch.tensor:
        text_features = self.encode_text_with_prompt()
        logits = features @ text_features.T
        return logits

    def forward(self, data_dict: dict, inference=False) -> dict:
        image_features = self.features(data_dict)
        text_features = self.encode_text_with_prompt()
        logits = image_features @ text_features.T
        prob = torch.softmax(logits, dim=1)[:, 1]

        pred_dict = {
            'cls': logits,
            'prob': prob
        }

        if not inference:
            self._debug_step += 1
            if self._debug_step % 100 == 0:
                print("\n[DEBUG] ctx value (first 5 elements):", self.ctx.flatten()[:5].data.cpu().numpy())
                print("[DEBUG] ctx mean/std:", self.ctx.mean().item(), self.ctx.std().item())
                print("[DEBUG] ctx requires_grad:", self.ctx.requires_grad)
                print("[DEBUG] ctx grad (first 5 elements):", None if self.ctx.grad is None else self.ctx.grad.flatten()[:5].data.cpu().numpy())
                print("[DEBUG] ctx param id:", id(self.ctx))
                found = False
                for name, param in self.named_parameters():
                    if id(param) == id(self.ctx):
                        print(f"[DEBUG] ctx is in model.named_parameters() as: {name}")
                        found = True
                if not found:
                    print("[DEBUG] ctx is NOT in model.named_parameters()!")
                if hasattr(self, 'optimizer') and self.optimizer is not None:
                    found_opt = False
                    for group in self.optimizer.param_groups:
                        for p in group['params']:
                            if id(p) == id(self.ctx):
                                print("[DEBUG] ctx is in optimizer param_groups!")
                                found_opt = True
                    if not found_opt:
                        print("[DEBUG] ctx is NOT in optimizer param_groups!")
        return pred_dict

    def get_preprocessing(self):
        processor = CLIPImageProcessor.from_pretrained(self.clip_path)
        def preprocess(image):
            return processor(images=image, return_tensors="pt")["pixel_values"][0]
        return preprocess

    def build_backbone(self, config):
        return self.vision_encoder

    def build_loss(self, config):
        loss_name = config.get('loss_func', 'CrossEntropyLoss')
        loss_class = LOSSFUNC[loss_name]
        loss_func = loss_class()
        return loss_func

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        return {'overall': loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']  
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def print_trainable_parameters(self):
        print("\nAll parameters:")
        for name, param in self.named_parameters():
            print(f"{name} shape = {tuple(param.shape)}")

        print("\n🔥 Trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name} shape = {tuple(param.shape)}")

        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"Total parameters: {all_params}, trainable: {trainable_params}, %: {trainable_params / all_params * 100:.4f}"
        )