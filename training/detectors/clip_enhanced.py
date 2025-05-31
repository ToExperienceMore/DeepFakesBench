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
        
        # Initialize CLIP models
        self.clip_path = config.get('clip_path', "weights/clip-vit-base-patch16")
        print(f"clip_path: {self.clip_path}")
        if not os.path.exists(self.clip_path):
            raise ValueError(f"本地模型文件不存在: {self.clip_path}，请确保模型文件已下载到正确位置")
        
        self.vision_encoder = CLIPVisionModel.from_pretrained(self.clip_path)
        self.text_encoder = CLIPTextModel.from_pretrained(self.clip_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.clip_path)

        # 首先冻结所有参数
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 配置可学习的 LayerNorm
        target_modules_list=['pre_layrnorm', 'layer_norm1', 'layer_norm2', 'post_layernorm', 'layernorm']
        peft_config = LNTuningConfig(target_modules=target_modules_list)

        self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)
        
        # 设置可学习的提示词
        self.n_ctx = config.get('n_ctx', 5)  # 可学习的上下文token数量
        self.ctx_dim = self.text_encoder.config.hidden_size
        self.classnames = ["a photo of a real face", "a photo of a fake face"]
        self.n_classes = len(self.classnames)
        
        # 可学习的上下文提示词 - 使用 nn.Parameter 注册为模型参数
        self.ctx = nn.Parameter(torch.randn(self.n_ctx, self.ctx_dim))
        self.ctx.requires_grad = True  # 确保可训练
        
        # 添加可学习的投影层
        self.projection = nn.Linear(self.vision_encoder.config.hidden_size, self.text_encoder.config.hidden_size)
        
        # Initialize loss function
        self.loss_func = self.build_loss(config)
        
        # 打印可训练参数
        self.print_trainable_parameters()
        # 初始化调试步数
        self._debug_step = 0
    
    def encode_text_with_prompt(self):
        device = self.ctx.device
        input_ids = self.prompt_ids.to(device)
        embed = self.text_encoder.get_input_embeddings()(input_ids)  # [n_class, seq_len, 512]

        # Remove BOS and EOS
        embed_cls = self.text_encoder.get_input_embeddings().weight[self.tokenizer.bos_token_id].unsqueeze(0).unsqueeze(0).expand(self.n_classes, 1, -1)
        embed_eos = self.text_encoder.get_input_embeddings().weight[self.tokenizer.eos_token_id].unsqueeze(0).unsqueeze(0).expand(self.n_classes, 1, -1)
        embed_class_tokens = embed[:, 1:-1, :]  # remove BOS and EOS

        # Final prompt = [CLS] + ctx + class name tokens + [EOS]
        full_prompt_embed = torch.cat([
            embed_cls,
            self.ctx.unsqueeze(0).expand(self.n_classes, -1, -1),
            embed_class_tokens,
            embed_eos
        ], dim=1)  # [n_class, seq_len, 512]

        attention_mask = torch.ones(full_prompt_embed.shape[:2], dtype=torch.long, device=device)
        output = self.text_encoder(inputs_embeds=full_prompt_embed, attention_mask=attention_mask)
        text_features = output.last_hidden_state[:, 0, :]  # <CLS>
        return F.normalize(text_features, dim=-1)
    """
    def encode_text_with_prompt(self):
        """使用可学习的提示词编码文本"""
        # 构建提示词模板
        prompts = []
        for classname in self.classnames:
            prompt = " ".join(["X"] * self.n_ctx) + " " + classname
            prompts.append(prompt)
        
        # 对提示词进行编码
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        
        # 确保所有张量都在正确的设备上
        device = next(self.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        self.ctx = self.ctx.to(device)
        
        # 获取输入嵌入
        inputs = self.text_encoder.get_input_embeddings()(tokenized["input_ids"])
        
        # 替换可学习的上下文token
        for i in range(self.n_classes):
            # 找到第一个 X token 的位置
            x_token_id = self.tokenizer.encode("X")[0]
            x_positions = (tokenized["input_ids"][i] == x_token_id).nonzero(as_tuple=True)[0]
            if len(x_positions) >= self.n_ctx:
                start_idx = x_positions[0]
                # 使用 detach() 来避免梯度传播问题
                inputs[i, start_idx:start_idx+self.n_ctx] = self.ctx
        
        # 使用文本编码器
        outputs = self.text_encoder(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"]
        )
        
        text_features = outputs.last_hidden_state[:, 0, :]  # 取CLS token
        return F.normalize(text_features, dim=-1)
    """
    
    def features(self, data_dict: dict) -> torch.tensor:
        """提取图像特征"""
        x = data_dict['image']
        # 确保输入在正确的设备上
        device = next(self.parameters()).device
        x = x.to(device)
        
        image_output = self.vision_encoder(x)
        image_features = image_output.last_hidden_state[:, 0, :]  # CLS token
        # 使用投影层
        image_features = self.projection(image_features)
        return F.normalize(image_features, dim=-1)
    
    def classifier(self, features: torch.tensor) -> torch.tensor:
        """使用文本特征作为分类器"""
        text_features = self.encode_text_with_prompt()
        logits = features @ text_features.T  # [B, 2]
        return logits
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        """前向传播"""
        image_features = self.features(data_dict)
        text_features = self.encode_text_with_prompt()
        
        # 计算图像特征和文本特征的相似度
        logits = image_features @ text_features.T  # [B, 2]
        prob = torch.softmax(logits, dim=1)[:, 1]
        
        pred_dict = {
            'cls': logits,
            'prob': prob
        }
        # ====== 调试 ctx 参数 ======
        if not inference:
            self._debug_step += 1
            if self._debug_step % 100 == 0:
                print("\n[DEBUG] ctx value (first 5 elements):", self.ctx.flatten()[:5].data.cpu().numpy())
                print("[DEBUG] ctx mean/std:", self.ctx.mean().item(), self.ctx.std().item())
                print("[DEBUG] ctx requires_grad:", self.ctx.requires_grad)
                print("[DEBUG] ctx grad (first 5 elements):", None if self.ctx.grad is None else self.ctx.grad.flatten()[:5].data.cpu().numpy())
                print("[DEBUG] ctx param id:", id(self.ctx))
                # 检查 ctx 是否在 model.parameters()
                found = False
                for name, param in self.named_parameters():
                    if id(param) == id(self.ctx):
                        print(f"[DEBUG] ctx is in model.named_parameters() as: {name}")
                        found = True
                if not found:
                    print("[DEBUG] ctx is NOT in model.named_parameters()!")
                # 检查 ctx 是否在 optimizer param group（如果 optimizer 已经赋值到 self.optimizer）
                if hasattr(self, 'optimizer') and self.optimizer is not None:
                    found_opt = False
                    for group in self.optimizer.param_groups:
                        for p in group['params']:
                            if id(p) == id(self.ctx):
                                print("[DEBUG] ctx is in optimizer param_groups!")
                                found_opt = True
                    if not found_opt:
                        print("[DEBUG] ctx is NOT in optimizer param_groups!")
        # ====== END ======
        return pred_dict

    def get_preprocessing(self):
        """获取预处理函数"""
        processor = CLIPImageProcessor.from_pretrained(self.clip_path)
        def preprocess(image):
            return processor(images=image, return_tensors="pt")["pixel_values"][0]
        return preprocess

    def build_backbone(self, config):
        """Build the backbone network"""
        return self.vision_encoder
    
    def build_loss(self, config):
        """Build the loss function"""
        loss_name = config.get('loss_func', 'CrossEntropyLoss')
        loss_class = LOSSFUNC[loss_name]
        loss_func = loss_class()
        return loss_func
    
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