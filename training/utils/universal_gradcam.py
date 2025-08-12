"""
Unified Grad-CAM Implementation
Supports multiple model architectures: Xception, CLIP Enhanced, etc.

Main Features:
1. Automatic model type identification
2. Unified Grad-CAM interface
3. Support for different visualization methods
4. Inter-model comparison analysis

Author: AI Assistant
Date: 2024-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import os
from typing import Union, Dict, Any, Tuple
from abc import ABC, abstractmethod

class BaseGradCAM(ABC):
    """Base class for Grad-CAM"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    @abstractmethod
    def generate_gradcam(self, data_dict: Dict, target_class: int = 1, method: str = 'input_grad') -> np.ndarray:
        """Generate Grad-CAM heatmap"""
        pass
    
    @abstractmethod
    def preprocess_image(self, image_path: str) -> Tuple[Dict, Image.Image]:
        """Preprocess image"""
        pass
    
    def _generate_input_gradient(self, data_dict: Dict, target_class: int) -> np.ndarray:
        """Universal input gradient method"""
        # Get input image and set requires_grad
        image_tensor = data_dict['image'].clone().detach().to(self.device)
        image_tensor.requires_grad_(True)
        
        # Create new data dictionary
        new_data_dict = {'image': image_tensor}
        for key, value in data_dict.items():
            if key != 'image':
                if isinstance(value, torch.Tensor):
                    new_data_dict[key] = value.to(self.device)
                else:
                    new_data_dict[key] = value
        
        # Forward pass
        pred_dict = self.model(new_data_dict, inference=True)
        predictions = pred_dict['cls']
        
        # Select target class score
        score = predictions[0, target_class]
        
        # Backward pass to compute gradients
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Get input gradients
        gradients = image_tensor.grad.data
        
        # Calculate gradient magnitude (L2 norm)
        grad_magnitude = torch.norm(gradients, dim=1)  # [B, H, W]
        
        # Convert to numpy
        heatmap = grad_magnitude.squeeze().cpu().numpy()
        
        # Normalization and contrast enhancement
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            # Contrast enhancement
            heatmap = np.power(heatmap, 0.5)
            # Highlight high-value regions
            threshold = np.percentile(heatmap, 90)
            heatmap = np.where(heatmap > threshold, heatmap, heatmap * 0.3)
        
        return heatmap

class XceptionGradCAM(BaseGradCAM):
    """Xception专用Grad-CAM"""
    
    def __init__(self, model, target_layer_name='conv4', **kwargs):
        super().__init__(model, **kwargs)
        self.target_layer_name = target_layer_name
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # 查找目标层并注册hook
        self._find_target_layer()
        self._register_hooks()
    
    def _find_target_layer(self):
        """查找目标层"""
        if hasattr(self.model, 'backbone'):
            backbone = self.model.backbone
        else:
            backbone = self.model
            
        if hasattr(backbone, self.target_layer_name):
            self.target_layer = getattr(backbone, self.target_layer_name)
            print(f"✅ Xception - 找到目标层: {self.target_layer_name}")
        else:
            print(f"❌ Xception - 未找到目标层: {self.target_layer_name}")
    
    def _register_hooks(self):
        """注册前向和反向hook"""
        if self.target_layer is not None:
            def forward_hook(module, input, output):
                self.activations = output
                
            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0]
            
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_backward_hook(backward_hook)
            print(f"✅ Xception - 已为 {self.target_layer_name} 注册hooks")
    
    def generate_gradcam(self, data_dict: Dict, target_class: int = 1, method: str = 'input_grad') -> np.ndarray:
        """生成Grad-CAM热力图"""
        if method == 'input_grad':
            return self._generate_input_gradient(data_dict, target_class)
        elif method == 'standard' and self.target_layer is not None:
            return self._generate_standard_gradcam(data_dict, target_class)
        else:
            print(f"⚠️ 方法 {method} 不可用，使用输入梯度方法")
            return self._generate_input_gradient(data_dict, target_class)
    
    def _generate_standard_gradcam(self, data_dict: Dict, target_class: int) -> np.ndarray:
        """生成标准Grad-CAM"""
        # 将数据移到正确设备
        device_data_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                device_data_dict[key] = value.to(self.device)
            else:
                device_data_dict[key] = value
        
        # 前向传播
        with torch.enable_grad():
            pred_dict = self.model(device_data_dict, inference=True)
            predictions = pred_dict['cls']
            
            # 选择目标类别的得分
            score = predictions[0, target_class]
            
            # 反向传播
            self.model.zero_grad()
            score.backward(retain_graph=True)
            
            # 获取梯度和激活值
            if self.gradients is None or self.activations is None:
                raise ValueError("未能获取梯度或激活值")
            
            # 计算权重 (全局平均池化)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            
            # 加权求和
            cam = torch.sum(weights * self.activations, dim=1)
            
            # ReLU确保只保留正贡献
            cam = F.relu(cam)
            
            # 转换为numpy
            heatmap = cam.squeeze().detach().cpu().numpy()
            
            # 归一化到[0,1]
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            return heatmap
    
    def preprocess_image(self, image_path: str, resolution: int = 256) -> Tuple[Dict, Image.Image]:
        """预处理图像"""
        image = Image.open(image_path).convert('RGB')
        
        transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        data_dict = {'image': image_tensor}
        
        return data_dict, image

class CLIPEnhancedGradCAM(BaseGradCAM):
    """CLIP Enhanced专用Grad-CAM"""
    
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.clip_path = getattr(model, 'clip_path', None)
        print(f"✅ CLIP Enhanced - 初始化完成")
    
    def generate_gradcam(self, data_dict: Dict, target_class: int = 1, method: str = 'input_grad') -> np.ndarray:
        """生成Grad-CAM热力图"""
        # CLIP Enhanced主要使用输入梯度方法，因为PEFT包装使得attention hook复杂
        return self._generate_input_gradient(data_dict, target_class)
    
    def preprocess_image(self, image_path: str) -> Tuple[Dict, Image.Image]:
        """预处理图像 - 使用CLIP的预处理方式"""
        image = Image.open(image_path).convert('RGB')
        
        # 使用与训练时一致的预处理
        transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                       std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        data_dict = {'image': image_tensor}
        
        return data_dict, image

class UniversalGradCAM:
    """统一的Grad-CAM接口"""
    
    def __init__(self, model, model_type: str = 'auto', **kwargs):
        """
        初始化统一Grad-CAM
        
        Args:
            model: 训练好的模型
            model_type: 模型类型 ('auto', 'xception', 'clip_enhanced')
            **kwargs: 传递给具体GradCAM实现的参数
        """
        self.model = model
        self.model_type = self._detect_model_type(model) if model_type == 'auto' else model_type
        
        # 根据模型类型创建对应的GradCAM实现
        if self.model_type == 'xception':
            self.gradcam = XceptionGradCAM(model, **kwargs)
        elif self.model_type == 'clip_enhanced':
            self.gradcam = CLIPEnhancedGradCAM(model, **kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        print(f"🔧 已创建 {self.model_type} 类型的Grad-CAM")
    
    def _detect_model_type(self, model) -> str:
        """自动检测模型类型"""
        model_class_name = type(model).__name__
        
        if 'Xception' in model_class_name or 'xception' in model_class_name.lower():
            return 'xception'
        elif 'CLIP' in model_class_name or 'clip' in model_class_name.lower():
            return 'clip_enhanced'
        else:
            # 尝试通过属性检测
            if hasattr(model, 'feature_extractor') and hasattr(model, 'clip_path'):
                return 'clip_enhanced'
            elif hasattr(model, 'backbone') and hasattr(model.backbone, 'conv4'):
                return 'xception'
            else:
                print(f"⚠️ 无法自动检测模型类型: {model_class_name}")
                return 'unknown'
    
    def generate_gradcam(self, image_path: str, target_class: int = 1, method: str = 'input_grad') -> Tuple[np.ndarray, Image.Image]:
        """
        生成Grad-CAM热力图
        
        Args:
            image_path: 图像路径
            target_class: 目标类别
            method: 方法类型
            
        Returns:
            heatmap: 热力图
            original_image: 原始图像
        """
        # 预处理图像
        data_dict, original_image = self.gradcam.preprocess_image(image_path)
        
        # 生成热力图
        heatmap = self.gradcam.generate_gradcam(data_dict, target_class, method)
        
        return heatmap, original_image
    
    def visualize_gradcam(self, image_path: str, target_class: int = 1, method: str = 'input_grad', 
                         save_path: str = None, alpha: float = 0.4) -> np.ndarray:
        """
        可视化Grad-CAM结果
        
        Args:
            image_path: 图像路径
            target_class: 目标类别
            method: 方法类型
            save_path: 保存路径
            alpha: 热力图透明度
            
        Returns:
            overlay: 叠加图像
        """
        # 生成热力图
        heatmap, original_image = self.generate_gradcam(image_path, target_class, method)
        
        # 可视化
        overlay = self._visualize_heatmap(original_image, heatmap, save_path, alpha)
        
        return overlay
    
    def _visualize_heatmap(self, image: Image.Image, heatmap: np.ndarray, 
                          save_path: str = None, alpha: float = 0.4) -> np.ndarray:
        """可视化热力图"""
        # 将原始图像转换为numpy数组
        img_array = np.array(image)
        
        # 调整热力图尺寸以匹配原始图像
        h, w = img_array.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # 创建热力图的彩色版本
        heatmap_colored = plt.cm.Reds(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # 叠加热力图到原始图像
        overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(img_array)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 热力图
        im = axes[1].imshow(heatmap_resized, cmap='Reds', vmin=0, vmax=1)
        axes[1].set_title(f'{self.model_type.upper()} Grad-CAM')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], shrink=0.6, label='Attention')
        
        # 叠加图像
        axes[2].imshow(overlay)
        axes[2].set_title('叠加图像')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 {self.model_type} 可视化结果已保存到: {save_path}")
        
        plt.show()
        
        return overlay
    
    def compare_models(self, image_path: str, other_gradcam: 'UniversalGradCAM', 
                      target_class: int = 1, save_dir: str = './comparison_results'):
        """
        比较两个模型的Grad-CAM结果
        
        Args:
            image_path: 图像路径
            other_gradcam: 另一个GradCAM对象
            target_class: 目标类别
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成两个模型的热力图
        heatmap1, image1 = self.generate_gradcam(image_path, target_class)
        heatmap2, image2 = other_gradcam.generate_gradcam(image_path, target_class)
        
        # 创建比较可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        models = [
            (self, heatmap1, image1, self.model_type),
            (other_gradcam, heatmap2, image2, other_gradcam.model_type)
        ]
        
        for i, (gradcam_obj, heatmap, image, model_name) in enumerate(models):
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (w, h))
            heatmap_colored = plt.cm.Reds(heatmap_resized)[:, :, :3]
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
            
            # 原始图像
            axes[i, 0].imshow(img_array)
            axes[i, 0].set_title(f'{model_name.upper()} - 原始图像')
            axes[i, 0].axis('off')
            
            # 热力图
            im = axes[i, 1].imshow(heatmap_resized, cmap='Reds', vmin=0, vmax=1)
            axes[i, 1].set_title(f'{model_name.upper()} - 热力图')
            axes[i, 1].axis('off')
            plt.colorbar(im, ax=axes[i, 1], shrink=0.6)
            
            # 叠加图像
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f'{model_name.upper()} - 叠加结果')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'model_comparison_{self.model_type}_vs_{other_gradcam.model_type}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 模型比较结果已保存到: {save_path}")
        plt.show()

# 便利函数
def create_gradcam(model, model_type: str = 'auto', **kwargs) -> UniversalGradCAM:
    """创建Grad-CAM对象的便利函数"""
    return UniversalGradCAM(model, model_type, **kwargs)

def load_model_and_create_gradcam(config_path: str, weights_path: str, model_type: str = 'auto'):
    """加载模型并创建Grad-CAM对象"""
    import yaml
    import sys
    sys.path.append('training')
    from detectors import DETECTOR
    
    # 读取配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新权重路径
    if 'pretrained' in config:
        config['pretrained'] = weights_path
    elif 'weight' in config:
        config['weight'] = weights_path
    
    # 创建模型
    detector_class = DETECTOR[config['model_name']]
    model = detector_class(config)
    
    # 创建Grad-CAM
    gradcam = create_gradcam(model, model_type)
    
    return model, gradcam

# 使用示例
if __name__ == "__main__":
    print("🔧 统一Grad-CAM工具")
    print("支持的模型类型: Xception, CLIP Enhanced")
    print("请在实际使用时导入此模块并加载训练好的模型")
