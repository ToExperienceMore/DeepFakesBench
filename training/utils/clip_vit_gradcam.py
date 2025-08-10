import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
import warnings

class CLIPViTGradCAM:
    """
    ViT Gradient-weighted Attention CAM for CLIP models
    
    实现原理：
    1. 注册attention层的forward和backward hooks
    2. 前向传播时保存attention weights
    3. 反向传播时计算attention的梯度
    4. 用梯度加权attention weights
    5. 将patch-level重要性映射回原图
    """
    
    def __init__(self, model, target_layers: Optional[List[int]] = None, head_fusion: str = "mean"):
        """
        Args:
            model: CLIPEnhanced model
            target_layers: 要分析的transformer层索引，None表示使用最后几层
            head_fusion: 多头attention融合方法 ("mean", "max", "min")
        """
        self.model = model
        # 先只用最后一层进行调试
        self.target_layers = target_layers or [-1]  # 先只测试最后一层
        self.head_fusion = head_fusion
        
        print(f"🎯 初始化GradCAM，目标层: {self.target_layers}")
        
        # 存储注册的hooks
        self.hooks = []
        
        # 存储attention weights和gradients
        self.attention_weights = {}
        self.attention_gradients = {}
        
        # 获取CLIP vision model
        self.clip_vision = self._get_clip_vision_model()
        
        # 注册hooks
        self._register_hooks()
        
    def _get_clip_vision_model(self):
        """获取CLIP vision transformer模型"""
        # 处理PEFT包装的模型
        if hasattr(self.model.feature_extractor, 'base_model'):
            clip_vision = self.model.feature_extractor.base_model.vision_model
        else:
            clip_vision = self.model.feature_extractor.vision_model
            
        print(f"🔍 CLIP Vision Model Info:")
        print(f"   - Type: {type(clip_vision)}")
        print(f"   - Has encoder: {hasattr(clip_vision, 'encoder')}")
        if hasattr(clip_vision, 'encoder'):
            print(f"   - Encoder layers: {len(clip_vision.encoder.layers)}")
            
        return clip_vision
    
    def _register_hooks(self):
        """注册attention层的hooks"""
        encoder_layers = self.clip_vision.encoder.layers
        
        for i, layer_idx in enumerate(self.target_layers):
            # 处理负索引
            if layer_idx < 0:
                layer_idx = len(encoder_layers) + layer_idx
                
            layer = encoder_layers[layer_idx]
            attention_module = layer.self_attn
            
            # 注册forward hook计算并保存attention weights
            def save_attention_hook(layer_name):
                def hook(module, input, output):
                    try:
                        print(f"🔍 Hook triggered for {layer_name}")
                        print(f"   - Input type: {type(input)}")
                        print(f"   - Input length: {len(input) if isinstance(input, (tuple, list)) else 'not sequence'}")
                        print(f"   - Output type: {type(output)}")
                        print(f"   - Output length: {len(output) if isinstance(output, (tuple, list)) else 'not sequence'}")
                        
                        # 尝试多种方式获取hidden_states
                        hidden_states = None
                        
                        if isinstance(input, tuple) and len(input) > 0:
                            hidden_states = input[0]
                        elif isinstance(input, torch.Tensor):
                            hidden_states = input
                        else:
                            print(f"❌ Cannot extract hidden_states from input for {layer_name}")
                            return
                        
                        if not isinstance(hidden_states, torch.Tensor):
                            print(f"❌ hidden_states is not a tensor for {layer_name}: {type(hidden_states)}")
                            return
                            
                        print(f"   - Hidden states shape: {hidden_states.shape}")
                        
                        # 检查维度
                        if len(hidden_states.shape) != 3:
                            print(f"❌ Unexpected hidden_states shape for {layer_name}: {hidden_states.shape}")
                            return
                            
                        B, L, D = hidden_states.shape
                        print(f"   - Batch size: {B}, Sequence length: {L}, Hidden dim: {D}")
                        
                        # 检查模块属性
                        if not hasattr(module, 'q_proj') or not hasattr(module, 'k_proj'):
                            print(f"❌ Module missing q_proj or k_proj for {layer_name}")
                            return
                        
                        # 获取Q, K, V
                        query = module.q_proj(hidden_states)  # [B, L, D]
                        key = module.k_proj(hidden_states)    # [B, L, D]
                        
                        print(f"   - Query shape: {query.shape}, Key shape: {key.shape}")
                        
                        # 检查模块配置
                        num_heads = getattr(module, 'num_heads', 12)
                        head_dim = getattr(module, 'head_dim', D // num_heads)
                        
                        print(f"   - Num heads: {num_heads}, Head dim: {head_dim}")
                        
                        # Reshape to multi-head format
                        query = query.view(B, L, num_heads, head_dim).transpose(1, 2)  # [B, H, L, head_dim]
                        key = key.view(B, L, num_heads, head_dim).transpose(1, 2)      # [B, H, L, head_dim]
                        
                        # 计算attention scores
                        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)  # [B, H, L, L]
                        attention_probs = torch.softmax(attention_scores, dim=-1)  # [B, H, L, L]
                        
                        # 确保tensor需要梯度
                        if attention_probs.requires_grad:
                            self.attention_weights[layer_name] = attention_probs
                        else:
                            self.attention_weights[layer_name] = attention_probs.detach().requires_grad_(True)
                            
                        print(f"✅ Successfully computed attention for {layer_name}: {attention_probs.shape}")
                            
                    except Exception as e:
                        print(f"❌ Failed to compute attention for {layer_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                return hook
            
            # 注册backward hook保存gradients
            def save_gradient_hook(layer_name):
                def hook(module, grad_input, grad_output):
                    # 保存attention weights的梯度
                    if layer_name in self.attention_weights and self.attention_weights[layer_name] is not None:
                        if self.attention_weights[layer_name].requires_grad:
                            def grad_hook(grad):
                                self.attention_gradients[layer_name] = grad
                                return grad
                            
                            self.attention_weights[layer_name].register_hook(grad_hook)
                return hook
            
            layer_name = f"layer_{layer_idx}"
            
            # 注册hooks
            forward_hook = attention_module.register_forward_hook(save_attention_hook(layer_name))
            backward_hook = attention_module.register_full_backward_hook(save_gradient_hook(layer_name))
            
            self.hooks.extend([forward_hook, backward_hook])
    
    def _fuse_attention_heads(self, attention: torch.Tensor) -> torch.Tensor:
        """
        融合多头attention
        Args:
            attention: [B, H, L, L] 多头attention权重
        Returns:
            fused_attention: [B, L, L] 融合后的attention
        """
        if self.head_fusion == "mean":
            return attention.mean(dim=1)
        elif self.head_fusion == "max":
            return attention.max(dim=1)[0]
        elif self.head_fusion == "min":
            return attention.min(dim=1)[0]
        else:
            raise ValueError(f"Unknown head fusion method: {self.head_fusion}")
    
    def _compute_gradient_weighted_attention(self) -> torch.Tensor:
        """
        计算梯度加权的attention
        Returns:
            cam: [B, H, W] 热力图
        """
        if not self.attention_weights or not self.attention_gradients:
            raise ValueError("No attention weights or gradients found. Make sure to run forward and backward first.")
        
        batch_size = None
        weighted_attentions = []
        
        for layer_name in self.attention_weights.keys():
            if layer_name not in self.attention_gradients:
                warnings.warn(f"No gradients found for layer {layer_name}, skipping...")
                continue
                
            attention = self.attention_weights[layer_name]  # [B, H, L, L]
            gradients = self.attention_gradients[layer_name]  # [B, H, L, L]
            
            if batch_size is None:
                batch_size = attention.size(0)
            
            # 梯度加权attention
            weighted_attention = attention * gradients  # [B, H, L, L]
            
            # 融合多头
            fused_attention = self._fuse_attention_heads(weighted_attention)  # [B, L, L]
            
            # 取CLS token对patch的attention (去掉CLS token)
            cls_attention = fused_attention[:, 0, 1:]  # [B, L-1] = [B, 196]
            
            weighted_attentions.append(cls_attention)
        
        if not weighted_attentions:
            raise ValueError("No valid weighted attentions computed.")
        
        # 平均多层attention
        final_attention = torch.stack(weighted_attentions).mean(dim=0)  # [B, 196]
        
        # 重塑为spatial维度 (假设是14x14的patch grid)
        patch_size = int(np.sqrt(final_attention.size(-1)))  # 14
        spatial_attention = final_attention.view(batch_size, patch_size, patch_size)  # [B, 14, 14]
        
        return spatial_attention
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None, 
                     retain_graph: bool = False) -> np.ndarray:
        """
        生成CAM热力图
        Args:
            input_tensor: 输入图像tensor [B, C, H, W]
            target_class: 目标类别，None表示使用预测的类别
            retain_graph: 是否保留计算图
        Returns:
            cam: [B, H, W] numpy热力图，值在[0,1]范围
        """
        # 清空之前的记录
        self.attention_weights.clear()
        self.attention_gradients.clear()
        
        # 设置模型为eval模式但保持requires_grad
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # 前向传播
        data_dict = {'image': input_tensor}
        pred_dict = self.model(data_dict, inference=True)
        
        # 获取预测和目标类别
        predictions = pred_dict['cls']  # [B, num_classes]
        
        if target_class is None:
            target_class = predictions.argmax(dim=1)
        elif isinstance(target_class, int):
            target_class = torch.tensor([target_class] * predictions.size(0), device=predictions.device)
        
        # 清零梯度
        self.model.zero_grad()
        
        # 反向传播
        target_scores = predictions.gather(1, target_class.unsqueeze(1)).squeeze(1)
        loss = target_scores.sum()
        loss.backward(retain_graph=retain_graph)
        
        # 计算梯度加权attention
        try:
            spatial_attention = self._compute_gradient_weighted_attention()  # [B, 14, 14]
        except ValueError as e:
            print(f"❌ Error computing attention: {e}")
            print(f"🔍 Debug info:")
            print(f"   - Attention weights collected: {list(self.attention_weights.keys())}")
            print(f"   - Attention gradients collected: {list(self.attention_gradients.keys())}")
            print(f"   - Target layers: {self.target_layers}")
            
            # 返回全零热力图
            batch_size, _, height, width = input_tensor.shape
            print(f"⚠️  Returning zero CAM due to attention computation failure")
            return np.zeros((batch_size, height, width))
        
        # 应用ReLU和归一化
        spatial_attention = F.relu(spatial_attention)
        
        # 归一化到[0,1]
        batch_size = spatial_attention.size(0)
        for i in range(batch_size):
            attention_i = spatial_attention[i]
            min_val = attention_i.min()
            max_val = attention_i.max()
            if max_val > min_val:
                spatial_attention[i] = (attention_i - min_val) / (max_val - min_val)
        
        # 上采样到原图尺寸
        input_height, input_width = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(
            spatial_attention.unsqueeze(1),  # [B, 1, 14, 14]
            size=(input_height, input_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B, H, W]
        
        return cam.detach().cpu().numpy()
    
    def visualize_cam(self, image: np.ndarray, cam: np.ndarray, alpha: float = 0.4,
                     colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        可视化CAM热力图
        Args:
            image: 原始图像 [H, W, C] 或 [H, W]，值在[0,255]
            cam: CAM热力图 [H, W]，值在[0,1]
            alpha: 热力图透明度
            colormap: OpenCV颜色映射
        Returns:
            visualization: 可视化结果 [H, W, C]
        """
        # 确保image是3通道
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = image.copy()
        
        # 归一化image到[0,1]
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # 确保CAM和image尺寸一致
        if cam.shape != image.shape[:2]:
            # 将CAM resize到image的尺寸
            cam = cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            print(f"📏 Resized CAM from shape to {cam.shape} to match image shape {image.shape[:2]}")
        
        # 生成热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        heatmap = np.float32(heatmap) / 255.0
        
        # 确保RGB顺序
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 叠加热力图
        visualization = alpha * heatmap + (1 - alpha) * image
        
        # 归一化到[0,1]
        visualization = np.clip(visualization, 0, 1)
        
        return visualization
    
    def generate_and_visualize(self, input_tensor: torch.Tensor, original_image: np.ndarray,
                              target_class: Optional[int] = None, save_path: Optional[str] = None) -> Dict:
        """
        一键生成并可视化CAM
        Args:
            input_tensor: 输入tensor [1, C, H, W] 
            original_image: 原始图像用于可视化 [H, W, C]
            target_class: 目标类别
            save_path: 保存路径
        Returns:
            results: 包含预测、CAM和可视化的字典
        """
        # 生成CAM
        cam = self.generate_cam(input_tensor, target_class)[0]  # 取第一个样本
        
        # 获取预测
        self.model.eval()
        with torch.no_grad():
            data_dict = {'image': input_tensor}
            pred_dict = self.model(data_dict, inference=True)
            predictions = pred_dict['prob'].cpu().numpy()[0]  # Fake概率
            predicted_class = int(predictions > 0.5)
        
        # 可视化
        visualization = self.visualize_cam(original_image, cam)
        
        # 创建结果图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 热力图
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title(f'Attention Heatmap\nFake Prob: {predictions:.3f}')
        axes[1].axis('off')
        
        # 叠加图
        axes[2].imshow(visualization)
        axes[2].set_title(f'Overlay\nPredicted: {"Fake" if predicted_class else "Real"}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return {
            'cam': cam,
            'visualization': visualization,
            'fake_probability': predictions,
            'predicted_class': predicted_class,
            'target_class': target_class
        }
    
    def cleanup(self):
        """清理注册的hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_weights.clear()
        self.attention_gradients.clear()
    
    def __del__(self):
        """析构函数，自动清理"""
        self.cleanup()

# 使用示例和测试函数
def test_clip_vit_gradcam():
    """测试函数"""
    print("Testing CLIP ViT GradCAM...")
    
    # 这里应该加载你的实际模型
    # model = load_your_clip_enhanced_model()
    # gradcam = CLIPViTGradCAM(model)
    
    # # 准备测试图像
    # image_tensor = torch.randn(1, 3, 224, 224)
    # original_image = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    
    # # 生成可视化
    # results = gradcam.generate_and_visualize(
    #     image_tensor, 
    #     original_image,
    #     save_path='test_gradcam.png'
    # )
    
    print("GradCAM implementation ready!")

if __name__ == "__main__":
    test_clip_vit_gradcam()
