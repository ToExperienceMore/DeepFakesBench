#!/usr/bin/env python3
"""
简单且可靠的CLIP梯度可视化
使用输入梯度方法，比attention hook更直接可靠
"""

import os
import sys
sys.path.append('/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench')
sys.path.append('/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/training')

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import torchvision.transforms as T

from training.detectors.clip_enhanced import CLIPEnhanced

class SimpleGradCAM:
    """
    简单的梯度可视化 - 基于输入梯度的方法
    这个方法比attention hook更直接，适用于任何模型
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def generate_gradcam(self, image_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        生成基于输入梯度的热力图
        Args:
            image_tensor: [1, 3, H, W] 输入图像tensor
            target_class: 目标类别 (0=Real, 1=Fake)
        Returns:
            gradcam: [H, W] 热力图
        """
        # 确保输入需要梯度
        image_tensor.requires_grad_(True)
        
        # 前向传播
        self.model.zero_grad()
        data_dict = {'image': image_tensor}
        pred_dict = self.model(data_dict, inference=True)
        
        # 获取预测
        predictions = pred_dict['cls']  # [1, 2]
        probs = pred_dict['prob']       # [1] fake probability
        
        print(f"📊 模型预测:")
        print(f"   - Fake概率: {probs.item():.4f}")
        print(f"   - 预测类别: {'Fake' if probs.item() > 0.5 else 'Real'}")
        
        # 选择目标类别
        if target_class is None:
            target_class = int(probs.item() > 0.5)
        
        # 反向传播获取梯度
        score = predictions[0, target_class]
        score.backward()
        
        # 获取输入梯度
        gradients = image_tensor.grad.data  # [1, 3, H, W]
        
        if gradients is None:
            print("❌ 无法获取梯度！")
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]))
        
        print(f"✅ 成功获取梯度: {gradients.shape}")
        
        # 计算梯度热力图
        # 方法1: 取梯度的L2范数
        grad_magnitude = torch.norm(gradients, dim=1, keepdim=True)  # [1, 1, H, W]
        
        # 方法2: 取梯度的绝对值然后求和
        # grad_magnitude = torch.sum(torch.abs(gradients), dim=1, keepdim=True)
        
        # 转换为numpy并归一化
        heatmap = grad_magnitude.squeeze().cpu().numpy()  # [H, W]
        
        # 增强对比度的归一化
        if heatmap.max() > heatmap.min():
            # 先归一化到[0,1]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            # 增强对比度：使用幂函数突出高值区域
            heatmap = np.power(heatmap, 0.5)  # 平方根变换，让中高值更突出
            
            # 可选：进一步增强最高值区域
            threshold = np.percentile(heatmap, 90)  # 90%分位数
            heatmap = np.where(heatmap > threshold, heatmap, heatmap * 0.3)  # 低于阈值的区域降低亮度
        
        print(f"📈 增强后热力图统计:")
        print(f"   - 形状: {heatmap.shape}")
        print(f"   - 最小值: {heatmap.min():.6f}")
        print(f"   - 最大值: {heatmap.max():.6f}")
        print(f"   - 平均值: {heatmap.mean():.6f}")
        print(f"   - 90%分位数: {np.percentile(heatmap, 90):.6f}")
        
        return heatmap
    
    def visualize_gradcam(self, original_image: np.ndarray, gradcam: np.ndarray, 
                         alpha: float = 0.4) -> np.ndarray:
        """
        可视化GradCAM
        """
        # 确保原图是[0,1]范围
        if original_image.max() > 1:
            original_image = original_image.astype(np.float32) / 255.0
        
        # 调整gradcam尺寸匹配原图
        if gradcam.shape != original_image.shape[:2]:
            gradcam = cv2.resize(gradcam, (original_image.shape[1], original_image.shape[0]))
        
        # 生成热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # 叠加
        visualization = alpha * heatmap + (1 - alpha) * original_image
        
        return np.clip(visualization, 0, 1)

def load_model_and_config(config_path: str, weights_path: str):
    """加载模型和配置"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    model = CLIPEnhanced(config)
    
    # 加载权重
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 模型权重加载成功")
    
    model.eval()
    return model, config

def preprocess_image(image_path: str) -> tuple:
    """预处理图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # 预处理pipeline
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    # 同时准备用于可视化的原图（resize到224x224）
    viz_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
    ])
    viz_image = np.array(viz_transform(image))
    
    return image_tensor, viz_image

def compare_real_fake_gradcam(model, real_image_path: str, fake_image_path: str, save_dir: str = "real_fake_comparison"):
    """
    对比真脸和假脸的GradCAM，生成差分图
    """
    print("🔍 开始真脸vs假脸GradCAM对比分析...")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建GradCAM
    gradcam = SimpleGradCAM(model)
    
    # 预处理两张图像
    print("📷 预处理真脸图像...")
    real_tensor, real_viz = preprocess_image(real_image_path)
    
    print("📷 预处理假脸图像...")
    fake_tensor, fake_viz = preprocess_image(fake_image_path)
    
    # 分析真脸
    print("\n🧑 分析真脸...")
    real_heatmap_real_class = gradcam.generate_gradcam(real_tensor.clone(), target_class=0)  # Real类别
    real_heatmap_fake_class = gradcam.generate_gradcam(real_tensor.clone(), target_class=1)  # Fake类别
    
    # 分析假脸  
    print("\n🤖 分析假脸...")
    fake_heatmap_real_class = gradcam.generate_gradcam(fake_tensor.clone(), target_class=0)  # Real类别
    fake_heatmap_fake_class = gradcam.generate_gradcam(fake_tensor.clone(), target_class=1)  # Fake类别
    
    # 计算差分图
    print("\n🔄 计算差分图...")
    # 真脸vs假脸在Real类别上的差异
    diff_real_class = real_heatmap_real_class - fake_heatmap_real_class
    # 真脸vs假脸在Fake类别上的差异  
    diff_fake_class = real_heatmap_fake_class - fake_heatmap_fake_class
    
    # 归一化差分图到[-1, 1]
    def normalize_diff(diff_map):
        max_abs = max(abs(diff_map.min()), abs(diff_map.max()))
        if max_abs > 0:
            return diff_map / max_abs
        return diff_map
    
    diff_real_class = normalize_diff(diff_real_class)
    diff_fake_class = normalize_diff(diff_fake_class)
    
    print(f"📊 差分图统计:")
    print(f"   - Real类别差分: [{diff_real_class.min():.3f}, {diff_real_class.max():.3f}]")
    print(f"   - Fake类别差分: [{diff_fake_class.min():.3f}, {diff_fake_class.max():.3f}]")
    
    # 创建综合可视化
    fig = plt.figure(figsize=(20, 16))
    
    # 第一行：原图
    plt.subplot(4, 5, 1)
    plt.imshow(real_viz)
    plt.title('Real Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(4, 5, 2)
    plt.imshow(fake_viz)
    plt.title('Fake Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # 第二行：Real类别热力图 - 使用更显眼的颜色映射
    plt.subplot(4, 5, 6)
    plt.imshow(real_heatmap_real_class, cmap='Reds', vmin=0, vmax=1)
    plt.title('Real Image\n→ Real Class', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=0.6, label='Attention')
    
    plt.subplot(4, 5, 7)
    plt.imshow(fake_heatmap_real_class, cmap='Reds', vmin=0, vmax=1)
    plt.title('Fake Image\n→ Real Class', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=0.6, label='Attention')
    
    plt.subplot(4, 5, 8)
    plt.imshow(diff_real_class, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title('Difference\n(Real-Fake)→Real', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=0.6, label='Diff')
    
    # 第三行：Fake类别热力图 - 使用更显眼的颜色映射
    plt.subplot(4, 5, 11)
    plt.imshow(real_heatmap_fake_class, cmap='Reds', vmin=0, vmax=1)
    plt.title('Real Image\n→ Fake Class', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=0.6, label='Attention')
    
    plt.subplot(4, 5, 12)
    plt.imshow(fake_heatmap_fake_class, cmap='Reds', vmin=0, vmax=1)
    plt.title('Fake Image\n→ Fake Class', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=0.6, label='Attention')
    
    plt.subplot(4, 5, 13)
    plt.imshow(diff_fake_class, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title('Difference\n(Real-Fake)→Fake', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=0.6, label='Diff')
    
    # 第四行：叠加可视化
    real_overlay_real = gradcam.visualize_gradcam(real_viz, real_heatmap_real_class)
    fake_overlay_fake = gradcam.visualize_gradcam(fake_viz, fake_heatmap_fake_class)
    
    plt.subplot(4, 5, 16)
    plt.imshow(real_overlay_real)
    plt.title('Real→Real Overlay', fontsize=10)
    plt.axis('off')
    
    plt.subplot(4, 5, 17)
    plt.imshow(fake_overlay_fake)
    plt.title('Fake→Fake Overlay', fontsize=10)
    plt.axis('off')
    
    plt.suptitle('Real vs Fake GradCAM Comparison & Difference Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存结果
    comparison_path = os.path.join(save_dir, 'real_fake_gradcam_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 对比分析完成！结果保存在: {comparison_path}")
    
    return {
        'real_heatmap_real': real_heatmap_real_class,
        'real_heatmap_fake': real_heatmap_fake_class,
        'fake_heatmap_real': fake_heatmap_real_class,
        'fake_heatmap_fake': fake_heatmap_fake_class,
        'diff_real_class': diff_real_class,
        'diff_fake_class': diff_fake_class
    }

def main():
    """主函数"""
    # 配置路径
    config_path = "training/config/detector/clip_enhanced.yaml"
    weights_path = "./logs/training/clip_enhanced_2025-06-01-19-22-52/test/avg/ckpt_best.pth"
    
    # 🔧 图像路径配置 - 你可以修改这些路径
    #fake_image_path = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/000_003/000.png"
    fake_image_path="/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/000_003/000.png"
    real_image_path = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/original_sequences/youtube/c23/frames/000/000.png"
    #real_image_path = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/original_sequences/youtube/c23/frames/999/150.png"
    
    print("🚀 开始真脸vs假脸GradCAM对比分析...")
    print(f"📂 假脸图像: {os.path.basename(fake_image_path)}")
    print(f"📂 真脸图像: {os.path.basename(real_image_path)}")
    
    # 检查文件
    paths_to_check = [
        (config_path, "配置文件"),
        (weights_path, "权重文件"), 
        (fake_image_path, "假脸图像"),
        (real_image_path, "真脸图像")
    ]
    
    for path, name in paths_to_check:
        if not os.path.exists(path):
            print(f"❌ {name}不存在: {path}")
            return
    
    try:
        # 加载模型
        print("📝 加载模型...")
        model, config = load_model_and_config(config_path, weights_path)
        
        # 执行真脸vs假脸对比分析
        results = compare_real_fake_gradcam(model, real_image_path, fake_image_path)
        
        print("\n🎯 分析总结:")
        print("=" * 60)
        print("📊 热力图颜色解读:")
        print("   - ⚪ 白色：模型不关注的区域（梯度很小）")
        print("   - 🟨 淡红色：模型有一定关注的区域")
        print("   - 🔴 深红色：模型高度关注的区域（梯度很大）")
        print("")
        print("📊 差分图颜色解读:")
        print("   - 🔴 红色：真脸在该位置的关注度 > 假脸")
        print("   - 🔵 蓝色：假脸在该位置的关注度 > 真脸") 
        print("   - ⚪ 白色：两者关注度相似")
        print("")
        print("📁 结果文件保存在: real_fake_comparison/")
        print("🔍 现在关注区域应该更加明显了！")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
