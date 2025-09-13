"""
CLIP ViT Attention 可视化示例
使用梯度加权attention方法分析深度伪造检测模型的决策过程
"""

import os
import sys
sys.path.append('/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench')

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPImageProcessor
import yaml

# 导入我们的模块
from training.utils.clip_vit_gradcam import CLIPViTGradCAM
from training.detectors.clip_enhanced import CLIPEnhanced

def load_clip_model(config_path: str, weights_path: str) -> CLIPEnhanced:
    """
    加载CLIP Enhanced模型
    Args:
        config_path: 配置文件路径
        weights_path: 权重文件路径
    Returns:
        加载的模型
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    model = CLIPEnhanced(config)
    
    # 加载权重
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 模型权重加载成功: {weights_path}")
    else:
        print(f"⚠️  权重文件不存在，使用随机初始化: {weights_path}")
    
    model.eval()
    return model

def preprocess_image(image_path: str, config: dict) -> tuple:
    """
    预处理图像 - 使用和训练时一致的预处理方式
    Args:
        image_path: 图像路径
        config: 配置字典
    Returns:
        (preprocessed_tensor, original_image_array)
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # 使用和训练时一致的预处理（来自config和predict.py）
    import torchvision.transforms as T
    
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # 预处理为tensor
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
    
    print(f"📷 图像预处理完成:")
    print(f"   - 原始尺寸: {original_image.shape}")
    print(f"   - 预处理后: {image_tensor.shape}")
    print(f"   - 值范围: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    return image_tensor, original_image

def analyze_single_image(model_config_path: str, weights_path: str, image_path: str, 
                        save_dir: str = "attention_results"):
    """
    分析单张图像的attention模式
    """
    print("🔍 开始单张图像attention分析...")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型
    model = load_clip_model(model_config_path, weights_path)
    
    # 创建GradCAM
    gradcam = CLIPViTGradCAM(
        model, 
        target_layers=[-4, -3, -2, -1],  # 分析最后4层
        head_fusion="mean"
    )
    
    try:
        # 预处理图像
        with open(model_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        image_tensor, original_image = preprocess_image(image_path, config)
        
        # 生成可视化 - 分析两个类别
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        print("📊 分析Real类别的attention...")
        results_real = gradcam.generate_and_visualize(
            image_tensor, 
            original_image,
            target_class=0,  # Real类别
            save_path=os.path.join(save_dir, f"{image_name}_real_attention.png")
        )
        
        print("📊 分析Fake类别的attention...")
        results_fake = gradcam.generate_and_visualize(
            image_tensor, 
            original_image,
            target_class=1,  # Fake类别
            save_path=os.path.join(save_dir, f"{image_name}_fake_attention.png")
        )
        
        # 创建对比图
        create_comparison_plot(original_image, results_real, results_fake, 
                             save_path=os.path.join(save_dir, f"{image_name}_comparison.png"))
        
        print(f"✅ 分析完成！结果保存在: {save_dir}")
        print(f"   - 模型预测Fake概率: {results_fake['fake_probability']:.3f}")
        print(f"   - 预测类别: {'Fake' if results_fake['predicted_class'] else 'Real'}")
        
    finally:
        # 清理
        gradcam.cleanup()

def create_comparison_plot(original_image: np.ndarray, results_real: dict, results_fake: dict,
                          save_path: str):
    """创建Real vs Fake attention对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：Real类别attention
    axes[0,0].imshow(original_image)
    axes[0,0].set_title('Original Image', fontsize=14)
    axes[0,0].axis('off')
    
    axes[0,1].imshow(results_real['cam'], cmap='jet')
    axes[0,1].set_title('Real Class Attention', fontsize=14)
    axes[0,1].axis('off')
    
    axes[0,2].imshow(results_real['visualization'])
    axes[0,2].set_title('Real Class Overlay', fontsize=14)
    axes[0,2].axis('off')
    
    # 第二行：Fake类别attention
    axes[1,0].imshow(original_image)
    axes[1,0].set_title('Original Image', fontsize=14)
    axes[1,0].axis('off')
    
    axes[1,1].imshow(results_fake['cam'], cmap='jet')
    axes[1,1].set_title('Fake Class Attention', fontsize=14)
    axes[1,1].axis('off')
    
    axes[1,2].imshow(results_fake['visualization'])
    axes[1,2].set_title(f'Fake Class Overlay\n(Prob: {results_fake["fake_probability"]:.3f})', fontsize=14)
    axes[1,2].axis('off')
    
    plt.suptitle('CLIP ViT Attention Analysis: Real vs Fake', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_attention_layers(model_config_path: str, weights_path: str, image_path: str,
                           save_dir: str = "layer_analysis"):
    """
    分析不同层的attention模式
    """
    print("🔍 开始多层attention分析...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型
    model = load_clip_model(model_config_path, weights_path)
    
    # 预处理图像
    with open(model_config_path, 'r') as f:
        config = yaml.safe_load(f)
    image_tensor, original_image = preprocess_image(image_path, config)
    
    # 分析不同层组合
    layer_configs = [
        ([-1], "last_layer"),
        ([-2, -1], "last_2_layers"), 
        ([-4, -3, -2, -1], "last_4_layers"),
        ([-6, -4, -2], "spaced_layers"),
        (list(range(-12, 0)), "all_layers")  # 所有层
    ]
    
    fig, axes = plt.subplots(2, len(layer_configs), figsize=(4*len(layer_configs), 8))
    
    for i, (target_layers, layer_name) in enumerate(layer_configs):
        print(f"📊 分析层配置: {layer_name} (layers: {target_layers})")
        
        # 创建GradCAM
        gradcam = CLIPViTGradCAM(model, target_layers=target_layers, head_fusion="mean")
        
        try:
            # 生成CAM
            cam = gradcam.generate_cam(image_tensor, target_class=1)[0]  # Fake类别
            visualization = gradcam.visualize_cam(original_image, cam)
            
            # 绘制热力图
            axes[0, i].imshow(cam, cmap='jet')
            axes[0, i].set_title(f'{layer_name}\nHeatmap', fontsize=10)
            axes[0, i].axis('off')
            
            # 绘制叠加图
            axes[1, i].imshow(visualization)
            axes[1, i].set_title(f'{layer_name}\nOverlay', fontsize=10)
            axes[1, i].axis('off')
            
        finally:
            gradcam.cleanup()
    
    plt.suptitle('Multi-Layer Attention Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'layer_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数 - 修改这里的路径来运行你的分析"""
    
    # 🔧 配置路径 - 请根据你的实际路径修改
    model_config_path = "training/config/detector/clip_enhanced.yaml"
    #weights_path = "training/FaceForensics++/ckpt_epoch_9_best.pth"  # 你的模型权重路径
    weights_path = "./logs/training/clip_enhanced_2025-06-01-19-22-52/test/avg/ckpt_best.pth"
    
    # 测试图像路径 - 请使用你要分析的图像
    #test_image_path = "test_image.jpg"  # 请替换为实际图像路径
    test_image_path = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/000_003/009.png"
    
    # 检查文件是否存在
    if not os.path.exists(model_config_path):
        print(f"❌ 配置文件不存在: {model_config_path}")
        return
    
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        print("请将你要分析的图像放在当前目录下，并命名为test_image.jpg")
        return
    
    try:
        print("🚀 开始CLIP ViT Attention可视化分析...")
        
        # 1. 单张图像分析
        analyze_single_image(model_config_path, weights_path, test_image_path)
        
        # 2. 多层分析 (可选)
        print("\n" + "="*60)
        response = input("是否进行多层attention分析？(y/n): ")
        if response.lower() == 'y':
            analyze_attention_layers(model_config_path, weights_path, test_image_path)
        
        print("\n✅ 所有分析完成！")
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
