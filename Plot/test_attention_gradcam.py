#!/usr/bin/env python3
"""
快速测试CLIP ViT Attention可视化功能
这个脚本会创建合成数据来测试GradCAM是否正常工作
"""

import os
import sys
sys.path.append('/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench')

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import yaml

from training.utils.clip_vit_gradcam import CLIPViTGradCAM
from training.detectors.clip_enhanced import CLIPEnhanced

# 确保可以找到training模块
sys.path.append('/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/training')

def create_test_config():
    """创建测试用的配置"""
    config = {
        'backbone': 'ViT-L/14',
        'clip_path': 'weights/clip-vit-base-patch16',
        'mlp_layer': 1,
        'loss_func': 'cross_entropy'  # 修正loss函数名称
    }
    return config

def create_synthetic_image(size=(224, 224)):
    """创建合成测试图像"""
    # 创建一个有结构的测试图像
    image = np.zeros((*size, 3), dtype=np.uint8)
    
    # 添加一些几何形状来测试attention
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # 红色圆形
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - center_x)**2 + (y - center_y)**2 <= 40**2
    image[mask] = [255, 100, 100]
    
    # 蓝色方形
    image[50:100, 50:100] = [100, 100, 255]
    
    # 绿色三角形区域
    for i in range(150, 200):
        for j in range(150, 150 + (i - 150)):
            if j < size[1]:
                image[i, j] = [100, 255, 100]
    
    # 添加一些噪声
    noise = np.random.randint(0, 50, size=(*size, 3))
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return image

def test_gradcam_basic():
    """基础功能测试"""
    print("🧪 开始基础功能测试...")
    
    try:
        # 创建测试配置和模型
        config = create_test_config()
        print("✅ 配置创建成功")
        
        # 检查CLIP路径是否存在
        if not os.path.exists(config['clip_path']):
            print(f"⚠️  CLIP权重不存在: {config['clip_path']}")
            print("   使用随机初始化进行测试...")
        
        # 创建模型（可能会失败如果没有权重）
        try:
            model = CLIPEnhanced(config)
            model.eval()
            print("✅ 模型创建成功")
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            return False
        
        # 创建GradCAM
        try:
            gradcam = CLIPViTGradCAM(model, target_layers=[-2, -1], head_fusion="mean")
            print("✅ GradCAM创建成功")
        except Exception as e:
            print(f"❌ GradCAM创建失败: {e}")
            return False
        
        # 创建测试数据
        test_image = create_synthetic_image()
        test_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0
        test_tensor = test_tensor.unsqueeze(0)  # [1, 3, 224, 224]
        print("✅ 测试数据创建成功")
        
        # 测试前向传播
        try:
            with torch.no_grad():
                data_dict = {'image': test_tensor}
                pred_dict = model(data_dict, inference=True)
                print(f"✅ 前向传播成功，输出形状: {pred_dict['cls'].shape}")
                print(f"   预测概率: {pred_dict['prob'].item():.3f}")
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return False
        
        # 测试GradCAM生成
        try:
            cam = gradcam.generate_cam(test_tensor, target_class=1)
            print(f"✅ CAM生成成功，形状: {cam.shape}")
            print(f"   CAM值范围: [{cam.min():.3f}, {cam.max():.3f}]")
        except Exception as e:
            print(f"❌ CAM生成失败: {e}")
            gradcam.cleanup()
            return False
        
        # 测试可视化
        try:
            visualization = gradcam.visualize_cam(test_image, cam[0])
            print(f"✅ 可视化成功，形状: {visualization.shape}")
        except Exception as e:
            print(f"❌ 可视化失败: {e}")
            gradcam.cleanup()
            return False
        
        # 保存测试结果
        try:
            os.makedirs('test_results', exist_ok=True)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(test_image)
            axes[0].set_title('Synthetic Test Image')
            axes[0].axis('off')
            
            axes[1].imshow(cam[0], cmap='jet')
            axes[1].set_title('Attention Heatmap')
            axes[1].axis('off')
            
            axes[2].imshow(visualization)
            axes[2].set_title('Overlay Visualization')
            axes[2].axis('off')
            
            plt.suptitle('CLIP ViT GradCAM Test Results', fontsize=16)
            plt.tight_layout()
            plt.savefig('test_results/gradcam_test.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ 测试结果保存成功: test_results/gradcam_test.png")
        except Exception as e:
            print(f"❌ 结果保存失败: {e}")
        
        # 清理
        gradcam.cleanup()
        print("✅ 资源清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试过程出现未预期错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_layers():
    """多层attention测试"""
    print("\n🔬 开始多层attention测试...")
    
    try:
        config = create_test_config()
        
        if not os.path.exists(config['clip_path']):
            print("⚠️  跳过多层测试（没有CLIP权重）")
            return True
            
        model = CLIPEnhanced(config)
        model.eval()
        
        # 测试不同的层配置
        layer_configs = [
            ([-1], "last_layer"),
            ([-2, -1], "last_2_layers"),
            ([-4, -3, -2, -1], "last_4_layers")
        ]
        
        test_image = create_synthetic_image()
        test_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0
        test_tensor = test_tensor.unsqueeze(0)
        
        results = {}
        
        for target_layers, name in layer_configs:
            print(f"   测试配置: {name}")
            
            gradcam = CLIPViTGradCAM(model, target_layers=target_layers)
            try:
                cam = gradcam.generate_cam(test_tensor, target_class=1)
                results[name] = cam[0]
                print(f"   ✅ {name} 成功")
            except Exception as e:
                print(f"   ❌ {name} 失败: {e}")
            finally:
                gradcam.cleanup()
        
        if results:
            # 保存多层对比结果
            fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
            if len(results) == 1:
                axes = [axes]
            
            for i, (name, cam) in enumerate(results.items()):
                axes[i].imshow(cam, cmap='jet')
                axes[i].set_title(f'{name}\nRange: [{cam.min():.2f}, {cam.max():.2f}]')
                axes[i].axis('off')
            
            plt.suptitle('Multi-Layer Attention Comparison', fontsize=16)
            plt.tight_layout()
            plt.savefig('test_results/multilayer_test.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ 多层测试结果保存: test_results/multilayer_test.png")
        
        return True
        
    except Exception as e:
        print(f"❌ 多层测试失败: {e}")
        return False

def test_head_fusion():
    """测试不同的多头融合方法"""
    print("\n🎯 开始多头融合测试...")
    
    try:
        config = create_test_config()
        
        if not os.path.exists(config['clip_path']):
            print("⚠️  跳过多头融合测试（没有CLIP权重）")
            return True
            
        model = CLIPEnhanced(config)
        model.eval()
        
        test_image = create_synthetic_image()
        test_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0
        test_tensor = test_tensor.unsqueeze(0)
        
        fusion_methods = ["mean", "max", "min"]
        results = {}
        
        for method in fusion_methods:
            print(f"   测试融合方法: {method}")
            
            gradcam = CLIPViTGradCAM(model, target_layers=[-2, -1], head_fusion=method)
            try:
                cam = gradcam.generate_cam(test_tensor, target_class=1)
                results[method] = cam[0]
                print(f"   ✅ {method} 成功")
            except Exception as e:
                print(f"   ❌ {method} 失败: {e}")
            finally:
                gradcam.cleanup()
        
        if results:
            # 保存融合方法对比
            fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
            if len(results) == 1:
                axes = [axes]
            
            for i, (method, cam) in enumerate(results.items()):
                axes[i].imshow(cam, cmap='jet')
                axes[i].set_title(f'{method.upper()} Fusion\nRange: [{cam.min():.2f}, {cam.max():.2f}]')
                axes[i].axis('off')
            
            plt.suptitle('Head Fusion Methods Comparison', fontsize=16)
            plt.tight_layout()
            plt.savefig('test_results/head_fusion_test.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ 多头融合测试结果保存: test_results/head_fusion_test.png")
        
        return True
        
    except Exception as e:
        print(f"❌ 多头融合测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始CLIP ViT GradCAM完整测试...")
    print("="*60)
    
    # 运行所有测试
    tests = [
        ("基础功能测试", test_gradcam_basic),
        ("多层attention测试", test_multiple_layers),
        ("多头融合测试", test_head_fusion)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        result = test_func()
        results.append((test_name, result))
        print("-" * 40)
    
    # 总结结果
    print("\n" + "="*60)
    print("📊 测试结果总结:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！GradCAM功能正常。")
        print("📁 查看测试结果: test_results/ 目录")
        print("📖 使用指南: examples/README_attention_visualization.md")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
        print("💡 常见问题:")
        print("   - 确保CLIP模型权重存在")
        print("   - 检查CUDA内存是否足够")
        print("   - 确认所有依赖包已安装")

if __name__ == "__main__":
    main()
