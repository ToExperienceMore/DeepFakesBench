# CLIP ViT Attention 可视化指南

## 🎯 功能概述

这个工具实现了**梯度加权attention可视化**，专门用于分析你的CLIP deepfake检测模型的决策过程。

### 核心原理
```
输入图像 → CLIP ViT → Multi-Head Attention → 梯度加权 → 可视化热力图
```

## 🚀 快速开始

### 1. 准备测试图像
```bash
# 将你要分析的图像放到项目根目录
cp your_test_image.jpg /root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/test_image.jpg
```

### 2. 运行可视化
```bash
cd /root/autodl-tmp/benchmark_deepfakes/DeepfakeBench
python examples/clip_attention_visualization.py
```

### 3. 查看结果
结果会保存在 `attention_results/` 目录下：
- `*_real_attention.png` - Real类别的attention
- `*_fake_attention.png` - Fake类别的attention  
- `*_comparison.png` - 对比分析图

## 📋 详细使用说明

### 修改配置路径

在 `examples/clip_attention_visualization.py` 的 `main()` 函数中修改：

```python
def main():
    # 🔧 修改这些路径
    model_config_path = "training/config/detector/clip_enhanced.yaml"  # 你的配置文件
    weights_path = "path/to/your/model_weights.pth"  # 你的模型权重
    test_image_path = "test_image.jpg"  # 测试图像
```

### 自定义分析参数

```python
# 在analyze_single_image函数中修改
gradcam = CLIPViTGradCAM(
    model, 
    target_layers=[-4, -3, -2, -1],  # 分析的层：最后4层
    head_fusion="mean"               # 多头融合方式：mean/max/min
)
```

## 🎨 可视化结果解读

### 1. Real Class Attention
- **高亮区域**：模型认为"看起来真实"的区域
- **暗色区域**：对Real分类贡献较小的区域

### 2. Fake Class Attention  
- **高亮区域**：模型检测到的"伪造痕迹"区域
- **常见模式**：面部边缘、眼部、嘴部异常区域

### 3. 对比分析
- **差异区域**：Real和Fake attention的不同关注点
- **重叠区域**：两类共同关注的重要特征

## 🔧 高级功能

### 1. 多层分析
运行时选择"y"进行多层分析，查看不同层的attention模式：
- `last_layer` - 最后一层（最具体的特征）
- `last_4_layers` - 最后4层（推荐）
- `all_layers` - 所有层（最全面）

### 2. 批量分析
修改脚本支持批量处理多张图像：

```python
# 在analyze_single_image基础上扩展
image_list = ["image1.jpg", "image2.jpg", "image3.jpg"]
for img_path in image_list:
    analyze_single_image(config_path, weights_path, img_path)
```

## 🛠️ 故障排除

### 常见问题

1. **模型加载失败**
   ```
   ❌ 权重文件不存在
   ```
   - 检查 `weights_path` 是否正确
   - 确保权重文件存在

2. **CUDA内存不足**
   ```python
   # 在脚本开头添加
   import torch
   torch.cuda.empty_cache()
   ```

3. **attention权重为空**
   ```
   No attention weights found
   ```
   - 检查模型是否正确加载
   - 确保使用的是CLIP vision model

4. **图像预处理错误**
   ```python
   # 确保图像格式正确
   image = Image.open(image_path).convert('RGB')
   ```

### 调试模式

添加调试信息：
```python
# 在CLIPViTGradCAM类中添加
def _debug_info(self):
    print(f"Target layers: {self.target_layers}")
    print(f"Attention weights keys: {list(self.attention_weights.keys())}")
    print(f"Attention gradients keys: {list(self.attention_gradients.keys())}")
```

## 📊 结果分析建议

### 1. 真实图像的特征
- attention通常分布在**面部结构**上
- 眼部、鼻部、嘴部有**均匀关注**
- 边缘区域attention较**弱**

### 2. 伪造图像的特征  
- attention集中在**异常区域**
- **边界线**、**不自然过渡**处高亮
- 可能在**背景融合**处有异常

### 3. 模型可解释性
- **高置信度区域**：attention强且集中
- **边界情况**：attention分散或矛盾
- **失败案例**：attention与人类直觉不符

## 📚 扩展阅读

- [Attention Rollout原理](https://arxiv.org/abs/2005.00928)
- [ViT可解释性研究](https://arxiv.org/abs/2010.11929)  
- [Transformer Attention可视化](https://arxiv.org/abs/1906.04341)

## 🤝 贡献与反馈

如果你发现bug或有改进建议，请：
1. 检查现有issues
2. 提供详细的错误信息和复现步骤
3. 分享你的使用场景和结果

---

**开始你的attention可视化之旅吧！** 🚀
