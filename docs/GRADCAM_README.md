# 🔥 Unified Grad-CAM Solution

This is a unified Grad-CAM implementation developed for the DeepfakeBench project, supporting visualization analysis for multiple deep learning models.

## 📋 Features

- ✅ **Multi-model Support**: Xception, CLIP Enhanced
- ✅ **Automatic Model Recognition**: Intelligent detection of model types
- ✅ **Unified Interface**: One API for all supported models
- ✅ **Multiple Visualization Methods**: Input gradients, standard Grad-CAM
- ✅ **Inter-model Comparison**: Intuitive comparison of attention points between different models
- ✅ **Easy to Use**: Provides convenience functions and complete examples
- ✅ **Extensible**: Easy to add support for new models

## 📁 File Structure

```
training/utils/
├── xception_gradcam.py      # Xception-specific Grad-CAM implementation
└── universal_gradcam.py     # Unified Grad-CAM interface

tests/
├── test_xception_gradcam.py    # Xception test script
├── test_universal_gradcam.py   # Unified interface test script
└── example_gradcam_usage.py    # Usage example script
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from training.utils.universal_gradcam import UniversalGradCAM, load_model_and_create_gradcam

# Method 1: Load model from config file
model, gradcam = load_model_and_create_gradcam(
    'training/config/detector/xception.yaml', 
    'training/pretrained/xception_best.pth'
)

# Generate heatmap
heatmap, image = gradcam.generate_gradcam('path/to/image.jpg', target_class=1)

# Visualize results
gradcam.visualize_gradcam('path/to/image.jpg', save_path='result.png')
```

### 2. Manual Creation

```python
from training.utils.universal_gradcam import UniversalGradCAM

# Assuming you have a trained model
gradcam = UniversalGradCAM(your_model, model_type='auto')  # Auto detect
# Or specify type
gradcam = UniversalGradCAM(your_model, model_type='xception')
```

### 3. Model Comparison

```python
# Compare two different models
gradcam1 = UniversalGradCAM(xception_model, 'xception')
gradcam2 = UniversalGradCAM(clip_model, 'clip_enhanced')

gradcam1.compare_models('image.jpg', gradcam2, save_dir='./comparison')
```

## 🧪 Running Tests

### Complete Test Suite
```bash
python test_universal_gradcam.py
```

### Xception-specific Tests
```bash
python test_xception_gradcam.py
```

### Usage Examples
```bash
# Run all examples
python example_gradcam_usage.py

# Run only Xception example
python example_gradcam_usage.py --example xception

# Run only model comparison example
python example_gradcam_usage.py --example comparison
```

## 📊 Supported Models

### ✅ Xception
- **Config File**: `training/config/detector/xception.yaml`
- **Weight File**: `training/pretrained/xception_best.pth`
- **Supported Methods**: Input gradient, standard Grad-CAM
- **Target Layer**: `conv4` (configurable)

### ✅ CLIP Enhanced
- **Config File**: `training/config/detector/clip_enhanced.yaml`
- **Weight File**: `weights/clip_enhanced_best.pth` (or other locations)
- **Supported Methods**: Input gradient (recommended due to PEFT wrapper)
- **Special Handling**: Automatically handles PEFT-wrapped attention layers

## 🎯 Visualization Methods

### 1. Input Gradient Visualization (Recommended)
- **Principle**: Computes gradients of output with respect to input pixels
- **Advantages**: Applicable to all models, simple and reliable implementation
- **Disadvantages**: May contain noise

### 2. Standard Grad-CAM
- **Principle**: Based on feature maps and gradients from the last convolutional layer
- **Advantages**: Classic method with strong interpretability
- **Disadvantages**: Only applicable to CNN architectures

## 📈 Test Results

Latest test results (in GPU environment):

```
========================= Test Results Summary =========================
Unified Interface Test: ✅ Passed
Auto Detection Test: ✅ Passed
Xception Functionality Test: ✅ Passed
CLIP Enhanced Functionality Test: ❌ Failed (weight file not found)
Single Image Test: ✅ Passed
Model Comparison Test: ❌ Failed (depends on CLIP)

📊 Overall result: 4/6 tests passed
```

## 💡 Usage Recommendations

### Recommended Settings

1. **Target Classes**:
   - `target_class=0`: Analyze model's attention on "real" samples
   - `target_class=1`: Analyze model's attention on "fake" samples

2. **Visualization Methods**:
   - Prefer `method='input_grad'` (more universal, more stable)
   - CNN models can try `method='standard'`

3. **Color Mapping**:
   - Use `'Reds'` color map (white=low attention, red=high attention)
   - Avoid `'jet'` (dark blue is hard to perceive)

### Performance Optimization

- Use GPU acceleration: `device='cuda'`
- Batch process multiple images
- Set appropriate image resolution

## 🔧 Custom Extensions

### Adding New Model Support

1. Inherit from `BaseGradCAM` class
2. Implement `generate_gradcam` and `preprocess_image` methods
3. Add model type detection in `UniversalGradCAM`

```python
class YourModelGradCAM(BaseGradCAM):
    def generate_gradcam(self, data_dict, target_class=1, method='input_grad'):
        # Implement your Grad-CAM logic
        return self._generate_input_gradient(data_dict, target_class)
    
    def preprocess_image(self, image_path):
        # Implement image preprocessing
        return data_dict, original_image
```

### Adding New Visualization Methods

Add new methods in the corresponding GradCAM class:

```python
def _generate_your_method(self, data_dict, target_class):
    # Implement your visualization method
    pass
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Type Detection Failed**
   - Manually specify `model_type='xception'` or `model_type='clip_enhanced'`

2. **Weight Files Not Found**
   - Check if file paths are correct
   - Ensure model weights are properly downloaded

3. **CUDA Out of Memory**
   - Use `device='cpu'`
   - Reduce input image resolution

4. **Unsatisfactory Visualization Results**
   - Try different `target_class`
   - Adjust contrast enhancement parameters
   - Use different visualization methods

### Error Logging

If you encounter issues, please check detailed error messages:

```python
import traceback
try:
    gradcam.visualize_gradcam('image.jpg')
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
```

## 📄 License

This project follows the same license as DeepfakeBench.

## 🙏 Acknowledgments

- DeepfakeBench project team
- Original Grad-CAM paper authors
- Contributors to PyTorch and related open-source libraries

---

**🎉 Now you can easily perform Grad-CAM visualization analysis on Xception and CLIP models!**
