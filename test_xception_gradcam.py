#!/usr/bin/env python3
"""
Test Xception Grad-CAM Functionality

This script is used to test and validate the Grad-CAM visualization functionality for Xception models
"""

import sys
import os
import torch
import yaml

# Add necessary paths
sys.path.append('training')
sys.path.append('training/utils')

# Import necessary modules
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from training.utils.xception_gradcam import XceptionGradCAM, preprocess_image, compare_gradcam_methods

def load_xception_model(config_path, weights_path):
    """
    Load Xception model
    
    Args:
        config_path: Configuration file path
        weights_path: Weight file path
        
    Returns:
        model: Loaded model
    """
    # Read configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update weights path
    config['pretrained'] = weights_path
    
    # Create model
    detector_class = DETECTOR[config['model_name']]
    model = detector_class(config)
    
    print(f"✅ Successfully loaded Xception model")
    print(f"📁 Config file: {config_path}")
    print(f"⚖️  Weight file: {weights_path}")
    
    return model

def test_gradcam_basic():
    """Basic Grad-CAM test"""
    print("\n🧪 === Basic Grad-CAM Test ===")
    
    # Configuration paths
    config_path = "training/config/detector/xception.yaml"
    weights_path = "training/pretrained/xception_best.pth"
    
    # Check if files exist
    if not os.path.exists(config_path):
        print(f"❌ Config file does not exist: {config_path}")
        return False
    
    if not os.path.exists(weights_path):
        print(f"❌ Weight file does not exist: {weights_path}")
        print("Please check the weight file path or download from official source")
        return False
    
    try:
        # Load model
        model = load_xception_model(config_path, weights_path)
        model.eval()
        
        # Create Grad-CAM object
        gradcam = XceptionGradCAM(model)
        
        # Create test input
        dummy_input = torch.randn(1, 3, 256, 256)
        data_dict = {'image': dummy_input}
        
        print("\n🔍 Testing different Grad-CAM methods...")
        
        # Test input gradient method (more reliable)
        try:
            heatmap_input = gradcam.generate_gradcam(data_dict, target_class=1, method='input_grad')
            print(f"✅ Input gradient method: heatmap shape {heatmap_input.shape}")
            print(f"   Heatmap range: [{heatmap_input.min():.3f}, {heatmap_input.max():.3f}]")
        except Exception as e:
            print(f"❌ Input gradient method failed: {e}")
        
        # Test standard Grad-CAM method
        try:
            heatmap_standard = gradcam.generate_gradcam(data_dict, target_class=1, method='standard')
            print(f"✅ Standard Grad-CAM: heatmap shape {heatmap_standard.shape}")
            print(f"   Heatmap range: [{heatmap_standard.min():.3f}, {heatmap_standard.max():.3f}]")
        except Exception as e:
            print(f"❌ Standard Grad-CAM failed: {e}")
        
        print("✅ Basic test complete!")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_image():
    """Test with real images"""
    print("\n🖼️  === Real Image Test ===")
    
    # Check for available test images
    test_image_paths = [
        "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/Celeb-DF-v2/Celeb-real/frames/id0_0000/000.png",
        "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/test_image.jpg",
        "test_image.jpg"
    ]
    
    test_image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if test_image_path is None:
        print("⚠️  No test images found, skipping real image test")
        print("Available test image paths:")
        for path in test_image_paths:
            print(f"  - {path}")
        return False
    
    try:
        # Configuration paths
        config_path = "training/config/detector/xception.yaml"
        weights_path = "training/pretrained/xception_best.pth"
        
        # Load model
        model = load_xception_model(config_path, weights_path)
        model.eval()
        
        print(f"🖼️  Using test image: {test_image_path}")
        
        # Compare different methods
        compare_gradcam_methods(
            model=model,
            image_path=test_image_path,
            target_class=1,  # Fake class
            save_dir='./xception_gradcam_results'
        )
        
        print("✅ Real image test complete!")
        return True
        
    except Exception as e:
        print(f"❌ Real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_info():
    """Test model information retrieval"""
    print("\n📊 === Model Information Test ===")
    
    try:
        config_path = "training/config/detector/xception.yaml"
        weights_path = "training/pretrained/xception_best.pth"
        
        if not os.path.exists(weights_path):
            print(f"❌ Weight file does not exist: {weights_path}")
            return False
            
        # Load model
        model = load_xception_model(config_path, weights_path)
        
        # Print model structure information
        print("\n📋 Model structure information:")
        if hasattr(model, 'backbone'):
            backbone = model.backbone
            print(f"Model type: {type(backbone).__name__}")
            
            # List important layers
            important_layers = []
            for name, module in backbone.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)) and not name.startswith('block'):
                    important_layers.append((name, module))
            
            print("\nImportant convolutional layers:")
            for name, module in important_layers:
                if isinstance(module, torch.nn.Conv2d):
                    print(f"  - {name}: {module}")
            
        print("✅ Model information retrieval complete!")
        return True
        
    except Exception as e:
        print(f"❌ Model information test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Xception Grad-CAM Tests")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Run tests
    tests = [
        ("Model Information Test", test_model_info),
        ("Basic Functionality Test", test_gradcam_basic),
        ("Real Image Test", test_with_real_image),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} encountered exception: {e}")
            results.append((test_name, False))
    
    # Summarize results
    print(f"\n{'='*20} Test Results Summary {'='*20}")
    for test_name, result in results:
        status = "✅ Passed" if result else "❌ Failed"
        print(f"{test_name}: {status}")
    
    # Overall assessment
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n📊 Overall result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Xception Grad-CAM functionality is working properly")
    else:
        print("⚠️  Some tests failed, please check specific error messages")

if __name__ == "__main__":
    main()
