#!/usr/bin/env python3
"""
Test Universal Grad-CAM Functionality

This script is used to test and validate the universal Grad-CAM interface support for different models
"""

import sys
import os
import torch
import yaml

# Add necessary paths
sys.path.append('training')
sys.path.append('training/utils')

from training.utils.universal_gradcam import UniversalGradCAM, load_model_and_create_gradcam

def test_xception_gradcam():
    """Test Xception model Grad-CAM"""
    print("\n🧪 === Test Xception Grad-CAM ===")
    
    config_path = "training/config/detector/xception.yaml"
    weights_path = "training/pretrained/xception_best.pth"
    
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        print(f"❌ Xception files do not exist")
        return False, None
    
    try:
        # Load model and create Grad-CAM
        model, gradcam = load_model_and_create_gradcam(config_path, weights_path, 'xception')
        
        print(f"✅ Xception model loaded successfully")
        return True, gradcam
        
    except Exception as e:
        print(f"❌ Xception test failed: {e}")
        return False, None

def test_clip_enhanced_gradcam():
    """Test CLIP Enhanced model Grad-CAM"""
    print("\n🧪 === Test CLIP Enhanced Grad-CAM ===")
    
    config_path = "training/config/detector/clip_enhanced.yaml"
    weights_path = "weights/clip_enhanced_best.pth"
    
    if not os.path.exists(config_path):
        print(f"❌ CLIP config file does not exist: {config_path}")
        return False, None
        
    if not os.path.exists(weights_path):
        print(f"⚠️ CLIP weight file does not exist: {weights_path}")
        print("Trying to find other possible weight files...")
        
        # Find possible weight files
        possible_weights = [
            "weights/clip_best.pth",
            "training/pretrained/clip_best.pth",
            "logs/clip_enhanced/best.pth"
        ]
        
        weights_path = None
        for path in possible_weights:
            if os.path.exists(path):
                weights_path = path
                break
        
        if weights_path is None:
            print("❌ Cannot find CLIP weight files")
            return False, None
    
    try:
        # Load model and create Grad-CAM
        model, gradcam = load_model_and_create_gradcam(config_path, weights_path, 'clip_enhanced')
        
        print(f"✅ CLIP Enhanced model loaded successfully")
        return True, gradcam
        
    except Exception as e:
        print(f"❌ CLIP Enhanced test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_auto_detection():
    """Test automatic model type detection"""
    print("\n🔍 === Test Automatic Model Type Detection ===")
    
    results = []
    
    # Test Xception auto detection
    config_path = "training/config/detector/xception.yaml"
    weights_path = "training/pretrained/xception_best.pth"
    
    if os.path.exists(config_path) and os.path.exists(weights_path):
        try:
            model, gradcam = load_model_and_create_gradcam(config_path, weights_path, 'auto')
            detected_type = gradcam.model_type
            expected_type = 'xception'
            
            if detected_type == expected_type:
                print(f"✅ Xception auto detection successful: {detected_type}")
                results.append(True)
            else:
                print(f"❌ Xception auto detection failed: expected {expected_type}, got {detected_type}")
                results.append(False)
                
        except Exception as e:
            print(f"❌ Xception auto detection exception: {e}")
            results.append(False)
    else:
        print("⚠️ Xception files do not exist, skipping auto detection test")
    
    return all(results) if results else False

def test_single_image():
    """Test single image Grad-CAM generation"""
    print("\n🖼️  === Test Single Image Grad-CAM ===")
    
    # Find test images
    test_image_paths = [
        "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/Celeb-DF-v2/Celeb-real/frames/id0_0000/000.png",
        "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/Celeb-DF-v2/Celeb-real/frames/id0_0000/015.png"
    ]
    
    test_image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if test_image_path is None:
        print("⚠️ Cannot find test images")
        return False
    
    print(f"🖼️  Using test image: {test_image_path}")
    
    # Test Xception
    success_xception, gradcam_xception = test_xception_gradcam()
    
    if success_xception:
        try:
            print("\n🔥 Generating Xception Grad-CAM...")
            heatmap, original_image = gradcam_xception.generate_gradcam(test_image_path, target_class=1)
            print(f"✅ Xception heatmap generated successfully: shape {heatmap.shape}, range [{heatmap.min():.3f}, {heatmap.max():.3f}]")
            
            # Save visualization results
            save_path = "./universal_gradcam_results/xception_single_test.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            gradcam_xception.visualize_gradcam(test_image_path, target_class=1, save_path=save_path)
            
        except Exception as e:
            print(f"❌ Xception image test failed: {e}")
            success_xception = False
    
    return success_xception

def test_model_comparison():
    """Test model comparison functionality"""
    print("\n⚖️  === Test Model Comparison Functionality ===")
    
    # Test images
    test_image_paths = [
        "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/Celeb-DF-v2/Celeb-real/frames/id0_0000/000.png"
    ]
    
    test_image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if test_image_path is None:
        print("⚠️ Cannot find test images")
        return False
    
    # Load two models
    success_xception, gradcam_xception = test_xception_gradcam()
    success_clip, gradcam_clip = test_clip_enhanced_gradcam()
    
    if success_xception and success_clip:
        try:
            print("🔥 Performing model comparison...")
            gradcam_xception.compare_models(
                test_image_path, 
                gradcam_clip, 
                target_class=1, 
                save_dir='./universal_gradcam_results'
            )
            print("✅ Model comparison complete")
            return True
            
        except Exception as e:
            print(f"❌ Model comparison failed: {e}")
            return False
    elif success_xception:
        print("⚠️ Only Xception available, cannot perform comparison")
        return False
    elif success_clip:
        print("⚠️ Only CLIP available, cannot perform comparison") 
        return False
    else:
        print("❌ No available models for comparison")
        return False

def test_unified_interface():
    """Test unified interface convenience"""
    print("\n🔧 === Test Unified Interface Convenience ===")
    
    try:
        # Test direct creation
        from training.utils.universal_gradcam import create_gradcam
        
        # Create a dummy model to test interface
        class DummyXception:
            def __init__(self):
                self.backbone = type('obj', (object,), {'conv4': torch.nn.Conv2d(1, 1, 1)})()
            def __call__(self, data_dict, inference=False):
                return {'cls': torch.randn(1, 2)}
            def to(self, device):
                return self
            def eval(self):
                return self
        
        dummy_model = DummyXception()
        gradcam = create_gradcam(dummy_model, model_type='xception')
        
        print(f"✅ Unified interface creation successful: {gradcam.model_type}")
        return True
        
    except Exception as e:
        print(f"❌ Unified interface test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Universal Grad-CAM Tests")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Run tests
    tests = [
        ("Unified Interface Test", test_unified_interface),
        ("Auto Detection Test", test_auto_detection),
        ("Xception Functionality Test", lambda: test_xception_gradcam()[0]),
        ("CLIP Enhanced Functionality Test", lambda: test_clip_enhanced_gradcam()[0]),
        ("Single Image Test", test_single_image),
        ("Model Comparison Test", test_model_comparison),
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
    print(f"\n{'='*25} Test Results Summary {'='*25}")
    for test_name, result in results:
        status = "✅ Passed" if result else "❌ Failed"
        print(f"{test_name}: {status}")
    
    # Overall assessment
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n📊 Overall result: {passed}/{total} tests passed")
    
    if passed >= total * 0.7:  # 70% or more pass considered success
        print("🎉 Most tests passed! Universal Grad-CAM functionality is basically working")
    else:
        print("⚠️  Most tests failed, please check specific error messages")
    
    # Give usage suggestions
    print(f"\n{'='*25} Usage Suggestions {'='*25}")
    print("✨ Universal Grad-CAM usage examples:")
    print("```python")
    print("from training.utils.universal_gradcam import UniversalGradCAM, load_model_and_create_gradcam")
    print("")
    print("# Method 1: Direct use with loaded model")
    print("gradcam = UniversalGradCAM(your_model, model_type='auto')")
    print("heatmap, image = gradcam.generate_gradcam('path/to/image.jpg')")
    print("")
    print("# Method 2: Load model from config file")
    print("model, gradcam = load_model_and_create_gradcam('config.yaml', 'weights.pth')")
    print("gradcam.visualize_gradcam('path/to/image.jpg', save_path='result.png')")
    print("```")

if __name__ == "__main__":
    main()
