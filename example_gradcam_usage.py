#!/usr/bin/env python3
"""
Grad-CAM Usage Examples

Demonstrates how to use the unified Grad-CAM interface for visualization analysis of Xception and CLIP models
"""

import sys
import os
import argparse

# Add necessary paths
sys.path.append('training')
sys.path.append('training/utils')

from training.utils.universal_gradcam import UniversalGradCAM, load_model_and_create_gradcam

def example_xception_gradcam():
    """Xception Grad-CAM example"""
    print("🔥 === Xception Grad-CAM Example ===")
    
    # Configuration paths
    config_path = "training/config/detector/xception.yaml"
    weights_path = "training/pretrained/xception_best.pth"
    
    # Check if files exist
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        print(f"❌ Xception files do not exist, please check paths")
        return
    
    try:
        # Method 1: Use convenience function to load model and create Grad-CAM
        print("📂 Loading Xception model...")
        model, gradcam = load_model_and_create_gradcam(config_path, weights_path, 'auto')
        
        # Find test images
        #test_image_paths = [
        #     "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/Celeb-DF-v2/Celeb-real/frames/id0_0000/000.png",
        #     "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/Celeb-DF-v2/Celeb-real/frames/id0_0000/015.png"
        # ]

        test_image_paths = [
            "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/000_003/000.png",
            "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/original_sequences/youtube/c23/frames/000/000.png"
        ]
        
        test_image_path = None
        for path in test_image_paths:
            if os.path.exists(path):
                test_image_path = path
                break
        
        if test_image_path is None:
            print("⚠️  Cannot find test images")
            return
        
        print(f"🖼️  Analyzing image: {os.path.basename(test_image_path)}")
        
        # Generate and visualize Grad-CAM
        print("🔥 Generating Grad-CAM heatmaps...")
        
        # Analyze Real class (0) and Fake class (1) separately
        for target_class, class_name in [(0, 'Real'), (1, 'Fake')]:
            print(f"\n📊 Analyzing target class: {class_name} (class {target_class})")
            
            # Generate both heatmaps
            input_heatmap, original_image = gradcam.generate_gradcam(
                test_image_path, 
                target_class=target_class, 
                method='input_grad'
            )
            
            try:
                standard_heatmap = gradcam.gradcam.generate_gradcam(
                    gradcam.gradcam.preprocess_image(test_image_path)[0], 
                    target_class, 
                    'standard'
                )
            except:
                standard_heatmap = None
            
            print(f"✅ Input gradient: shape {input_heatmap.shape}, range [{input_heatmap.min():.3f}, {input_heatmap.max():.3f}]")
            if standard_heatmap is not None:
                print(f"✅ Standard CAM: shape {standard_heatmap.shape}, range [{standard_heatmap.min():.3f}, {standard_heatmap.max():.3f}]")
            
            # Create simple comparison visualization
            import matplotlib.pyplot as plt
            import cv2
            import numpy as np
            img_array = np.array(original_image)
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Original
            axes[0].imshow(img_array)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # Input gradient overlay
            h, w = img_array.shape[:2]
            input_resized = cv2.resize(input_heatmap, (w, h))
            input_colored = plt.cm.Reds(input_resized)[:, :, :3] * 255
            input_overlay = cv2.addWeighted(img_array, 0.7, input_colored.astype(np.uint8), 0.3, 0)
            axes[1].imshow(input_overlay)
            axes[1].set_title('Input Gradient')
            axes[1].axis('off')
            
            # Standard CAM pure
            if standard_heatmap is not None:
                standard_resized = cv2.resize(standard_heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
                axes[2].imshow(standard_resized, cmap='Reds')
                axes[2].set_title('Standard CAM')
                axes[2].axis('off')
                
                # Standard CAM overlay - simple and direct
                standard_colored = plt.cm.Reds(standard_resized)[:, :, :3] * 255
                standard_overlay = cv2.addWeighted(img_array, 0.7, standard_colored.astype(np.uint8), 0.3, 0)
                axes[3].imshow(standard_overlay)
                axes[3].set_title('Standard CAM Overlay')
                axes[3].axis('off')
            else:
                axes[2].text(0.5, 0.5, 'Standard CAM\nNot Available', ha='center', va='center')
                axes[2].axis('off')
                axes[3].text(0.5, 0.5, 'Standard CAM\nOverlay\nNot Available', ha='center', va='center')
                axes[3].axis('off')
            
            plt.tight_layout()
            save_path = f"./gradcam_examples/xception_{class_name.lower()}_comparison.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved to: {save_path}")
            plt.show()
        
        print("🎉 Xception Grad-CAM analysis complete!")
        
    except Exception as e:
        print(f"❌ Xception example failed: {e}")
        import traceback
        traceback.print_exc()

def example_clip_gradcam():
    """CLIP Enhanced Grad-CAM example"""
    print("\n🔥 === CLIP Enhanced Grad-CAM Example ===")
    
    # Configuration paths
    config_path = "training/config/detector/clip_enhanced.yaml"
    
    # Try to find weight files
    possible_weights = [
        "weights/clip_enhanced_best.pth",
        "weights/clip_best.pth",
        "training/pretrained/clip_best.pth",
        "logs/clip_enhanced/best.pth"
    ]
    
    weights_path = None
    for path in possible_weights:
        if os.path.exists(path):
            weights_path = path
            break
    
    if not os.path.exists(config_path):
        print(f"❌ CLIP config file does not exist: {config_path}")
        return
        
    if weights_path is None:
        print("❌ Cannot find CLIP weight files")
        print("Please ensure weight files exist in one of the following paths:")
        for path in possible_weights:
            print(f"  - {path}")
        return
    
    try:
        print("📂 Loading CLIP Enhanced model...")
        model, gradcam = load_model_and_create_gradcam(config_path, weights_path, 'auto')
        
        # Find test images
        test_image_paths = [
            "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/Celeb-DF-v2/Celeb-real/frames/id0_0000/000.png"
        ]
        
        test_image_path = None
        for path in test_image_paths:
            if os.path.exists(path):
                test_image_path = path
                break
        
        if test_image_path is None:
            print("⚠️  Cannot find test images")
            return
        
        print(f"🖼️  Analyzing image: {os.path.basename(test_image_path)}")
        
        # Generate and visualize Grad-CAM
        print("🔥 Generating CLIP Grad-CAM heatmaps...")
        
        # Analyze Fake class (1)
        heatmap, original_image = gradcam.generate_gradcam(
            test_image_path, 
            target_class=1, 
            method='input_grad'
        )
        
        print(f"✅ CLIP heatmap generated successfully: shape {heatmap.shape}, value range [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        
        # Save visualization results
        save_path = "./gradcam_examples/clip_enhanced_fake_class.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        gradcam.visualize_gradcam(
            test_image_path, 
            target_class=1,
            save_path=save_path
        )
        
        print("🎉 CLIP Enhanced Grad-CAM analysis complete!")
        
    except Exception as e:
        print(f"❌ CLIP example failed: {e}")
        import traceback
        traceback.print_exc()

def example_model_comparison():
    """Model comparison example"""
    print("\n⚖️  === Model Comparison Example ===")
    
    # Xception configuration
    xception_config = "training/config/detector/xception.yaml"
    xception_weights = "training/pretrained/xception_best.pth"
    
    # CLIP configuration  
    clip_config = "training/config/detector/clip_enhanced.yaml"
    clip_weights_candidates = [
        "weights/clip_enhanced_best.pth",
        "weights/clip_best.pth",
        "logs/clip_enhanced/best.pth"
    ]
    
    clip_weights = None
    for path in clip_weights_candidates:
        if os.path.exists(path):
            clip_weights = path
            break
    
    # Check file existence
    if not (os.path.exists(xception_config) and os.path.exists(xception_weights)):
        print("❌ Xception files do not exist")
        return
        
    if not (os.path.exists(clip_config) and clip_weights):
        print("❌ CLIP files do not exist or are incomplete")
        return
    
    try:
        # Load both models
        print("📂 Loading Xception model...")
        xception_model, xception_gradcam = load_model_and_create_gradcam(
            xception_config, xception_weights, 'xception'
        )
        
        print("📂 Loading CLIP Enhanced model...")
        clip_model, clip_gradcam = load_model_and_create_gradcam(
            clip_config, clip_weights, 'clip_enhanced'
        )
        
        # Find test image
        test_image_path = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/Celeb-DF-v2/Celeb-real/frames/id0_0000/000.png"
        
        if not os.path.exists(test_image_path):
            print("⚠️  Cannot find test images")
            return
        
        print(f"🖼️  Comparative analysis image: {os.path.basename(test_image_path)}")
        
        # Perform model comparison
        print("⚖️  Performing inter-model comparison...")
        xception_gradcam.compare_models(
            test_image_path,
            clip_gradcam,
            target_class=1,
            save_dir='./gradcam_examples'
        )
        
        print("🎉 Model comparison complete!")
        
    except Exception as e:
        print(f"❌ Model comparison failed: {e}")
        import traceback
        traceback.print_exc()

def example_manual_usage():
    """Manual usage example"""
    print("\n🔧 === Manual Usage Example ===")
    
    try:
        # Manually load model (using Xception as example)
        import yaml
        from detectors import DETECTOR
        
        config_path = "training/config/detector/xception.yaml"
        weights_path = "training/pretrained/xception_best.pth"
        
        if not (os.path.exists(config_path) and os.path.exists(weights_path)):
            print("❌ Files do not exist")
            return
        
        # Read configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['pretrained'] = weights_path
        
        # Create model
        print("📂 Manually creating Xception model...")
        detector_class = DETECTOR[config['model_name']]
        model = detector_class(config)
        
        # Create Grad-CAM object
        print("🔧 Creating Grad-CAM object...")
        gradcam = UniversalGradCAM(model, model_type='auto')
        
        print(f"✅ Manual creation successful: {gradcam.model_type}")
        
        # Test basic functionality
        test_image_path = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/Celeb-DF-v2/Celeb-real/frames/id0_0000/000.png"
        
        if os.path.exists(test_image_path):
            heatmap, image = gradcam.generate_gradcam(test_image_path, target_class=1)
            print(f"✅ Manual method heatmap generation successful: {heatmap.shape}")
        
        print("🎉 Manual usage example complete!")
        
    except Exception as e:
        print(f"❌ Manual usage example failed: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Grad-CAM Usage Examples')
    parser.add_argument('--example', choices=['xception', 'clip', 'comparison', 'manual', 'all'], 
                       default='all', help='Select which example to run')
    
    args = parser.parse_args()
    
    print("🚀 Grad-CAM Usage Examples")
    print("=" * 50)
    
    if args.example in ['xception', 'all']:
        example_xception_gradcam()
    
    if args.example in ['clip', 'all']:
        example_clip_gradcam()
    
    if args.example in ['comparison', 'all']:
        example_model_comparison()
    
    if args.example in ['manual', 'all']:
        example_manual_usage()
    
    print("\n✨ Example execution complete!")
    print("\n📁 Generated files:")
    if os.path.exists('./gradcam_examples'):
        for file in os.listdir('./gradcam_examples'):
            print(f"  - ./gradcam_examples/{file}")

if __name__ == "__main__":
    main()
