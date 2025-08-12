#!/usr/bin/env python3
"""
Grad-CAM Usage Examples

Demonstrates how to use the unified Grad-CAM interface for visualization analysis of Xception and CLIP models
"""

import sys
import os
import argparse

# Add necessary paths
sys.path.append('/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/training')
sys.path.append('/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench')
sys.path.append('/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/training/utils')

from training.utils.universal_gradcam import UniversalGradCAM, load_model_and_create_gradcam
import torch
import numpy as np

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
        
        # Test different layer resolutions
        print("\n🔍 Testing feature map resolutions for different layers:")
        test_input = torch.randn(1, 3, 256, 256).to(next(model.parameters()).device)
        
        # Test some intermediate layers
        test_layers = ['conv1', 'conv2', 'block3', 'block12', 'conv3', 'conv4']
        for layer_name in test_layers:
            if hasattr(model.backbone, layer_name):
                try:
                    # Create a temporary forward hook
                    layer = getattr(model.backbone, layer_name)
                    activation = None
                    
                    def temp_hook(module, input, output):
                        nonlocal activation
                        activation = output
                    
                    handle = layer.register_forward_hook(temp_hook)
                    
                    with torch.no_grad():
                        _ = model.backbone.features(test_input)
                    
                    if activation is not None:
                        print(f"  {layer_name}: {activation.shape[2:]} (channels: {activation.shape[1]})")
                    
                    handle.remove()
                except Exception as e:
                    print(f"  {layer_name}: Error - {e}")
        print("="*60)
        
        # Find test images
        #test_image_paths = [
        #     "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/Celeb-DF-v2/Celeb-real/frames/id0_0000/000.png",
        #     "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/Celeb-DF-v2/Celeb-real/frames/id0_0000/015.png"
        # ]

        # Image paths and corresponding labels (first is fake, second is real)
        test_image_info = [
            {
                "path": "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/000_003/000.png",
                "label": "Fake",
                "target_class": 1,
                "description": "Fake image"
            },
            {
                "path": "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/original_sequences/youtube/c23/frames/000/000.png", 
                "label": "Real",
                "target_class": 0,
                "description": "Real image"
            }
        ]

        #/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/manipulated_sequences/FaceShifter/c23/frames/000_003/000.png
        
        # Check if images exist
        available_images = []
        for img_info in test_image_info:
            if os.path.exists(img_info["path"]):
                available_images.append(img_info)
            else:
                print(f"⚠️  Image not found: {img_info['path']}")
        
        if not available_images:
            print("❌ No available test images found")
            return
        
        print(f"📊 Found {len(available_images)} available images for analysis")
        
        # Perform Grad-CAM analysis for each image
        for i, img_info in enumerate(available_images, 1):
            print(f"\n{'='*60}")
            print(f"🖼️  Analyzing image {i}/{len(available_images)}: {os.path.basename(img_info['path'])}")
            print(f"📋 Image type: {img_info['description']}")
            print(f"🎯 Target class: {img_info['label']} (class {img_info['target_class']})")
            print('='*60)
            
            test_image_path = img_info["path"]
            target_class = img_info["target_class"] 
            class_name = img_info["label"]
            
            # First, get model prediction
            print("🔍 Getting model prediction...")
            try:
                # Load and preprocess image exactly like the dataset
                import cv2
                import numpy as np
                from PIL import Image
                import torchvision.transforms as transforms
                
                # Load image using the same method as dataset.load_rgb()
                img = cv2.imread(test_image_path)
                if img is None:
                    raise ValueError(f'Loaded image is None: {test_image_path}')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                size = 256  # Same as config['resolution']
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
                pil_image = Image.fromarray(np.array(img, dtype=np.uint8))
                
                # Apply the same preprocessing as dataset
                transform = transforms.Compose([
                    transforms.ToTensor(),  # No additional resize needed
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                       std=[0.5, 0.5, 0.5])
                ])
                
                # Prepare input
                input_tensor = transform(pil_image).unsqueeze(0).to(next(model.parameters()).device)
                
                # Debug: Print tensor stats to verify different inputs
                print(f"🔍 Input tensor stats: mean={input_tensor.mean().item():.6f}, std={input_tensor.std().item():.6f}")
                print(f"🔍 Input tensor range: [{input_tensor.min().item():.6f}, {input_tensor.max().item():.6f}]")
                print(f"🔍 Image path: {test_image_path}")
                
                # Clear any potential cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Get model prediction
                model.eval()
                with torch.no_grad():
                    if hasattr(model, 'backbone'):
                        # If using detector wrapper
                        data_dict = {'image': input_tensor}
                        pred_dict = model(data_dict)
                        logits = pred_dict['cls']
                    else:
                        # Direct model call
                        logits = model(input_tensor)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                
                # Calculate probabilities
                probabilities = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0][pred_class].item()
                
                # Class names
                class_names = ['Real', 'Fake']
                predicted_label = class_names[pred_class]
                true_label = class_names[target_class]
                
                print(f"📊 Model Prediction Results:")
                print(f"  - True Label: {true_label} (class {target_class})")
                print(f"  - Predicted: {predicted_label} (class {pred_class})")
                print(f"  - Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
                print(f"  - Raw Logits: Real={logits[0][0].item():.4f}, Fake={logits[0][1].item():.4f}")
                print(f"  - Probabilities: Real={probabilities[0][0].item():.4f}, Fake={probabilities[0][1].item():.4f}")
                
                # Check if prediction is correct
                is_correct = (pred_class == target_class)
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                print(f"  - Prediction Status: {status}")
                
            except Exception as e:
                print(f"⚠️ Error getting model prediction: {e}")
            
            # Generate both heatmaps
            print("\n🔥 Generating Grad-CAM heatmaps...")
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
            
            # Create visualization
            import matplotlib.pyplot as plt
            import cv2
            import numpy as np
            img_array = np.array(original_image)
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Original
            axes[0].imshow(img_array)
            axes[0].set_title(f'Original\n({class_name})')
            axes[0].axis('off')
            
            # Input gradient overlay
            h, w = img_array.shape[:2]
            input_resized = cv2.resize(input_heatmap, (w, h))
            input_colored = plt.cm.Reds(input_resized)[:, :, :3] * 255
            input_overlay = cv2.addWeighted(img_array, 0.7, input_colored.astype(np.uint8), 0.3, 0)
            axes[1].imshow(input_overlay)
            axes[1].set_title('Input Gradient\nOverlay')
            axes[1].axis('off')
            
            # Standard CAM pure
            if standard_heatmap is not None:
                # Use better interpolation for smoother heatmap
                standard_resized = cv2.resize(standard_heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
                
                # Apply stronger normalization for better visibility
                standard_norm = (standard_resized - standard_resized.min()) / (standard_resized.max() - standard_resized.min() + 1e-8)
                
                # Use a more vibrant colormap
                axes[2].imshow(standard_norm, cmap='jet', vmin=0, vmax=1)
                axes[2].set_title('Standard CAM\nHeatmap (16x16)')
                axes[2].axis('off')
                
                # Standard CAM overlay with enhanced visibility
                # Use jet colormap for better visibility
                standard_colored = plt.cm.jet(standard_norm)[:, :, :3] * 255
                
                # Increase overlay strength for better visibility
                standard_overlay = cv2.addWeighted(img_array, 0.6, standard_colored.astype(np.uint8), 0.4, 0)
                axes[3].imshow(standard_overlay)
                axes[3].set_title('Enhanced CAM\nOverlay')
                axes[3].axis('off')
            else:
                axes[2].text(0.5, 0.5, 'Standard CAM\nNot Available', ha='center', va='center')
                axes[2].axis('off')
                axes[3].text(0.5, 0.5, 'Standard CAM\nOverlay\nNot Available', ha='center', va='center')
                axes[3].axis('off')
            
            plt.tight_layout()
            
            # Use image index and class name for file naming
            save_path = f"./gradcam_examples/xception_{class_name.lower()}_image{i}_analysis.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved to: {save_path}")
            plt.show()
            
            print(f"✅ Grad-CAM analysis for {class_name} image completed!")
        
        print(f"\n🎉 All image Grad-CAM analysis completed!")
        print(f"📊 Total analyzed {len(available_images)} images")
        
        # Display generated files
        if os.path.exists('./gradcam_examples'):
            print("\n📁 Generated files:")
            for file in sorted(os.listdir('./gradcam_examples')):
                if 'xception' in file and 'analysis' in file:
                    print(f"  - ./gradcam_examples/{file}")
        
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
