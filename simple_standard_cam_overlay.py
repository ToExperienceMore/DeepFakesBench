#!/usr/bin/env python3
"""
Simple Standard CAM Overlay

Direct overlay of Standard CAM on original image without over-processing
"""

import sys
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

# Add necessary paths
sys.path.append('training')
sys.path.append('training/utils')

from training.utils.universal_gradcam import load_model_and_create_gradcam

def simple_cam_overlay():
    """Simple and direct Standard CAM overlay"""
    
    print("🎯 === Simple Standard CAM Overlay ===")
    
    # Load model
    config_path = "./training/config/detector/xception.yaml"
    weights_path = "./training/weights/xception_best.pth"
    
    model, gradcam = load_model_and_create_gradcam(config_path, weights_path, 'xception')
    
    # Get test image
    test_image_path = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/000_003/000.png"
    if not os.path.exists(test_image_path):
        print("Test image not found")
        return
    
    # Preprocess
    image = Image.open(test_image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    data_dict = {'image': image_tensor.to(next(model.parameters()).device)}
    img_array = np.array(image)
    
    # Generate Standard CAM
    standard_heatmap = gradcam.gradcam.generate_gradcam(data_dict, 1, 'standard')
    
    print(f"Original image shape: {img_array.shape}")
    print(f"Standard CAM shape: {standard_heatmap.shape}")
    print(f"Standard CAM range: [{standard_heatmap.min():.3f}, {standard_heatmap.max():.3f}]")
    
    # Simple resize (keep the block characteristics)
    h, w = img_array.shape[:2]
    cam_resized = cv2.resize(standard_heatmap, (w, h), interpolation=cv2.INTER_NEAREST)  # Keep blocks
    
    # Simple overlay methods
    overlay_methods = {
        'Direct_Alpha': create_direct_overlay,
        'Simple_Mask': create_mask_overlay,
        'Color_Only_Hot': create_hot_regions_only
    }
    
    # Create comparison
    fig, axes = plt.subplots(2, len(overlay_methods) + 1, figsize=(16, 8))
    fig.suptitle('Simple Standard CAM Overlay Methods', fontsize=14)
    
    # Original image and CAM
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    im = axes[1, 0].imshow(cam_resized, cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Standard CAM\n({standard_heatmap.shape[0]}×{standard_heatmap.shape[1]})')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
    
    # Test different overlay methods
    col = 1
    for method_name, method_func in overlay_methods.items():
        overlay = method_func(img_array, cam_resized)
        
        axes[0, col].imshow(overlay)
        axes[0, col].set_title(f'{method_name}\nOverlay')
        axes[0, col].axis('off')
        
        # Show just the overlay effect
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f'{method_name}\nResult')
        axes[1, col].axis('off')
        
        col += 1
    
    plt.tight_layout()
    os.makedirs('./simple_results', exist_ok=True)
    plt.savefig('./simple_results/simple_standard_cam_overlay.png', dpi=150, bbox_inches='tight')
    print("💾 Simple overlay saved to: ./simple_results/simple_standard_cam_overlay.png")
    plt.show()

def create_direct_overlay(img_array, cam_resized, alpha=0.4):
    """Direct alpha blending"""
    # Convert CAM to RGB (red channel only)
    cam_rgb = np.zeros((*cam_resized.shape, 3))
    cam_rgb[:, :, 0] = cam_resized  # Red channel
    cam_rgb = (cam_rgb * 255).astype(np.uint8)
    
    # Simple alpha blend
    overlay = cv2.addWeighted(img_array, 1-alpha, cam_rgb, alpha, 0)
    return overlay

def create_mask_overlay(img_array, cam_resized, threshold=0.5):
    """Show only high-attention regions"""
    # Create mask for high attention regions
    mask = cam_resized > threshold
    
    # Create colored overlay
    overlay = img_array.copy()
    overlay[mask] = overlay[mask] * 0.6 + np.array([255, 0, 0]) * 0.4  # Add red tint
    
    return overlay

def create_hot_regions_only(img_array, cam_resized, threshold=0.7):
    """Show only the hottest regions"""
    # Only show top regions
    hot_mask = cam_resized > threshold
    
    overlay = img_array.copy()
    
    # Create a red highlight for hot regions
    overlay[hot_mask, 0] = np.minimum(overlay[hot_mask, 0] + 100, 255)  # Boost red
    overlay[hot_mask, 1] = np.maximum(overlay[hot_mask, 1] - 50, 0)     # Reduce green
    overlay[hot_mask, 2] = np.maximum(overlay[hot_mask, 2] - 50, 0)     # Reduce blue
    
    return overlay

def compare_with_input_gradient():
    """Compare Standard CAM with Input Gradient side by side"""
    
    print("\n📊 === Standard CAM vs Input Gradient ===")
    
    # Load model
    config_path = "./training/config/detector/xception.yaml"
    weights_path = "./training/weights/xception_best.pth"
    
    model, gradcam = load_model_and_create_gradcam(config_path, weights_path, 'xception')
    
    # Get test image
    test_image_path = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/000_003/000.png"
    if not os.path.exists(test_image_path):
        print("Test image not found")
        return
    
    # Preprocess
    image = Image.open(test_image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    data_dict = {'image': image_tensor.to(next(model.parameters()).device)}
    img_array = np.array(image)
    
    # Generate both heatmaps
    input_grad_heatmap = gradcam.gradcam.generate_gradcam(data_dict, 1, 'input_grad')
    standard_heatmap = gradcam.gradcam.generate_gradcam(data_dict, 1, 'standard')
    
    # Resize standard CAM (simple)
    h, w = img_array.shape[:2]
    standard_resized = cv2.resize(standard_heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create simple overlays
    standard_overlay = create_direct_overlay(img_array, standard_resized)
    input_overlay = create_direct_overlay(img_array, input_grad_heatmap)
    
    # Create comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Standard CAM vs Input Gradient (Simple Comparison)', fontsize=14)
    
    # Row 1: Standard CAM
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(standard_resized, cmap='Reds', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Standard CAM\n({standard_heatmap.shape[0]}×{standard_heatmap.shape[1]} → {standard_resized.shape[0]}×{standard_resized.shape[1]})')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    axes[0, 2].imshow(standard_overlay)
    axes[0, 2].set_title('Standard CAM\nSimple Overlay')
    axes[0, 2].axis('off')
    
    # Row 2: Input Gradient
    axes[1, 0].imshow(img_array)
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(input_grad_heatmap, cmap='Reds', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Input Gradient\n({input_grad_heatmap.shape[0]}×{input_grad_heatmap.shape[1]})')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)
    
    axes[1, 2].imshow(input_overlay)
    axes[1, 2].set_title('Input Gradient\nSimple Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('./simple_results/standard_vs_input_simple.png', dpi=150, bbox_inches='tight')
    print("💾 Simple comparison saved to: ./simple_results/standard_vs_input_simple.png")
    plt.show()
    
    # Print statistics
    print(f"\n📈 Statistics:")
    print(f"Standard CAM - Shape: {standard_heatmap.shape}, Range: [{standard_heatmap.min():.3f}, {standard_heatmap.max():.3f}]")
    print(f"Input Gradient - Shape: {input_grad_heatmap.shape}, Range: [{input_grad_heatmap.min():.3f}, {input_grad_heatmap.max():.3f}]")

def main():
    """Main function"""
    print("🚀 Simple Standard CAM Overlay")
    print("=" * 40)
    
    # Simple overlay methods
    simple_cam_overlay()
    
    # Compare with input gradient
    compare_with_input_gradient()
    
    print("\n✨ Simple analysis complete!")
    print("📋 Key Points:")
    print("- Standard CAM: 8×8 blocks show model's true attention regions")
    print("- No over-smoothing - preserves original CAM characteristics")
    print("- Simple INTER_NEAREST resize keeps block structure")
    print("- Direct alpha blending for clean overlay")

if __name__ == "__main__":
    main()
