"""
Xception Grad-CAM Implementation
Provides Gradient-weighted Class Activation Mapping (Grad-CAM) visualization for Xception models

Main Features:
1. Standard Grad-CAM - Based on feature maps and gradients from the last convolutional layer
2. Input Gradient Visualization - Directly visualizes the importance of input pixels

Author: AI Assistant
Date: 2024-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

class XceptionGradCAM:
    """
    Grad-CAM implementation for Xception models
    
    Supports two visualization methods:
    1. Traditional Grad-CAM: Based on feature maps and gradients from the last convolutional layer (conv4)
    2. Input Gradient: Directly computes the importance of input pixels
    """
    
    def __init__(self, model, target_layer_name='conv4'):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Xception model
            target_layer_name: Target layer name, default is 'conv4' (last convolutional layer)
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # Find target layer
        self._find_target_layer()
        
        # Register hooks
        self._register_hooks()
        
    def _find_target_layer(self):
        """Find target layer"""
        if hasattr(self.model, 'backbone'):
            # If it's a detector format, use backbone
            backbone = self.model.backbone
        else:
            # Direct backbone
            backbone = self.model
            
        if hasattr(backbone, self.target_layer_name):
            self.target_layer = getattr(backbone, self.target_layer_name)
            print(f"✅ Found target layer: {self.target_layer_name}")
        else:
            print(f"❌ Target layer not found: {self.target_layer_name}")
            print("Available layers:")
            for name, module in backbone.named_modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    print(f"  - {name}: {module}")
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        if self.target_layer is not None:
            # Forward hook: save activations
            def forward_hook(module, input, output):
                self.activations = output
                
            # Backward hook: save gradients
            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0]
            
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_backward_hook(backward_hook)
            print(f"✅ Registered hooks for {self.target_layer_name}")
    
    def generate_gradcam(self, data_dict, target_class=1, method='standard'):
        """
        Generate Grad-CAM heatmap
        
        Args:
            data_dict: Input data dictionary containing 'image' key
            target_class: Target class (0=Real, 1=Fake)
            method: Method type ('standard'=Standard Grad-CAM, 'input_grad'=Input Gradient)
            
        Returns:
            heatmap: Heatmap numpy array [H, W]
        """
        if method == 'input_grad':
            return self._generate_input_gradient(data_dict, target_class)
        else:
            return self._generate_standard_gradcam(data_dict, target_class)
    
    def _generate_standard_gradcam(self, data_dict, target_class):
        """Generate standard Grad-CAM"""
        if self.target_layer is None:
            raise ValueError("Target layer not found, cannot generate standard Grad-CAM")
        
        # Forward pass
        self.model.eval()
        with torch.enable_grad():
            pred_dict = self.model(data_dict, inference=True)
            predictions = pred_dict['cls']
            
            # Select target class score
            score = predictions[0, target_class]
            
            # Backward pass
            self.model.zero_grad()
            score.backward(retain_graph=True)
            
            # Get gradients and activations
            if self.gradients is None or self.activations is None:
                raise ValueError("Failed to get gradients or activations, please check if hooks are registered correctly")
            
            # Calculate weights (global average pooling)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
            
            # Weighted sum
            cam = torch.sum(weights * self.activations, dim=1)  # [1, H, W]
            
            # ReLU to keep only positive contributions
            cam = F.relu(cam)
            
            # Convert to numpy
            heatmap = cam.squeeze().detach().cpu().numpy()
            
            # Normalize to [0,1]
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            return heatmap
    
    def _generate_input_gradient(self, data_dict, target_class):
        """Generate input gradient visualization"""
        # Get input image and set requires_grad
        image_tensor = data_dict['image'].clone().detach()
        image_tensor.requires_grad_(True)
        
        # Create new data dictionary
        new_data_dict = {'image': image_tensor}
        for key, value in data_dict.items():
            if key != 'image':
                new_data_dict[key] = value
        
        # Forward pass
        self.model.eval()
        pred_dict = self.model(new_data_dict, inference=True)
        predictions = pred_dict['cls']
        
        # Select target class score
        score = predictions[0, target_class]
        
        # Backward pass to compute gradients
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Get input gradients
        gradients = image_tensor.grad.data
        
        # Calculate gradient magnitude (L2 norm)
        grad_magnitude = torch.norm(gradients, dim=1)  # [B, H, W]
        
        # Convert to numpy
        heatmap = grad_magnitude.squeeze().cpu().numpy()
        
        # Normalization and contrast enhancement
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            # Contrast enhancement
            heatmap = np.power(heatmap, 0.5)
            # Highlight high-value regions
            threshold = np.percentile(heatmap, 90)
            heatmap = np.where(heatmap > threshold, heatmap, heatmap * 0.3)
        
        return heatmap

def preprocess_image(image_path, resolution=256):
    """
    Preprocess image, consistent with Xception training
    
    Args:
        image_path: Image path
        resolution: Image resolution
        
    Returns:
        data_dict: Data dictionary containing preprocessed image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transform (consistent with training)
    transform = T.Compose([
        T.Resize((resolution, resolution)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Based on config file
    ])
    
    # Apply transform and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    
    # Create data dictionary
    data_dict = {'image': image_tensor}
    
    return data_dict, image

def visualize_gradcam(image, heatmap, save_path=None, alpha=0.4):
    """
    Visualize Grad-CAM results
    
    Args:
        image: Original PIL image
        heatmap: Heatmap numpy array
        save_path: Save path
        alpha: Heatmap transparency
    """
    # Convert original image to numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Resize heatmap to match original image
    h, w = img_array.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Create colored version of heatmap
    heatmap_colored = plt.cm.Reds(heatmap_resized)[:, :, :3]  # Use red colormap
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap_resized, cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], shrink=0.6, label='Attention')
    
    # Overlay image
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Visualization results saved to: {save_path}")
    
    plt.show()
    
    return overlay

def compare_gradcam_methods(model, image_path, target_class=1, save_dir='./gradcam_results'):
    """
    Compare the effectiveness of different Grad-CAM methods
    
    Args:
        model: Xception model
        image_path: Image path
        target_class: Target class
        save_dir: Save directory
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Preprocess image
    data_dict, original_image = preprocess_image(image_path)
    
    # Create Grad-CAM object
    gradcam = XceptionGradCAM(model)
    
    # Generate heatmaps with different methods
    methods = [
        ('standard', 'Standard Grad-CAM'),
        ('input_grad', 'Input Gradient')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, (method, method_name) in enumerate(methods):
        try:
            # Generate heatmap
            heatmap = gradcam.generate_gradcam(data_dict, target_class, method)
            
            # Visualize
            img_array = np.array(original_image)
            h, w = img_array.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (w, h))
            heatmap_colored = plt.cm.Reds(heatmap_resized)[:, :, :3]
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
            
            # Plot subgraphs
            axes[i, 0].imshow(img_array)
            axes[i, 0].set_title(f'{method_name} - Original Image')
            axes[i, 0].axis('off')
            
            im = axes[i, 1].imshow(heatmap_resized, cmap='Reds', vmin=0, vmax=1)
            axes[i, 1].set_title(f'{method_name} - Heatmap')
            axes[i, 1].axis('off')
            plt.colorbar(im, ax=axes[i, 1], shrink=0.6)
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f'{method_name} - Overlay Result')
            axes[i, 2].axis('off')
            
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f'Failed\n{e}', ha='center', va='center', 
                               transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'xception_gradcam_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Comparison results saved to: {save_path}")
    plt.show()

# Usage example
if __name__ == "__main__":
    print("🔧 Xception Grad-CAM Tool")
    print("Please import this module and load your trained model for actual use")
