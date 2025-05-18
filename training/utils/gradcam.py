import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, x, index=None):
        # Forward pass
        output = self.model(x)
        
        if index is None:
            index = output.argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[:, index] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3))
        
        # Weighted combination of feature maps
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        # ReLU on the weighted combination
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        # Resize to input size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        return cam.numpy()

def visualize_gradcam(image, cam, alpha=0.4):
    """
    Visualize GradCAM heatmap on the input image
    Args:
        image: Input image (numpy array)
        cam: GradCAM heatmap (numpy array)
        alpha: Weight for the heatmap overlay
    Returns:
        Visualization image
    """
    # Convert image to RGB if it's grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert heatmap to RGB
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Overlay heatmap on image
    output = image * alpha + heatmap * (1 - alpha)
    output = output / output.max()
    
    return output 