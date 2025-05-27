"""
Predict deepfake probability for a single image.
"""
import os
import numpy as np
import cv2
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from detectors import DETECTOR

torch.set_float32_matmul_precision("high")

def load_config(config_path):
    """Load model configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(config, weights_path):
    """Load model and weights."""
    # Initialize model
    detector = DETECTOR[config['model_name']](config)
    detector = detector.to('cuda' if torch.cuda.is_available() else 'cpu')

    
    # Load weights
    checkpoint = torch.load(weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # 转换键值
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('feature_extractor.base_model.model.', 'feature_extractor.')
            new_state_dict[new_key] = v
        detector.load_state_dict(new_state_dict, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('===> Load checkpoint error!')
        exit()
    
    detector.eval()
    return detector

def preprocess_image(image_path, config):
    """Preprocess single image for inference."""
    # Read image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    """
    transform = T.Compose([
        T.Resize((config['resolution'], config['resolution'])),
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])
    """

    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # Apply transforms
    image = transform(image)
    
    # Also save the normalized tensor for comparison
    torch.save(image, 'debug_preprocessed_tensor.pt')
    
    image = image.unsqueeze(0)  # Add batch dimension
    return image

class LayerOutputHook:
    def __init__(self):
        self.outputs = {}
        
    def __call__(self, name):
        def hook(module, input, output):
            self.outputs[name] = output.detach()
        return hook

def register_hooks(model):
    """Register hooks for all layers in the model."""
    hooks = []
    layer_outputs = LayerOutputHook()
    
    # Register hooks for all named modules
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            hook = module.register_forward_hook(layer_outputs(name))
            hooks.append(hook)
    
    return hooks, layer_outputs

def predict_image(model, image, device):
    """Run inference on single image and save intermediate outputs."""
    # Create debug directory
    os.makedirs('debug_outputs', exist_ok=True)
    
    # Register hooks
    hooks, layer_outputs = register_hooks(model)
    
    with torch.no_grad():
        image = image.to(device)
        pred_dict = model({'image': image}, inference=True)
        prob = pred_dict['prob'].item()
        
        # Save intermediate outputs
        for name, output in layer_outputs.outputs.items():
            # Save tensor
            torch.save(output, f'debug_outputs/{name}_output.pt')
            
            # If the output is a feature map, also save a visualization
            if len(output.shape) == 4:  # [B, C, H, W]
                # Take the first channel for visualization
                feature_map = output[0, 0].cpu().numpy()
                # Normalize to [0, 1]
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                # Convert to uint8
                feature_map = (feature_map * 255).astype(np.uint8)
                # Save as image
                cv2.imwrite(f'debug_outputs/{name}_feature.png', feature_map)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
        
    return prob

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Predict deepfake probability for a single image.')
    parser.add_argument('--detector_path', type=str, required=True,
                        help='path to detector YAML file')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='path to model weights')
    parser.add_argument('--image_path', type=str, required=True,
                        help='path to input image')
    args = parser.parse_args()

    # Load config and model
    config = load_config(args.detector_path)
    model = load_model(config, args.weights_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preprocess image
    image = preprocess_image(args.image_path, config)

    # Run prediction
    prob = predict_image(model, image, device)
    
    # Print result
    print(f"Deepfake probability: {prob:.4f}")
    print(f"Prediction: {'Fake' if prob > 0.5 else 'Real'}")

if __name__ == '__main__':
    main()
