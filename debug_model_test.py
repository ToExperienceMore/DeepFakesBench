#!/usr/bin/env python3
"""
Direct model test to debug the probability issue
"""
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from training.utils.universal_gradcam import load_model_and_create_gradcam

def test_model_directly():
    print("🔍 Direct Model Test - Debugging Probability Issue")
    print("="*60)
    
    # Load model
    config_path = "training/config/detector/xception.yaml"
    weights_path = "training/weights/xception_best.pth"
    model, _ = load_model_and_create_gradcam(config_path, weights_path, 'auto')
    
    # Two different images
    image_paths = [
        "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/000_003/000.png",
        "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++/original_sequences/youtube/c23/frames/000/000.png"
    ]
    
    def load_and_preprocess(image_path):
        # Same preprocessing as dataset
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f'Loaded image is None: {image_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        pil_image = Image.fromarray(np.array(img, dtype=np.uint8))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return transform(pil_image).unsqueeze(0)
    
    # Process images independently
    model.eval()
    with torch.no_grad():
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n📸 Image {i}: {image_path.split('/')[-1]}")
            print(f"   Full path: {image_path}")
            print(f"   Expected: {'Fake (Face2Face)' if 'manipulated' in image_path else 'Real'}")
            
            # Load and preprocess image
            input_tensor = load_and_preprocess(image_path)
            input_tensor = input_tensor.to(next(model.parameters()).device)
            
            print(f"   Input stats: mean={input_tensor.mean().item():.6f}, std={input_tensor.std().item():.6f}")
            print(f"   Input range: [{input_tensor.min().item():.6f}, {input_tensor.max().item():.6f}]")
            
            # Model inference using detector wrapper (same as training)
            data_dict = {'image': input_tensor}
            pred_dict = model(data_dict)
            logits = pred_dict['cls']
            
            # Calculate probabilities using softmax
            probabilities = F.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][pred_class].item()
            
            print(f"   Logits: {logits[0]}")
            print(f"   Raw values: Real={logits[0][0].item():.8f}, Fake={logits[0][1].item():.8f}")
            print(f"   Probabilities: Real={probabilities[0][0].item():.8f}, Fake={probabilities[0][1].item():.8f}")
            print(f"   Predicted class: {'Real' if pred_class == 0 else 'Fake'} (class {pred_class})")
            print(f"   Confidence: {confidence:.8f} ({confidence*100:.4f}%)")
            
            # Check if this matches expectation
            expected_class = 1 if 'manipulated' in image_path else 0
            is_correct = (pred_class == expected_class)
            print(f"   Prediction status: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
    
    print(f"\n🔄 Testing model consistency with different random seeds:")
    input_tensor = load_and_preprocess(image_paths[0])
    input_tensor = input_tensor.to(next(model.parameters()).device)
    
    for run in range(3):
        # Add small random noise to test sensitivity
        noisy_input = input_tensor + torch.randn_like(input_tensor) * 0.001
        data_dict = {'image': noisy_input}
        pred_dict = model(data_dict)
        logits = pred_dict['cls']
        probabilities = F.softmax(logits, dim=1)
        print(f"   Run {run+1} (with tiny noise): Real={probabilities[0][0].item():.8f}, Fake={probabilities[0][1].item():.8f}")
    
    print(f"\n🧪 Testing with random input to verify model responsiveness:")
    random_input = torch.randn(1, 3, 256, 256).to(next(model.parameters()).device)
    data_dict = {'image': random_input}
    pred_dict = model(data_dict)
    logits = pred_dict['cls']
    probabilities = F.softmax(logits, dim=1)
    print(f"   Random input: Real={probabilities[0][0].item():.8f}, Fake={probabilities[0][1].item():.8f}")

if __name__ == "__main__":
    test_model_directly()
