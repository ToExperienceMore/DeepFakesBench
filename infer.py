import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPVisionModel, CLIPProcessor
from dataclasses import dataclass
from typing import Optional

@dataclass
class HeadOutput:
    logits_labels: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes, normalize_inputs=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.normalize_inputs = normalize_inputs

    def forward(self, x):
        if self.normalize_inputs:
            x = F.normalize(x, p=2, dim=1)
        logits = self.linear(x)
        features = x if self.normalize_inputs else F.normalize(x, p=2, dim=1)
        return HeadOutput(logits_labels=logits, features=features)

class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.features_dim = self.vision_model.config.hidden_size

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, preprocessed_images: torch.Tensor) -> torch.Tensor:
        return self.vision_model(preprocessed_images).pooler_output

    def get_features_dim(self):
        return self.features_dim

class DeepfakeDetectionModel(nn.Module):
    def __init__(self, backbone="openai/clip-vit-large-patch14", num_classes=2):
        super().__init__()
        self.feature_extractor = CLIPEncoder(backbone)
        self.model = LinearProbe(self.feature_extractor.get_features_dim(), num_classes, normalize_inputs=True)

    def forward(self, inputs) -> HeadOutput:
        features = self.feature_extractor(inputs)
        outputs = self.model(features)
        return outputs

def load_model(model_path):
    """加载模型和权重"""
    # 加载 checkpoint
    ckpt = torch.load(model_path, map_location="cpu")
    
    # 从 hyper_parameters 获取配置
    config = ckpt["hyper_parameters"]
    
    # 初始化模型
    model = DeepfakeDetectionModel(
        backbone=config["backbone"],
        num_classes=config["num_classes"]
    )
    model.eval()
    
    # 加载权重
    model.load_state_dict(ckpt["state_dict"])
    return model

def predict_image(model, image_path):
    """预测单张图片"""
    # 1. 加载图片
    image = Image.open(image_path)
    
    # 2. 预处理
    preprocessed_image = model.feature_extractor.preprocess(image)
    # 添加 batch 维度
    preprocessed_image = preprocessed_image.unsqueeze(0)
    
    # 3. 推理
    with torch.no_grad():
        outputs = model(preprocessed_image)
        probs = outputs.logits_labels.softmax(1)
        pred = probs.argmax(1).item()
    
    return {
        'prediction': pred,  # 0: real, 1: fake
        'probability': probs[0].tolist()  # [real_prob, fake_prob]
    }

if __name__ == "__main__":
    # 使用示例
    model_path = "models_epoch9/checkpoints/best_mAP.ckpt"
    image_path = "path/to/your/image.jpg"

    # 加载模型
    model = load_model(model_path)

    # 预测单张图片
    result = predict_image(model, image_path)
    print(f"Prediction: {'Fake' if result['prediction'] == 1 else 'Real'}")
    print(f"Probabilities: Real: {result['probability'][0]:.3f}, Fake: {result['probability'][1]:.3f}")
