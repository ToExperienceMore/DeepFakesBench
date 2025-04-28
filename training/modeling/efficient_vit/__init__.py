from .efficient_vit import EfficientViT

def build_efficient_vit(config):
    model = EfficientViT(num_classes=config.num_classes)
    return model 