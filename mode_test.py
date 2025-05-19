#import torch
#model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
#pretrained_path = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/training/pretrained/dinov2_vitl14_reg4_pretrain.pth"
#state_dict = torch.load(pretrained_path)
#model.load_state_dict(state_dict)

import torch
import dinov2
from dinov2.models.vision_transformer import vit_base
from dinov2.models.vision_transformer import vit_large

#pretrained_path = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/training/pretrained/dinov2_vitl14_reg4_pretrain.pth"
pretrained_path = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/training/pretrained/dinov2_vitb14_reg4_pretrain.pth"

"""
# Step 1: 创建 ViT-L/14 模型（img_size=224, patch_size=14）
model = vit_large(patch_size=14, img_size=224)

# Step 2: 加载本地预训练权重（注意 checkpoint_key="teacher"）
load_pretrained_weights(model, pretrained_path, checkpoint_key="teacher")

# 可选：冻结模型权重（如果你不打算fine-tune）
for param in model.parameters():
    param.requires_grad = False
"""


# Initialize model with correct configuration
model = vit_base(
    patch_size=14,  # Match the checkpoint's patch size
    img_size=518,   # This will give us the correct pos_embed size
    init_values=1.0,
    block_chunks=0
)

# Load pretrained weights
state_dict = torch.load(pretrained_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

"""
torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
"""