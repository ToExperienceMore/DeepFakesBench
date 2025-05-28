# 在另一个工程中加载张量进行对比
import torch

#tensor1 = torch.load('debug_preprocessed_tensor.pt')  # 当前工程
path1="/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/debug_outputs/feature_extractor.base_model.model.vision_model.embeddings.patch_embedding_output.pt"
#path2="/root/autodl-tmp/benchmark_deepfakes/deepfake-detection/layer_outputs/debug_preprocessed_tensor.pt"
path2="/root/autodl-tmp/benchmark_deepfakes/deepfake-detection/layer_outputs/image_0__forward_module.feature_extractor.base_model.model.vision_model.embeddings.patch_embedding_output.pt"

tensor1 = torch.load(path1)
tensor2 = torch.load(path2)

# 确保两个张量在同一个设备上
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tensor1 = tensor1.to(device)
tensor2 = tensor2.to(device)

# 计算差异
diff = torch.abs(tensor1 - tensor2)
print(f"Max difference: {diff.max().item()}")
print(f"Mean difference: {diff.mean().item()}")