# 在另一个工程中加载张量进行对比
import torch

tensor1 = torch.load('debug_preprocessed_tensor.pt')  # 当前工程
path="/root/autodl-tmp/benchmark_deepfakes/deepfake-detection/layer_outputs/debug_preprocessed_tensor.pt"
tensor2 = torch.load(path)      # 另一个工程

# 计算差异
diff = torch.abs(tensor1 - tensor2)
print(f"Max difference: {diff.max().item()}")
print(f"Mean difference: {diff.mean().item()}")