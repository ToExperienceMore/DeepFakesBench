import json
import random

# 读取 JSON 文件
#dataset_name = 'FMFCC-V'  # 数据集名称变量
dataset_name = 'DFDC'  # 数据集名称变量

input_file = f'preprocessing/dataset_json/{dataset_name}.json'  # 可以更改为 DFDC.json
#output_file = f'preprocessing/dataset_json/{dataset_name}_split.json'
with open(input_file, 'r') as file:
    data = json.load(file)

#{"DFDC": {"DFDC_Real": {"train": {}, "test": {"aalscayrfi": {"label": "DFDC_Real", "frames": ["/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/DFDC/DFDC_Real/aalscayrfi/000000000000.jpg"]}}}}

# 准备训练集和测试集
train_data = {
    f'{dataset_name}_Real': {'train': {}, 'test': {}},
    f'{dataset_name}_Fake': {'train': {}, 'test': {}}
}

# 遍历每个类别
for category, values in data[dataset_name].items():
    print("category:", category)
    # 获取所有的样本 ID
    sample_ids = list(values['test'].keys())
    print("len(sample_ids):", len(sample_ids))

    random.shuffle(sample_ids)  # 打乱顺序

    # 计算分割点
    split_index = int(len(sample_ids) * 0.8)
    
    # 划分训练集和测试集
    train_samples = sample_ids[:split_index]
    test_samples = sample_ids[split_index:]

    # 将样本添加到训练集和测试集中
    if category == f'{dataset_name}_Real':
        train_data[f'{dataset_name}_Real']['train'] = {sample_id: values['test'][sample_id] for sample_id in train_samples}
        train_data[f'{dataset_name}_Real']['test'] = {sample_id: values['test'][sample_id] for sample_id in test_samples}
    elif category == f'{dataset_name}_Fake':
        train_data[f'{dataset_name}_Fake']['train'] = {sample_id: values['test'][sample_id] for sample_id in train_samples}
        train_data[f'{dataset_name}_Fake']['test'] = {sample_id: values['test'][sample_id] for sample_id in test_samples}

# 更新原始数据结构
data[dataset_name] = train_data

# 将更新后的数据写入 JSON 文件
output_file = f'preprocessing/dataset_json/{dataset_name}_split.json'
with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)