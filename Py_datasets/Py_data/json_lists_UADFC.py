import json
import os
from collections import defaultdict
import sys

data_dir = '/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb'

def generate_dataset_lists(json_path, dataset_name, output_dir='.'):
    """
    从JSON文件中提取训练集和测试集数据并生成列表文件
    
    Args:
        json_path: JSON文件的路径
        dataset_name: 数据集名称 (例如 'DFDC', 'FaceForensics++', 'UADFV','Celeb-DF-v2')
        output_dir: 输出目录
    """
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取数据集
    train_samples = []
    test_samples = []
    val_samples = []
    video_count = defaultdict(int)  # 统计每个视频的文件数
    total_files = 0
    train_video_count = 0
    test_video_count = 0
    val_video_count = 0
    
    # 根据数据集名称确定类别
    if dataset_name == 'DFDC':
        categories = ['DFDC_Real', 'DFDC_Fake']
        dataset_key = 'DFDC'
    elif dataset_name == 'DFDCP':
        categories = ['DFDCP_Real', 'DFDCP_FakeA', 'DFDCP_FakeB']
        dataset_key = 'DFDCP'
    elif dataset_name == 'FaceForensics++':
        categories = ['FF-real', 'FF-NT', 'FF-F2F', 'FF-DF', 'FF-FS']
        dataset_key = 'FaceForensics++'
    elif dataset_name == 'UADFV':
        categories = ['UADFV_Real', 'UADFV_Fake']
        dataset_key = 'UADFV'
    elif dataset_name == 'Celeb-DF-v2':
        categories = ['CelebDFv2_real', 'CelebDFv2_fake']
        dataset_key = 'Celeb-DF-v2'
    elif dataset_name == 'FaceShifter':
        categories = ['FF-FH', 'FF-real']
        dataset_key = 'FaceShifter'
    elif dataset_name == 'DeepFakeDetection':
        categories = ['DFD_fake', 'DFD_real']
        dataset_key = 'DeepFakeDetection'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # 处理真实和伪造类别
    if dataset_name == 'UADFV' or dataset_name == 'Celeb-DF-v2' or dataset_name == 'DFDCP' or dataset_name == 'DFDC':
        # 处理UADFV数据集
        for category in categories:
            if category in data[dataset_key]:
                # 处理测试集
                if 'test' in data[dataset_key][category]:
                    for video_id, video_info in data[dataset_key][category]['test'].items():
                        label = 0 if 'Real' in category or 'real' in category else 1
                        test_video_count += 1
                        for frame_path in video_info['frames']:
                            frame_path = frame_path.replace('\\', '/')
                            test_samples.append((frame_path, label))
                            video_count[video_id] += 1
                            total_files += 1
                if 'train' in data[dataset_key][category]:
                    for video_id, video_info in data[dataset_key][category]['train'].items():
                        label = 0 if 'Real' in category or 'real' in category else 1
                        train_video_count += 1
                        for frame_path in video_info['frames']:
                            frame_path = frame_path.replace('\\', '/')
                            train_samples.append((frame_path, label))
                            video_count[video_id] += 1
                            total_files += 1
                if 'val' in data[dataset_key][category]:
                    for video_id, video_info in data[dataset_key][category]['val'].items():
                        label = 0 if 'Real' in category or 'real' in category else 1
                        val_video_count += 1
                        for frame_path in video_info['frames']:
                            frame_path = frame_path.replace('\\', '/')
                            val_samples.append((frame_path, label))
                            video_count[video_id] += 1
                            total_files += 1
    else:
        # FaceForensics++处理逻辑
        for category in categories:
            if category in data[dataset_key]:
                # 获取训练集和测试集数据
                train_data = data[dataset_key][category].get('train', {})
                test_data = data[dataset_key][category].get('test', {})
                
                # 处理训练集
                if dataset_name == 'FaceForensics++' or dataset_name == 'FaceShifter' or dataset_name == 'DeepFakeDetection':
                    for quality in ['c23']:  # 可以根据需要添加其他质量级别
                        if quality in train_data:
                            for video_id, video_info in train_data[quality].items():
                                label = 0 if 'real' in category else 1
                                train_video_count += 1 
                                for frame_path in video_info['frames']:
                                    frame_path = frame_path.replace('\\', '/')
                                    train_samples.append((frame_path, label))
                                    video_count[video_id] += 1
                                    total_files += 1
                
                # 处理测试集
                if dataset_name == 'FaceForensics++' or dataset_name == 'FaceShifter' or dataset_name == 'DeepFakeDetection':
                    for quality in ['c23']:  # 可以根据需要添加其他质量级别
                        if quality in test_data:
                            for video_id, video_info in test_data[quality].items():
                                test_video_count += 1 
                                label = 0 if 'real' in category else 1
                                for frame_path in video_info['frames']:
                                    frame_path = frame_path.replace('\\', '/')
                                    test_samples.append((frame_path, label))
                                    video_count[video_id] += 1
                                    total_files += 1
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    train_file = os.path.join(output_dir, f'{dataset_name}_train_list.txt')
    test_file = os.path.join(output_dir, f'{dataset_name}_test_list.txt')
    val_file = os.path.join(output_dir, f'{dataset_name}_val_list.txt')
    
    # 写入训练列表文件
    with open(train_file, 'w') as f:
        for file_path, label in train_samples:
            f.write(f'{file_path} {label}\n')
    
    # 写入测试列表文件
    with open(test_file, 'w') as f:
        for file_path, label in test_samples:
            f.write(f'{file_path} {label}\n')
            file_path = file_path.replace('\\', '/')
            path = os.path.join(data_dir, file_path)
            if not os.path.exists(path):
                print(f'Error: {path} does not exist')
                exit(1)

    
    # 写入验证列表文件
    with open(val_file, 'w') as f:
        for file_path, label in val_samples:
            f.write(f'{file_path} {label}\n')
    
    # 打印统计信息
    print(f'\nGenerated {train_file} with {len(train_samples)} training samples')
    print(f'Generated {test_file} with {len(test_samples)} test samples')
    print(f'Total number of videos: {len(video_count)}')
    print(f'Total number of files: {total_files}')
    
    # 统计真实和伪造的数量
    train_real = sum(1 for _, label in train_samples if label == 0)
    train_fake = sum(1 for _, label in train_samples if label == 1)
    test_real = sum(1 for _, label in test_samples if label == 0)
    test_fake = sum(1 for _, label in test_samples if label == 1)
    val_real = sum(1 for _, label in val_samples if label == 0)
    val_fake = sum(1 for _, label in val_samples if label == 1)
    
    print(f'\nTraining set:')
    print(f'train videos: {train_video_count}')
    print(f'Real samples: {train_real}')
    print(f'Fake samples: {train_fake}')
    
    print(f'\nTest set:')
    print(f'test videos: {test_video_count}')
    print(f'Real samples: {test_real}')
    print(f'Fake samples: {test_fake}')
    
    print(f'\nValidation set:')
    print(f'val videos: {val_video_count}')
    print(f'Real samples: {val_real}')
    print(f'Fake samples: {val_fake}')
    
    """
    print('\nFiles per video:')
    for video, count in video_count.items():
        print(f'{video}: {count} files')
    """
    
    return train_file, test_file

if __name__ == '__main__':
    # 示例用法
    json_dir = '/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/preprocessing/dataset_json'
    output_dir = 'test_lists'
    #dataset_name = 'Celeb-DF-v2'
    #dataset_name = 'DFDCP'
    #dataset_name = 'FaceShifter'
    dataset_name = 'DeepFakeDetection'

    # 处理UADFV数据集
    dataset_json = os.path.join(json_dir, dataset_name + '.json')
    if not os.path.exists(dataset_json):
        print(f"Error: {dataset_json} does not exist")
        exit(1)
    train_file, test_file = generate_dataset_lists(dataset_json, dataset_name, output_dir)
