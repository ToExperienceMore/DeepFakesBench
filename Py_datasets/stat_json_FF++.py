import json
from collections import defaultdict

def analyze_dataset(dataset_name):
    # 读取 JSON 文件
    input_file = f'preprocessing/dataset_json/{dataset_name}.json'
    with open(input_file, 'r') as file:
        data = json.load(file)

    # 初始化统计变量
    stats = {
        'train': {'real': 0, 'fake': 0},
        'test': {'real': 0, 'fake': 0}
    }

    # 初始化帧数区间统计
    frame_intervals = {
        '0-7': 0,
        '8': 0,
        '9-15': 0,
        '16':0,
        '17-31': 0,
        '32':0,
        '>32': 0
    }

    # 遍历每个类别
    for category, values in data[dataset_name].items():
        # 统计训练集
        train_count = len(values['train'])
        if 'Real' in category:
            stats['train']['real'] += train_count
        else:
            stats['train']['fake'] += train_count
        
        # 统计测试集
        test_count = len(values['test'])
        if 'Real' in category:
            stats['test']['real'] += test_count
        else:
            stats['test']['fake'] += test_count
        
        # 统计测试集中每个视频的帧数分布
        for video_id, video_data in values['test'].items():
            frame_count = len(video_data['frames'])
            if frame_count < 8:
                frame_intervals['0-7'] += 1
            elif frame_count == 8:
                frame_intervals['8'] += 1
            elif frame_count < 16:
                frame_intervals['9-15'] += 1
            elif frame_count == 16:
                frame_intervals['16'] += 1
            elif frame_count < 32:
                frame_intervals['17-31'] += 1
            elif frame_count == 32:
                frame_intervals['32'] += 1
            else:
                frame_intervals['>32'] += 1

    # 打印统计结果
    print("\n=== 数据集统计信息 ===")
    print(f"\n训练集:")
    print(f"真实视频数量: {stats['train']['real']}")
    print(f"伪造视频数量: {stats['train']['fake']}")
    print(f"训练集总视频数: {stats['train']['real'] + stats['train']['fake']}")

    print(f"\n测试集:")
    print(f"真实视频数量: {stats['test']['real']}")
    print(f"伪造视频数量: {stats['test']['fake']}")
    print(f"测试集总视频数: {stats['test']['real'] + stats['test']['fake']}")

    print(f"\n测试集视频帧数分布:")
    for interval, count in frame_intervals.items():
        print(f"{interval} 帧的视频数量: {count}")

    # 保存统计结果到JSON文件
    stats_output = {
        'video_counts': stats,
        'frame_distribution': frame_intervals
    }

    output_file = f'preprocessing/dataset_json/{dataset_name}_stats.json'
    with open(output_file, 'w') as file:
        json.dump(stats_output, file, indent=4)

if __name__ == "__main__":
    dataset_name = 'DFDC'  # 数据集名称变量
    analyze_dataset(dataset_name)