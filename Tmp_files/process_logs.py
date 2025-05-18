import re
import os
from datetime import datetime
import tensorboardX as tb
import numpy as np

def parse_log_line(line):
    # 解析训练指标
    train_metrics = {}
    if 'training-metric' in line:
        metrics = re.findall(r'training-metric, (\w+): ([\d.]+)', line)
        for metric, value in metrics:
            train_metrics[f'train/{metric}'] = float(value)
    
    # 解析训练损失
    if 'training-loss' in line:
        loss = re.search(r'training-loss, overall: ([\d.]+)', line)
        if loss:
            train_metrics['train/loss'] = float(loss.group(1))
    
    # 解析测试指标
    test_metrics = {}
    if 'testing-metric' in line:
        metrics = re.findall(r'testing-metric, (\w+): ([\d.]+)', line)
        for metric, value in metrics:
            test_metrics[f'test/{metric}'] = float(value)
    
    # 解析测试损失
    if 'testing-loss' in line:
        loss = re.search(r'testing-loss, overall: ([\d.]+)', line)
        if loss:
            test_metrics['test/loss'] = float(loss.group(1))
    
    # 解析步骤数
    step = None
    if 'Iter:' in line:
        step = int(re.search(r'Iter: (\d+)', line).group(1))
    elif 'step:' in line:
        step = int(re.search(r'step: (\d+)', line).group(1))
    
    return train_metrics, test_metrics, step

def process_log_file(log_file, output_dir):
    # 创建 TensorBoard 写入器
    writer = tb.SummaryWriter(output_dir)
    
    with open(log_file, 'r') as f:
        for line in f:
            train_metrics, test_metrics, step = parse_log_line(line)
            
            if step is not None:
                # 写入训练指标
                for metric, value in train_metrics.items():
                    writer.add_scalar(metric, value, step)
                
                # 写入测试指标
                for metric, value in test_metrics.items():
                    writer.add_scalar(metric, value, step)
    
    writer.close()

if __name__ == '__main__':
    log_file = 'logs/training/uia_vit_2025-05-17-18-46-28/training.log'
    output_dir = 'logs/tensorboard/uia_vit_2025-05-17-18-46-28'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    process_log_file(log_file, output_dir)
    print(f"TensorBoard logs have been saved to {output_dir}")
    print("To view the logs, run: tensorboard --logdir=" + output_dir) 