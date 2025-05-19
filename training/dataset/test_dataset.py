import os
import random
from .abstract_dataset import DeepfakeAbstractBaseDataset
import cv2
from PIL import Image
import numpy as np

class TestDataset(DeepfakeAbstractBaseDataset):
    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """从 list.txt 文件加载数据
        
        list.txt 的格式：
        manipulated_sequences/FaceSwap/c23/frames/378_368/000.png 1
        """
        label_list = []
        frame_path_list = []
        video_name_list = []  # 保持与父类接口一致
        

        #data_dir = '/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++'
        #data_dir = '/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/'
        data_dir = '/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/ForgeryNet/OpenDataLab___ForgeryNet/raw/validation/Validation/'

        # 读取 list.txt
        with open(self.config['test_list'], 'r') as f:
            for line in f:
                # 解析每行：图片路径 标签
                img_path, label = line.strip().split()
                img_path = os.path.join(data_dir, img_path)
                # 将标签转换为数字
                label = int(label)
                if not os.path.exists(img_path):
                    print("img_path not exists:", img_path)
                    exit()
                # 添加到列表中
                frame_path_list.append(img_path)
                label_list.append(label)
                # 使用图片路径作为视频名称（保持与父类接口一致）
                video_name_list.append(img_path)
        
        return frame_path_list, label_list, video_name_list