import os
import random
from pathlib import Path
from collections import defaultdict

def generate_dataset_lists(root_dir, train_ratio=0.8, seed=42):
    """
    Generate train_list.txt and val_list.txt for FF++ dataset
    Ensure frames from the same video stay together in either train or val set
    Use relative paths from root_dir
    Exclude DeepFakeDetection and FaceShifter folders
    
    Args:
        root_dir: Root directory of FF++ dataset
        train_ratio: Ratio of training data (default: 0.8)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Initialize dictionary to store video paths and their frames
    video_samples = defaultdict(list)
    
    # Process manipulated sequences (label 1)
    manipulated_dir = os.path.join(root_dir, 'manipulated_sequences')
    for method in os.listdir(manipulated_dir):
        # Skip DeepFakeDetection and FaceShifter
        if method in ['DeepFakeDetection', 'FaceShifter']:
            continue
            
        method_dir = os.path.join(manipulated_dir, method, 'c23')
        if not os.path.isdir(method_dir):
            continue
            
        frames_dir = os.path.join(method_dir, 'frames')
        if not os.path.exists(frames_dir):
            continue
            
        for video_folder in os.listdir(frames_dir):
            video_path = os.path.join(frames_dir, video_folder)
            if not os.path.isdir(video_path):
                continue
                
            # Get all frames for this video
            frames = []
            for img_file in os.listdir(video_path):
                if img_file.endswith('.png'):
                    img_path = os.path.join(video_path, img_file)
                    # Convert to relative path
                    rel_path = os.path.relpath(img_path, root_dir)
                    frames.append((rel_path, 1))
            
            if frames:  # Only add if video has frames
                video_samples[video_path] = frames
    
    # Process original sequences (label 0)
    original_dir = os.path.join(root_dir, 'original_sequences')
    for source in os.listdir(original_dir):
        source_dir = os.path.join(original_dir, source, 'c23')
        if not os.path.isdir(source_dir):
            continue
            
        frames_dir = os.path.join(source_dir, 'frames')
        if not os.path.exists(frames_dir):
            continue
            
        for video_folder in os.listdir(frames_dir):
            video_path = os.path.join(frames_dir, video_folder)
            if not os.path.isdir(video_path):
                continue
                
            # Get all frames for this video
            frames = []
            for img_file in os.listdir(video_path):
                if img_file.endswith('.png'):
                    img_path = os.path.join(video_path, img_file)
                    # Convert to relative path
                    rel_path = os.path.relpath(img_path, root_dir)
                    frames.append((rel_path, 0))
            
            if frames:  # Only add if video has frames
                video_samples[video_path] = frames
    
    # Convert to list of videos and shuffle
    video_list = list(video_samples.items())
    random.shuffle(video_list)
    
    # Split videos into train and validation sets
    split_idx = int(len(video_list) * train_ratio)
    train_videos = video_list[:split_idx]
    val_videos = video_list[split_idx:]
    
    # Write train list
    with open('train_list.txt', 'w') as f:
        for _, frames in train_videos:
            for img_path, label in frames:
                f.write(f'{img_path} {label}\n')
    
    # Write validation list
    with open('val_list.txt', 'w') as f:
        for _, frames in val_videos:
            for img_path, label in frames:
                f.write(f'{img_path} {label}\n')
    
    # Print statistics
    train_frames = sum(len(frames) for _, frames in train_videos)
    val_frames = sum(len(frames) for _, frames in val_videos)
    print(f'Generated train_list.txt with {len(train_videos)} videos ({train_frames} frames)')
    print(f'Generated val_list.txt with {len(val_videos)} videos ({val_frames} frames)')

if __name__ == '__main__':
    root_dir = '/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++'
    generate_dataset_lists(root_dir) 