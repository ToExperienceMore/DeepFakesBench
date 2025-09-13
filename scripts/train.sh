#!/bin/bash
# Training script for deepfake detection models
# Usage: ./train.sh

# Configuration - uncomment the model you want to train
config=./training/config/detector/clip_enhanced.yaml
# config=./training/config/detector/xception.yaml
# config=./training/config/detector/timesformer.yaml
# config=./training/config/detector/efficientnet.yaml
# config=./training/config/detector/clip_stan.yaml

# Optional: Set checkpoint path for resume training
# checkpoint="./logs/training/model_checkpoint.pth"

# Task name for logging
task_name="deepfake_detection"

set -x

# Available datasets: FaceForensics++, DFDC, Celeb-DF-v2, UADFV, etc.

# Main training command
nohup python training/train.py \
--detector_path $config \
--train_dataset "FaceForensics++" \
--test_dataset "DFDC" \
--task_target $task_name \
> logs/training/train_${task_name}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Training started in background. Check logs/training/ for progress."

# Alternative training examples (uncomment to use):

# Train on multiple datasets
# python training/train.py --detector_path $config --train_dataset "DFDC" "FaceForensics++" --test_dataset "DFDC" "FaceForensics++"

# Resume from checkpoint
# python training/train.py --detector_path $config --checkpoint $checkpoint --train_dataset "FaceForensics++" --test_dataset "DFDC"

# Debug mode with limited samples
# python training/train.py --detector_path $config --train_dataset "FaceForensics++" --test_dataset "DFDC" --max_train_images 100 --max_test_images 100