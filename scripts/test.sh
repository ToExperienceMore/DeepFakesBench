#!/bin/bash
# Test script for deepfake detection models
# Usage: ./test.sh

# Configuration
weight=./training/weights/xception_best.pth
config=./training/config/detector/xception.yaml

# Alternative configurations (uncomment to use)
# config=./training/config/detector/timesformer.yaml
# config=./training/config/detector/efficientnet.yaml
# config=./training/config/detector/clip_enhanced.yaml
# config=./training/config/detector/clip_stan.yaml

set -x

# Available datasets:
# FaceForensics++, FF-F2F, FF-DF, FF-FS, FF-NT, FaceShifter, DeepFakeDetection, 
# Celeb-DF-v1, Celeb-DF-v2, DFDCP, DFDC, DeeperForensics-1.0, UADFV

# Test on single dataset
python3 training/test.py --detector_path $config --test_dataset "FaceShifter" --weights_path $weight

# Test on multiple datasets (uncomment to use)
# python3 training/test.py --detector_path $config --test_dataset "DFDC" "FaceForensics++" "DFDCP" "Celeb-DF-v2" "UADFV" "FaceShifter" "DeepFakeDetection" --weights_path $weight

# Alternative test script for custom test lists
# python3 training/test2.py --detector_path $config --test_list $test_list --weights_path $weight