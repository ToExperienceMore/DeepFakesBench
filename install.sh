#!/bin/bash
# Environment setup script for deepfake detection models
# Usage: ./install.sh

echo "Installing dependencies for deepfake detection project..."

# Core PyTorch dependencies
echo "Installing PyTorch and related packages..."
pip install torch==2.6.0 torchvision torchaudio
pip install transformers==4.50.0
pip install timm==1.0.15

# Computer vision and machine learning
echo "Installing computer vision and ML packages..."
pip install opencv-python
pip install dlib
pip install scikit-learn
pip install scikit-image==0.19.2
pip install albumentations==1.1.0
pip install efficientnet-pytorch==0.7.1

# Utilities and visualization
echo "Installing utilities and visualization packages..."
pip install numpy
pip install pandas
pip install Pillow
pip install imageio==2.9.0
pip install tqdm
pip install scipy
pip install seaborn
pip install pyyaml
pip install imutils==0.5.4
pip install tensorboard

# Deep learning utilities
echo "Installing deep learning utilities..."
pip install einops
pip install loralib
pip install peft==0.14.0
pip install lightning==2.5.0
pip install torchtoolbox==0.1.8.2
pip install segmentation-models-pytorch==0.3.2

# Additional utilities
echo "Installing additional utilities..."
pip install wandb==0.19.4
pip install pydantic==2.9.2
pip install fire==0.7.0
pip install filterpy
pip install simplejson
pip install kornia
pip install fvcore
pip install setuptools

echo "✅ All dependencies installed successfully!"
echo "You can now run training and testing scripts."
