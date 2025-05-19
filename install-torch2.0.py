#!/bin/bash

#echo "🔧 Creating conda environment: DeepfakeBench-torch2.0 with Python 3.9"
#conda create -y -n DeepfakeBench-torch2.0 python=3.9
#conda activate DeepfakeBench-torch2.0

echo "📦 Installing PyTorch 2.0 + torchvision + torchaudio (CUDA 11.8)"
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "📦 Installing general dependencies"
pip install \
    numpy \
    tqdm \
    opencv-python \
    scikit-learn \
    matplotlib \
    scikit-image \
    albumentations \
    einops \
    timm \
    decord==0.6.0 \
    torchmetrics==0.10.3 \
    fvcore

echo "📦 Installing transformers (≥ 4.35) for ViT, TimeSformer, etc."
pip install transformers==4.35.2

echo "📦 Installing dinov2"
pip install git+https://github.com/facebookresearch/dinov2.git

echo "✅ All dependencies installed. Now you can run DeepfakeBench!"