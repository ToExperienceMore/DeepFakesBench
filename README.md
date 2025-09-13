# Video Deepfake Detection with CLIP and CLIP-STAN

A comprehensive video deepfake detection system based on parameter-efficient fine-tuning of CLIP models. This project addresses the critical challenge of cross-dataset generalization in deepfake detection through vision-language model adaptation.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7.2-brightgreen)

## Research Overview

This research develops a complete video deepfake detection system that addresses cross-dataset generalization challenges in real-world scenarios. Built on the [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) framework, the system implements:

- **CLIP-based Detection**: Parameter-efficient fine-tuning of CLIP ViT-B/16 and ViT-L/14 models
- **Temporal Modeling**: CLIP-STAN extension for video-level temporal analysis
- **Cross-Dataset Evaluation**: Comprehensive evaluation across 7 benchmark datasets
- **Parameter Efficiency**: Achieving state-of-the-art results with minimal parameter updates

## Key Results

### Performance Highlights

Our CLIP ViT-B model achieves **superior cross-dataset generalization** performance:

- **Average Video-level AUC**: **90.9%** across 7 benchmark datasets
- **Parameter Efficiency**: Only **0.0344%** of model parameters updated
- **State-of-the-Art Results**: Best performance on DFDC (83.6%) and DFD (96.2%)

### Comparison with State-of-the-Art Methods

**Video-Level AUC (%) Results:**

| Method | CDF | DFDCP | DFDC | DFD |
|--------|-----|-------|------|-----|
| Xception | 81.6 | 74.2 | 73.2 | 89.6 |
| EfficientNet-B4 | 80.8 | 68.0 | 72.4 | 86.2 |
| TALL++ | 92.0 | - | 78.5 | - |
| UDD | **93.1** | **88.1** | 81.2 | 95.5 |
| **CLIP ViT-B (Ours)** | 89.0 | 86.6 | **83.6** | **96.2** |

### Ablation Study Results

**Cross-Dataset Performance (Video-Level AUC %):**

| Model | DFDC | DFDCP | DFD | FS | FF++ | UADFV | CDF-v2 | Avg |
|-------|------|-------|-----|----|----- |-------|--------|-----|
| Xception | 73.9 | 76.0 | 89.6 | 60.0 | 99.6 | 96.3 | 81.6 | 82.4 |
| EffB0+TimeSformer | 71.8 | 85.0 | 86.0 | 71.6 | 98.2 | 78.4 | 78.7 | 81.4 |
| CLIP ViT-B (LN+1MLP) | 83.6 | 86.6 | 96.2 | 74.4 | 99.6 | 98.4 | 89.0 | **90.9** |
| CLIP ViT-B (LN+2MLP) | 82.25 | 94.9 | 95.9 | 84.4 | 97.36 | 99.5 | 81.7 | 91.0 |
| CLIP+STAN | 85.0 | 85.0 | 98.1 | 79.0 | 96.7 | 99.0 | 83.8 | 89.5 |
| **CLIP ViT-L** | **88.6** | **89.2** | **98.2** | **85.7** | **99.8** | **99.4** | **95.0** | **93.7** |

### Parameter Efficiency Analysis

| Fine-tuning Strategy | Trainable Params | FF++ AUC | Avg AUC |
|---------------------|------------------|----------|---------|
| LayerNorm + 1MLP | **0.0344%** | 99.6% | **90.9%** |
| LayerNorm + 2MLP | 0.5041% | 98.6% | 91.0% |
| STAN (4 layers) | 9.46% | 96.7% | 88.3% |
| Full fine-tuning | 100% | Failed | Failed |

## Project Structure

```
DeepFakesDetection/
├── training/                    # Training framework and models
│   ├── detectors/              # Detector implementations (CLIP, CLIP-STAN)
│   ├── config/                 # Configuration files for different models
│   ├── dataset/                # Dataset processing and loading
│   └── ...
├── scripts/                    # Execution scripts
│   ├── train.sh               # Training script
│   ├── test.sh                # Testing script
│   └── test_clip.sh           # CLIP-specific testing
├── analysis/                   # Analysis and visualization tools
├── preprocessing/              # Data preprocessing pipeline
├── logs/                      # Training and testing logs
│   ├── training/              # Training logs and checkpoints
│   └── testing/               # Testing logs
├── results/                   # Experimental results
│   ├── predictions/           # Model predictions
│   ├── confusion_matrices/    # Confusion matrices
│   ├── attention/            # Attention visualizations
│   └── gradcam/              # GradCAM visualizations
├── docs/                     # Documentation and examples
├── datasets/                 # Dataset storage directory
└── figures/                  # Figures and charts
```

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n deepfake python=3.7.2
conda activate deepfake

# Option 1: Automatic installation (Recommended)
./install.sh

# Option 2: Manual installation
pip install torch==2.6.0 torchvision torchaudio
pip install transformers==4.50.0 timm==1.0.15
pip install opencv-python dlib scikit-learn tensorboard
```

### 2. Data Preparation

Place datasets in the `datasets/` directory. Supported datasets include:
- **FaceForensics++** (training dataset)
- **DFDC, DFDCP, DFD** (testing datasets)
- **Celeb-DF-v2, UADFV, FaceShifter** (testing datasets)

### 3. Model Training

```bash
# Using training script (CLIP ViT-B with LayerNorm tuning)
./scripts/train.sh

# CLIP ViT-B model training
python training/train.py \
    --detector_path ./training/config/detector/clip_enhanced.yaml \
    --train_dataset "FaceForensics++" \
    --test_dataset "DFDC"

# CLIP-STAN temporal model training
python training/train.py \
    --detector_path ./training/config/detector/clip_stan.yaml \
    --train_dataset "FaceForensics++" \
    --test_dataset "DFDC"
```

### 4. Model Testing

```bash
# Using testing script
./scripts/test.sh

# Or direct Python command for cross-dataset evaluation
python training/test.py \
    --detector_path ./training/config/detector/clip_enhanced.yaml \
    --test_dataset "DFDC" "DFDCP" "DFD" "Celeb-DF-v2" "UADFV" "FaceShifter" \
    --weights_path ./training/weights/clip_enhanced_best.pth
```

## Supported Models

### Primary CLIP-based Models
- **CLIP ViT-B/16**: Parameter-efficient fine-tuning with LayerNorm adaptation (0.0344% trainable params)
- **CLIP ViT-L/14**: Larger model variant for enhanced performance (best overall results)
- **CLIP-STAN**: Temporal extension with Spatial-Temporal Attention Networks (9.46% trainable params)

### Implemented Baseline Detector
- **Temporal Detector**: TimeSformer+EfficientB0

## Research Contributions

### Key Findings from Experiments

✅ **Superior Cross-Dataset Performance**: CLIP ViT-B achieves 90.9% average video-level AUC across 7 datasets  
✅ **Parameter Efficiency**: State-of-the-art results with only 0.0344% trainable parameters  
✅ **Vision-Language Advantage**: CLIP pre-training outperforms ImageNet pre-training by significant margins  
✅ **Temporal Extension Effectiveness**: CLIP-STAN achieves 89.5% average AUC with temporal attention  
✅ **Design Insights**: Backbone quality dominates over architectural complexity for generalization  
✅ **Temporal Analysis**: Vision-language models outperform traditional temporal approaches for cross-dataset generalization  

## Configuration

### Model Configuration
Model configuration files are located in `training/config/detector/` directory:
- **clip_enhanced.yaml**: CLIP ViT-B/16 with LayerNorm tuning
- **clip_stan.yaml**: CLIP-STAN temporal model configuration
- **timesformer.yaml**: TimeSformer+EfficientB0 baseline

### Script Configuration
Configure training/testing parameters in `scripts/` directory:
- Model architecture selection
- Dataset configuration
- Training hyperparameters
- Evaluation settings

## Implementation Details

### Key Technical Features
- **Parameter-Efficient Fine-Tuning**: LayerNorm tuning strategy preserves pre-trained representations
- **Cross-Dataset Evaluation**: Standardized evaluation across 7 benchmark datasets
- **DeepfakeBench Integration**: Unified framework for fair comparison with state-of-the-art methods
- **Comprehensive Analysis**: Frame-level and video-level AUC computation with detailed ablation studies

### Implemented Model Files

**Primary Contribution - CLIP-based Models:**
- `training/config/detector/clip_enhanced.yaml` + `training/detectors/clip_enhanced.py` (CLIP ViT-B)
- `training/config/detector/clip_stan.yaml` + `training/detectors/clip_stan.py` (CLIP-STAN)
- `training/dataset/clip_enhanced_dataset.py` - Custom dataset implementation for CLIP models

**Baseline Detector Implementation:**
- `training/config/detector/timesformer.yaml` + `training/detectors/timesformer_detector.py` (EffB0+TimeSformer)

**Infrastructure Files:**
- `scripts/train.sh` - Training script with CLIP model support
- `scripts/test.sh` - Cross-dataset evaluation script
- `install.sh` - Environment setup script

**Total Implementation:** 3 model implementations + 1 custom dataset + infrastructure files

## Experimental Insights

### Design Guidelines for Cross-Dataset Detection
Based on comprehensive experiments across 7 datasets:

1. **Prioritize backbone pre-training quality**: Strong, broadly pre-trained backbones (e.g., CLIP) yield larger cross-domain gains than architectural modifications
2. **Vision-language models show superior generalization**: CLIP-based approaches consistently outperform traditional CNN-based temporal methods
3. **Use parameter-efficient tuning**: When temporal cues in source datasets are limited, lightweight adaptation reduces negative transfer risk
4. **Validate signal strength before adding complexity**: Complex modules should be introduced only after verifying strong corresponding cues in source datasets

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Acknowledgments

This project is built upon the [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) framework. We thank the original authors for their contributions.

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@mastersthesis{ling2025deepfake,
  title={Video deepfake detection with focus on human face fakes},
  author={Ling, Maomao},
  year={2025},
  school={Warsaw University of Technology}
}
```

---

**Note**: This project is intended for academic research purposes only. Please do not use it for illegal activities.
