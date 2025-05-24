"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/training/FaceForensics++/ckpt_epoch_9_best.pth')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        # Calculate number of batches needed for 200 images
        batch_size = config['test_batchSize']
        num_batches = min(200 // batch_size + (1 if 200 % batch_size else 0), len(test_set) // batch_size + (1 if len(test_set) % batch_size else 0))
        
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    """
    Test the model on a dataset.
    
    Args:
        model: The model to test
        data_loader: DataLoader containing the test data
    """
    prediction_lists = []  # List to store prediction probabilities
    feature_lists = []    # List to store features (if needed)
    label_lists = []      # List to store ground truth labels
    
    # Calculate number of batches needed for 200 images
    batch_size = data_loader.batch_size
    num_batches = 40
    
    for i, data_dict in tqdm(enumerate(data_loader), total=num_batches):
        if i >= num_batches:
            break
            
        # get data
        # data: torch.Tensor, shape [B, C, H, W] where B=batch_size, C=3 (RGB), H=height, W=width
        # label: torch.Tensor, shape [B] containing binary labels (0 or 1)
        # mask: torch.Tensor or None, shape [B, 1, H, W] if present
        # landmark: torch.Tensor or None, shape [B, 81, 2] if present
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)  # Convert to binary labels
        
        # Get preprocessing function from model and process images
        preprocess = model.get_preprocessing()  # Returns CLIPImageProcessor
        
        # Convert tensor to PIL images for CLIP preprocessing
        # data: torch.Tensor [B, C, H, W] -> numpy.ndarray [B, C, H, W]
        data = data.cpu().numpy()
        # data: numpy.ndarray [B, C, H, W] -> [B, H, W, C] for PIL
        data = np.transpose(data, (0, 2, 3, 1))
        # Convert to PIL Images
        # data: List[PIL.Image.Image], length B
        data = [pil_image.fromarray(img) for img in data]
        
        # Process images using CLIP processor
        # processed_data: BatchFeature containing 'pixel_values' tensor or dict
        processed_data = preprocess(data)
        
        # Extract pixel values based on return type
        if hasattr(processed_data, 'pixel_values'):
            # If BatchFeature object
            processed_data = processed_data.pixel_values
        elif isinstance(processed_data, dict) and 'pixel_values' in processed_data:
            # If dictionary with pixel_values
            processed_data = processed_data['pixel_values']
        elif isinstance(processed_data, torch.Tensor):
            # If already a tensor
            pass
        else:
            raise TypeError(f"Unexpected return type from preprocess: {type(processed_data)}")
        
        # Ensure processed_data is a tensor
        if not isinstance(processed_data, torch.Tensor):
            processed_data = torch.tensor(processed_data)
        
        # move data to GPU and convert to float32
        # processed_data: torch.Tensor [B, C, H, W] on GPU with float32 dtype
        data_dict['image'] = processed_data.to(device).to(torch.float32)
        # label: torch.Tensor [B] on GPU
        data_dict['label'] = label.to(device)
        if mask is not None:
            # mask: torch.Tensor [B, 1, H, W] on GPU
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            # landmark: torch.Tensor [B, 81, 2] on GPU
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        # predictions: dict containing 'prob' tensor [B] with probabilities
        predictions = inference(model, data_dict)
        # Collect predictions and labels
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())

    # Convert predictions to binary labels using 0.5 threshold
    # pred_labels: numpy.ndarray [N] containing 0s and 1s
    pred_labels = (np.array(prediction_lists) > 0.5).astype(int)
    # true_labels: numpy.ndarray [N] containing ground truth labels
    true_labels = np.array(label_lists)
    
    # Calculate confusion matrix
    # cm: numpy.ndarray [2, 2] containing confusion matrix values
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Create directory for saving confusion matrices
    save_dir = 'confusion_matrices'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save confusion matrix plot
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{data_loader.dataset.__class__.__name__}.png'))
    plt.close()
    
    return np.array(prediction_lists), np.array(label_lists)

def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        print(f"\nDataset: {key}")
        print("Number of images:", len(data_dict['image']))
        print("Number of labels:", len(data_dict['label']))
        
        predictions_nps, label_nps = test_one_dataset(model, test_data_loaders[key])
        
        # 计算并打印混淆矩阵的详细指标
        pred_labels = (predictions_nps > 0.5).astype(int)
        cm = confusion_matrix(label_nps, pred_labels)
        tn, fp, fn, tp = cm.ravel()
        
        # 计算额外指标
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 打印详细指标
        tqdm.write(f"\nDetailed metrics for {key}:")
        tqdm.write(f"True Negatives (Real classified as Real): {tn}")
        tqdm.write(f"False Positives (Real classified as Fake): {fp}")
        tqdm.write(f"False Negatives (Fake classified as Real): {fn}")
        tqdm.write(f"True Positives (Fake classified as Fake): {tp}")
        tqdm.write(f"Accuracy: {accuracy:.4f}")
        tqdm.write(f"Precision: {precision:.4f}")
        tqdm.write(f"Recall: {recall:.4f}")
        tqdm.write(f"F1 Score: {f1:.4f}")
        
        # compute metric for each dataset
        print("\nPredictions shape:", predictions_nps.shape)
        print("Labels shape:", label_nps.shape)
        print("Number of processed images:", len(data_dict['image']))
        
        # 确保只使用处理过的图像
        processed_image_names = data_dict['image'][:len(predictions_nps)]
        print("Number of processed image names:", len(processed_image_names))
        
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=processed_image_names)
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"\ndataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    if weights_path:
        # Load checkpoint
        ckpt = torch.load(weights_path, map_location=device)
        
        # Get hyperparameters from checkpoint
        if 'hyper_parameters' in ckpt:
            hyper_params = ckpt['hyper_parameters']
            # Update config with hyperparameters
            config.update(hyper_params)
        
        # Initialize model with updated config
        model_class = DETECTOR[config['model_name']]
        model = model_class(config).to(device)
        
        # Load state dict
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            # Remove 'model.' prefix if it exists
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
        model_class = DETECTOR[config['model_name']]
        model = model_class(config).to(device)
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders)
    print('===> Test Done!')

if __name__ == '__main__':
    main()
