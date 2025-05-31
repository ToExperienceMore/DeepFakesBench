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
import sklearn.metrics as metrics

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

    print(f"batch_size: {config['test_batchSize']}")
    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def find_optimal_threshold(y_true, y_pred):
    """
    Find the optimal threshold that maximizes accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        
    Returns:
        optimal_threshold: The threshold that gives the best accuracy
        best_accuracy: The best accuracy achieved
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    accuracies = []
    for threshold in thresholds:
        pred_labels = (y_pred >= threshold).astype(int)
        accuracy = (pred_labels == y_true).mean()
        accuracies.append(accuracy)
    
    best_idx = np.argmax(accuracies)
    optimal_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]
    
    return optimal_threshold, best_accuracy

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
    #num_batches = 10
    print(f"test images: {len(data_loader.dataset)}")
    
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        #if i >= num_batches:
        #    break
            
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        
        data_dict['image'] = data.to(device)
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

    # Convert to numpy arrays
    predictions_nps = np.array(prediction_lists)
    label_nps = np.array(label_lists)
    
    # Find optimal threshold
    optimal_threshold, best_accuracy = find_optimal_threshold(label_nps, predictions_nps)
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"Best accuracy with optimal threshold: {best_accuracy:.4f}")
    
    # Use optimal threshold for predictions
    pred_labels = (predictions_nps >= optimal_threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(label_nps, pred_labels)
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix (Threshold: {optimal_threshold:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Create directory for saving confusion matrices
    save_dir = 'confusion_matrices'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save confusion matrix plot
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{data_loader.dataset.__class__.__name__}.png'))
    plt.close()
    
    # Calculate metrics using optimal threshold
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    tqdm.write(f"\nDetailed metrics with optimal threshold:")
    tqdm.write(f"True Negatives (Real classified as Real): {tn}")
    tqdm.write(f"False Positives (Real classified as Fake): {fp}")
    tqdm.write(f"False Negatives (Fake classified as Real): {fn}")
    tqdm.write(f"True Positives (Fake classified as Fake): {tp}")
    tqdm.write(f"Accuracy: {accuracy:.4f}")
    tqdm.write(f"Precision: {precision:.4f}")
    tqdm.write(f"Recall: {recall:.4f}")
    tqdm.write(f"F1 Score: {f1:.4f}")
    
    return predictions_nps, label_nps

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
        #print("processed_image_names:", processed_image_names)
        print("predictions_nps:", predictions_nps)
        print("label_nps:", label_nps)
        
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
        
        # Load state dict - handle both direct state_dict and wrapped state_dict
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
            
        model.load_state_dict(state_dict, strict=True)
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
