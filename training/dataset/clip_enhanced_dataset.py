import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2
import numpy as np
from typing import Dict, Any, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CLIPEnhancedDataset(Dataset):
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        self.config = config
        self.split = split
        self.use_face_detection = config['data'].get('use_face_detection', True)
        
        # Setup face detector if needed
        if self.use_face_detection:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        # Setup augmentations
        self.transform = self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup image transformations based on config"""
        transforms = []
        
        # Add augmentations from config
        for aug_config in self.config['data'].get('augmentations', []):
            aug_type = aug_config.pop('type')
            if hasattr(A, aug_type):
                transforms.append(getattr(A, aug_type)(**aug_config))
        
        # Add ToTensor at the end
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def _detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect and crop face from image"""
        if not self.use_face_detection:
            return image
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Get the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Add margin
        margin = int(0.1 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        return image[y1:y2, x1:x2]
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Get image path and label
        image_path = self.image_paths[index]
        label = self.labels[index]
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect and crop face
        face_image = self._detect_face(image)
        if face_image is None:
            # If no face detected, use the original image
            face_image = image
        
        # Apply transformations
        transformed = self.transform(image=face_image)
        image = transformed['image']
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': image_path
        }
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def set_data(self, image_paths: list, labels: list):
        """Set dataset data"""
        self.image_paths = image_paths
        self.labels = labels 