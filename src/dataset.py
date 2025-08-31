import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

class COD10KDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=256, augment=False):
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment
        
        # Define paths based on split
        if split in ['train', 'val']:
            image_dir = os.path.join(root_dir, 'train', 'Images')
            mask_dir = os.path.join(root_dir, 'train', 'GT_Object')
        else:  # test
            image_dir = os.path.join(root_dir, 'test', 'Images')
            mask_dir = os.path.join(root_dir, 'test', 'GT_Object')
        
        # Get all image paths
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                                  if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # Create corresponding mask paths
        self.mask_paths = []
        for img_path in self.image_paths:
            img_name = os.path.basename(img_path)
            # Remove extension and add appropriate suffix if needed
            base_name = os.path.splitext(img_name)[0]
            mask_name = base_name + '.png'  # Masks are typically PNG
            mask_path = os.path.join(mask_dir, mask_name)
            self.mask_paths.append(mask_path)
        
        # For train/val split, we need to split the training data
        if split in ['train', 'val']:
            train_paths, val_paths, train_mask_paths, val_mask_paths = train_test_split(
                self.image_paths, self.mask_paths, test_size=0.2, random_state=42
            )
            if split == 'train':
                self.image_paths = train_paths
                self.mask_paths = train_mask_paths
            else:  # val
                self.image_paths = val_paths
                self.mask_paths = val_mask_paths
        
        # Augmentations
        self.transform = self.get_transforms(augment)
        
    def get_transforms(self, augment):
        if augment:
            return A.Compose([
                A.Rotate(limit=30, p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.CLAHE(p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.MotionBlur(blur_limit=7, p=0.5),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], 0)  # Read as grayscale
        
        # Check if mask was loaded correctly
        if mask is None:
            print(f"Warning: Could not load mask at {self.mask_paths[idx]}")
            # Create a blank mask as fallback
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Normalize mask to 0-1 range
        mask = (mask > 0).astype(np.float32)
        
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].float()
        
        return image, mask
