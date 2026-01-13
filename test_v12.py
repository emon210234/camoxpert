"""
Test script for CamoXpertV12 with enhanced architecture.
This tests the improved model with boundary refinement and enhanced loss.
"""
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import py_sod_metrics
import prettytable as pt
import sys

# Configuration
PROJECT_ROOT = os.getcwd()
DATA_ROOT = os.path.join(PROJECT_ROOT, "data/COD-TestDataset/COD-TestDataset")

# Model Configuration
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints_v12/camoxpert_v12_epoch_120.pth")
MODEL_NAME = "CamoXpertV12 (B5 + ASW + BRM + Enhanced Loss)"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_SIZE = 448

# Dataset Definitions
DATASETS = {
    "CAMO":      {"sub_path": "CAMO",      "img_dir": "Image", "gt_dir": "Mask", "ext": ".jpg"},
    "CHAMELEON": {"sub_path": "CHAMELEON", "img_dir": "Image", "gt_dir": "Mask", "ext": ".jpg"},
    "COD10K":    {"sub_path": "COD10K",    "img_dir": "Image", "gt_dir": "Mask", "ext": ".jpg"},
    "NC4K":      {"sub_path": "NC4K",      "img_dir": "Imgs",  "gt_dir": "GT",   "ext": ".jpg"}
}

# Import Model
sys.path.append(PROJECT_ROOT)
try:
    from methods.zoomnext.camoxpert_v12 import CamoXpertV12_Base
except ImportError:
    print("⚠️  Could not import CamoXpertV12. Check methods/zoomnext/camoxpert_v12.py")
    sys.exit(1)


def evaluate_dataset_mstta(model, name, config):
    """
    Evaluate on a single dataset using Multi-Scale Test-Time Augmentation.
    """
    base_path = os.path.join(DATA_ROOT, config["sub_path"])
    img_dir = os.path.join(base_path, config["img_dir"])
    gt_dir = os.path.join(base_path, config["gt_dir"])
    
    if not os.path.exists(img_dir):
        print(f"⚠️  Skipping {name}: Folder not found at {img_dir}")
        return None

    image_list = [f for f in os.listdir(img_dir) if f.endswith(config["ext"])]
    if len(image_list) == 0:
        print(f"⚠️  No images found in {name}. Check folder structure.")
        return None

    MAE = py_sod_metrics.MAE()
    Smeasure = py_sod_metrics.Smeasure()
    
    # Multi-Scale Test Time Augmentation
    SCALES = [0.75, 1.0, 1.25]
    
    for img_name in tqdm(image_list, desc=f"Testing {name}", leave=False):
        # Load Image
        image_path = os.path.join(img_dir, img_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Ground Truth
        gt_name = img_name.replace(config["ext"], ".png")
        gt_path = os.path.join(gt_dir, gt_name)
        if not os.path.exists(gt_path):
            gt_path = gt_path.replace(".png", ".jpg")
        if not os.path.exists(gt_path):
            continue
            
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        gt_mask[gt_mask > 0] = 255
        gt_h, gt_w = gt_mask.shape[:2]
        
        preds_list = []

        # Multi-Scale Inference Loop
        for scale in SCALES:
            target_h, target_w = int(BASE_SIZE * scale), int(BASE_SIZE * scale)
            transform = A.Compose([
                A.Resize(target_h, target_w),
                A.Normalize(),
                ToTensorV2()
            ])
            
            # 1. Standard View
            aug = transform(image=image)
            input_tensor = aug['image'].unsqueeze(0).to(DEVICE)
            
            # Multi-scale inputs
            img_l = input_tensor
            img_m = F.interpolate(input_tensor, scale_factor=0.75, mode='bilinear')
            img_s = F.interpolate(input_tensor, scale_factor=0.5, mode='bilinear')
            
            with torch.no_grad():
                out = model(data={"image_l": img_l, "image_m": img_m, "image_s": img_s})
                p = out['pred']
                p = F.interpolate(p, size=(gt_h, gt_w), mode='bilinear', align_corners=False)
                preds_list.append(p)
            
            # 2. Flipped View (Horizontal Flip)
            image_flipped = cv2.flip(image, 1)
            aug_f = transform(image=image_flipped)
            input_flip = aug_f['image'].unsqueeze(0).to(DEVICE)
            
            img_l_f = input_flip
            img_m_f = F.interpolate(input_flip, scale_factor=0.75, mode='bilinear')
            img_s_f = F.interpolate(input_flip, scale_factor=0.5, mode='bilinear')
            
            with torch.no_grad():
                out_f = model(data={"image_l": img_l_f, "image_m": img_m_f, "image_s": img_s_f})
                p_f = out_f['pred']
                p_f = torch.flip(p_f, dims=[3])  # Flip back
                p_f = F.interpolate(p_f, size=(gt_h, gt_w), mode='bilinear', align_corners=False)
                preds_list.append(p_f)

        # Average all 6 predictions (3 scales × 2 views)
        pred_final = torch.mean(torch.stack(preds_list), dim=0)
        
        # Normalize & Update Metrics
        pred_map = pred_final.squeeze().cpu().numpy()
        pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
        pred_uint8 = (pred_map * 255).astype(np.uint8)
        gt_uint8 = gt_mask.astype(np.uint8)
        
        MAE.step(pred_uint8, gt_uint8)
        Smeasure.step(pred_uint8, gt_uint8)
        
    return {"Sm": Smeasure.get_results()['sm'], "MAE": MAE.get_results()['mae']}


def main():
    print(f"--- 🚀 CAMOXPERT V12 EVALUATION: {MODEL_NAME} ---")
    print(f"Device: {DEVICE}")
    
    # Load Model
    try:
        model = CamoXpertV12_Base(num_frames=1, pretrained=False)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return

    # Load Checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        try:
            state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict, strict=True)
            print(f"✓ Weights loaded from: {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load checkpoint: {e}")
            print("   Proceeding with random weights for architecture verification...")
    else:
        print(f"⚠️  Checkpoint not found: {CHECKPOINT_PATH}")
        print("   Proceeding with random weights for architecture verification...")
    
    model.to(DEVICE)
    model.eval()
    
    # Evaluate on all datasets
    results = {}
    for name, config in DATASETS.items():
        print(f"\nEvaluating on {name}...")
        results[name] = evaluate_dataset_mstta(model, name, config)
    
    # Display Results
    tb = pt.PrettyTable()
    tb.field_names = ["Dataset", "S-measure (↑)", "MAE (↓)"]
    
    for name in DATASETS.keys():
        res = results[name]
        if res:
            tb.add_row([name, f"{res['Sm']:.4f}", f"{res['MAE']:.4f}"])
        else:
            tb.add_row([name, "Not Found", "Not Found"])
    
    print("\n" + "="*50)
    print("         CAMOXPERT V12 RESULTS")
    print("="*50)
    print(str(tb))
    print("="*50)
    print("\nImprovements in V12:")
    print("  ✓ Boundary Refinement Module")
    print("  ✓ Adaptive Scale Weighting")
    print("  ✓ Enhanced Loss (BCE + IoU + SSIM + Edge)")
    print("  ✓ Multi-component loss monitoring")
    print("="*50)


if __name__ == "__main__":
    main()
