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

# --- USER CONFIGURATION ---
PROJECT_ROOT = os.getcwd() 

# DATA PATH
DATA_ROOT = os.path.join(PROJECT_ROOT, "data/COD-TestDataset/COD-TestDataset")

# CHECKPOINT: V11 (B5 + Mamba + MoE)
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints_v11/zoomnext_epoch_120.pth")
MODEL_NAME = "CamoXpertV11 (B5 + Mamba Scan + Spectral MoE)"
DEVICE = 'cuda'

# Base Resolution
BASE_SIZE = 448 

# --- DATASET DEFINITIONS ---
# ⚠️ VERIFY FOLDER NAMES (Image/Mask vs Imgs/GT) IN YOUR DISK ⚠️
DATASETS = {
    "CAMO":      {"sub_path": "CAMO",      "img_dir": "Image", "gt_dir": "Mask", "ext": ".jpg"},
    "CHAMELEON": {"sub_path": "CHAMELEON", "img_dir": "Image", "gt_dir": "Mask", "ext": ".jpg"},
    "COD10K":    {"sub_path": "COD10K",    "img_dir": "Image", "gt_dir": "Mask", "ext": ".jpg"},
    # NC4K usually has different folder names in some downloads
    "NC4K":      {"sub_path": "NC4K",      "img_dir": "Imgs",  "gt_dir": "GT",   "ext": ".jpg"} 
}

# --- IMPORT MODEL ---
sys.path.append(PROJECT_ROOT)
try:
    from methods.zoomnext.zoomnext import CamoXpertV11
except ImportError:
    print("⚠️  Could not import CamoXpertV11. Check methods/zoomnext/zoomnext.py")
    sys.path.append(os.path.join(PROJECT_ROOT, "methods"))
    from zoomnext.zoomnext import CamoXpertV11

def evaluate_dataset_mstta(model, name, config):
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
    
    # SOTA STRATEGY: Multi-Scale Test Time Augmentation
    # Scales: 0.75 (Context), 1.0 (Standard), 1.25 (Detail)
    SCALES = [0.75, 1.0, 1.25] 
    
    for img_name in tqdm(image_list, desc=f"Testing {name}", leave=False):
        # Load Image
        image_path = os.path.join(img_dir, img_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load GT
        gt_name = img_name.replace(config["ext"], ".png")
        gt_path = os.path.join(gt_dir, gt_name)
        if not os.path.exists(gt_path): gt_path = gt_path.replace(".png", ".jpg")
        if not os.path.exists(gt_path): continue
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None: continue
        gt_mask[gt_mask > 0] = 255 
        gt_h, gt_w = gt_mask.shape[:2]
        
        preds_list = []

        # --- MULTI-SCALE INFERENCE LOOP ---
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
            
            # Inputs
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
                p_f = torch.flip(p_f, dims=[3]) # Flip back to original
                p_f = F.interpolate(p_f, size=(gt_h, gt_w), mode='bilinear', align_corners=False)
                preds_list.append(p_f)

        # Average all 6 predictions (3 scales * 2 views)
        pred_final = torch.mean(torch.stack(preds_list), dim=0)
        
        # Normalize & Metric Update
        pred_map = pred_final.squeeze().cpu().numpy()
        pred_map = (pred_map - pred_map.min()) / (pred_map.max() + 1e-8)
        pred_uint8 = (pred_map * 255).astype(np.uint8)
        gt_uint8 = gt_mask.astype(np.uint8)
        
        MAE.step(pred_uint8, gt_uint8)
        Smeasure.step(pred_uint8, gt_uint8)
        
    return {"Sm": Smeasure.get_results()['sm'], "MAE": MAE.get_results()['mae']}

def main():
    print(f"--- 🚀 FINAL V11 EVALUATION: {MODEL_NAME} ---")
    
    # Load V11
    try:
        model = CamoXpertV11(num_frames=1, pretrained=False)
    except NameError:
        print("❌ Error: CamoXpertV11 class not found.")
        return

    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH), strict=True)
        print(f"✓ Weights Loaded: {CHECKPOINT_PATH}")
    else:
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        return
    model.to(DEVICE)
    model.eval()
    
    results = {}
    for name, config in DATASETS.items():
        results[name] = evaluate_dataset_mstta(model, name, config)
    
    tb = pt.PrettyTable()
    tb.field_names = ["Dataset", "S-measure (↑)", "MAE (↓)"]
    for name in DATASETS.keys():
        res = results[name]
        if res: tb.add_row([name, f"{res['Sm']:.4f}", f"{res['MAE']:.4f}"])
        else: tb.add_row([name, "Not Found", "Not Found"])
            
    print("\n" + "="*45)
    print(str(tb))
    print("="*45)

if __name__ == "__main__":
    main()