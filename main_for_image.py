import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import argparse
import logging
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torch.utils import data
from tqdm import tqdm
from mmengine import Config
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append(".")
import methods as model_zoo
from utils import scaler, scheduler, optimizer

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("main")

class KaggleCODDataset(data.Dataset):
    def __init__(self, root, image_sub, mask_sub, shape, is_train=True):
        self.image_root = os.path.join(root, image_sub)
        self.mask_root = os.path.join(root, mask_sub)
        self.is_train = is_train
        self.shape = shape
        self.image_list = sorted([f for f in os.listdir(self.image_root) if f.endswith(('.jpg', '.png'))])
        print(f"Found {len(self.image_list)} images in {self.image_root}")
        if is_train:
            self.transform = A.Compose([
                A.Resize(shape['h'], shape['w']),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(p=0.3),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(shape['h'], shape['w']),
                A.Normalize(),
                ToTensorV2()
            ])

    def __len__(self): return len(self.image_list)

    def __getitem__(self, index):
        name = self.image_list[index]
        img_path = os.path.join(self.image_root, name)
        mask_name = name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_root, mask_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
        augmented = self.transform(image=image, mask=mask)
        return dict(data={"image": augmented['image'], "mask": augmented['mask'].float().unsqueeze(0)}, info={"name": name})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="PvtV2B2_ZoomNeXt")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Folder to save checkpoints")
    # --- ADDED THIS LINE ---
    parser.add_argument("--load-from", type=str, default=None, help="Path to checkpoint to load")
    args = parser.parse_args()
    
    cfg = Config.fromfile(args.config)
    
    print(f"Creating Model: {args.model_name}")
    model_class = model_zoo.__dict__.get(args.model_name)
    model = model_class(num_frames=1, pretrained=True)
    model.cuda()
    
    # --- RESUME LOGIC ---
    if args.load_from:
        if os.path.exists(args.load_from):
            print(f"🔄 Resuming from checkpoint: {args.load_from}")
            # Load weights (strict=False to be safe with partial matches if any)
            state_dict = torch.load(args.load_from)
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"⚠️ Warning: Checkpoint {args.load_from} not found! Starting from scratch.")
    
    train_set = KaggleCODDataset(root=cfg.train.data.root_path, image_sub=cfg.train.data.train_image_path, mask_sub=cfg.train.data.train_mask_path, shape=cfg.train.data.shape, is_train=True)
    train_loader = data.DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, drop_last=True)
    
    optim = optimizer.construct_optimizer(model, initial_lr=cfg.train.lr, mode=cfg.train.optimizer.mode, group_mode=cfg.train.optimizer.group_mode, cfg=cfg.train.optimizer.cfg)
    sche = scheduler.Scheduler(optimizer=optim, num_iters=cfg.train.num_epochs * len(train_loader), epoch_length=len(train_loader), scheduler_cfg=cfg.train.scheduler)
    amp_scaler = scaler.Scaler(optim, use_fp16=cfg.train.use_amp)
    
    print(f"Starting Training for {cfg.train.num_epochs} epochs...")
    print(f"Checkpoints will be saved to: {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    total_step = 0
    for epoch in range(cfg.train.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.num_epochs}")
        for batch in pbar:
            images = batch['data']['image'].cuda()
            masks = batch['data']['mask'].cuda()
            masks = masks / 255.0
            
            # Multi-scale inputs
            images_l = images
            images_m = F.interpolate(images, scale_factor=0.75, mode='bilinear', align_corners=True)
            images_s = F.interpolate(images, scale_factor=0.5, mode='bilinear', align_corners=True)
            
            preds = model(data={"image_l": images_l, "image_m": images_m, "image_s": images_s, "mask": masks})
            
            if isinstance(preds, dict) and "loss" in preds: 
                loss = preds["loss"]
            else: 
                loss = torch.tensor(0.0, requires_grad=True).cuda()

            optim.zero_grad()
            amp_scaler.calculate_grad(loss)
            amp_scaler.update_grad()
            sche.step(total_step)
            total_step += 1
            pbar.set_postfix(loss=loss.item())
            
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"zoomnext_epoch_{epoch+1}.pth"))
        print(f"Saved checkpoint for epoch {epoch+1}")

if __name__ == "__main__":
    main()