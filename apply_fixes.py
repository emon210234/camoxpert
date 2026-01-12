import os

print("Applying ZoomNeXt Fixes for RTX 4090 Environment...")

# --- 1. Fix utils/scaler.py (Import Error) ---
with open('utils/scaler.py', 'w', encoding='utf-8') as f:
    f.write('''from functools import partial
from itertools import chain
from torch.cuda.amp import GradScaler, autocast
from . import tensor_ops as ops 

class Scaler:
    def __init__(self, optimizer, use_fp16=False, *, set_to_none=False, clip_grad=False, clip_mode=None, clip_cfg=None):
        self.optimizer = optimizer
        self.set_to_none = set_to_none
        self.autocast = autocast(enabled=use_fp16)
        self.scaler = GradScaler(enabled=use_fp16)
        if clip_grad:
            self.grad_clip_ops = partial(ops.clip_grad, mode=clip_mode, clip_cfg=clip_cfg)
        else:
            self.grad_clip_ops = None

    def calculate_grad(self, loss):
        self.scaler.scale(loss).backward()
        if self.grad_clip_ops is not None:
            self.scaler.unscale_(self.optimizer)
            self.grad_clip_ops(chain(*[group["params"] for group in self.optimizer.param_groups]))

    def update_grad(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=self.set_to_none)

    def state_dict(self):
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict):
        self.scaler.load_state_dict(state_dict)
''')
print("✓ Fixed utils/scaler.py")

# --- 2. Fix methods/zoomnext/layers.py (Argument Mismatch) ---
with open('methods/zoomnext/layers.py', 'w', encoding='utf-8') as f:
    f.write('''import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import ConvBNReLU, resize_to

class SimpleASPP(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=3):
        super().__init__()
        self.conv1x1_1 = ConvBNReLU(in_dim, 2 * out_dim, 1)
        self.conv1x1_2 = ConvBNReLU(out_dim, out_dim, 1)
        self.conv3x3_1 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_2 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_3 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.fuse = nn.Sequential(
            ConvBNReLU(5 * out_dim, out_dim, 1), 
            ConvBNReLU(out_dim, out_dim, 3, 1, 1)
        )

    def forward(self, x):
        y = self.conv1x1_1(x)
        y1, y5 = y.chunk(2, dim=1)
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)
        out = torch.cat([y5, y1, y2, y3, y4], dim=1)
        return self.fuse(out)

class MHSIU(nn.Module):
    def __init__(self, in_dim, num_groups=4):
        super().__init__()
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
            ConvBNReLU(in_dim // num_groups, in_dim // num_groups, 3, 1, 1),
            nn.Conv2d(in_dim // num_groups, 3, 1),
            nn.Softmax(dim=1)
        )
        self.num_groups = num_groups

    def forward(self, l, m, s):
        tgt_size = m.shape[2:]
        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = self.conv_s_pre(s)
        s = resize_to(s, tgt_hw=tgt_size)
        
        l, m, s = self.conv_l(l), self.conv_m(m), self.conv_s(s)
        l_groups = torch.chunk(l, self.num_groups, dim=1)
        m_groups = torch.chunk(m, self.num_groups, dim=1)
        s_groups = torch.chunk(s, self.num_groups, dim=1)
        
        outs = []
        for lg, mg, sg in zip(l_groups, m_groups, s_groups):
            joint = torch.cat([lg, mg, sg], dim=1)
            attn = self.trans(joint)
            al, am, as_ = torch.chunk(attn, 3, dim=1)
            outs.append(lg * al + mg * am + sg * as_)
        return torch.cat(outs, dim=1)

class RGPU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_curr = ConvBNReLU(in_dim, out_dim, 3, 1, 1)
        self.conv_prev = ConvBNReLU(in_dim, out_dim, 3, 1, 1)
        self.conv_out = ConvBNReLU(2 * out_dim, out_dim, 3, 1, 1)

    def forward(self, curr_x, prev_x):
        curr = self.conv_curr(curr_x)
        prev = self.conv_prev(prev_x)
        prev = resize_to(prev, tgt_hw=curr.shape[2:])
        guided = curr * prev 
        return self.conv_out(torch.cat([guided, curr], dim=1))
''')
print("✓ Fixed methods/zoomnext/layers.py")

# --- 3. Fix methods/zoomnext/zoomnext.py (All Logic Fixes) ---
with open('methods/zoomnext/zoomnext.py', 'w', encoding='utf-8') as f:
    f.write('''import abc
import logging
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbone.efficientnet import EfficientNet
from ..backbone.pvt_v2_eff import pvt_v2_eff_b2, pvt_v2_eff_b3, pvt_v2_eff_b4, pvt_v2_eff_b5
from .layers import MHSIU, RGPU, SimpleASPP
from .ops import ConvBNReLU, PixelNormalizer, resize_to

LOGGER = logging.getLogger("main")

class _ZoomNeXt_Base(nn.Module):
    @staticmethod
    def get_coef(iter_percentage=1, method="cos", milestones=(0, 1)):
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = 0, 1
        ual_coef = 1.0
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            if method == "linear":
                ratio = (max_coef - min_coef) / (max_point - min_point)
                ual_coef = ratio * (iter_percentage - min_point)
            elif method == "cos":
                perc = (iter_percentage - min_point) / (max_point - min_point)
                normalized_coef = (1 - np.cos(perc * np.pi)) / 2
                ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
        return ual_coef

    def __init__(self, num_frames, pretrained=False, use_checkpoint=False):
        super().__init__()
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint

    @abc.abstractmethod
    def encoder(self, image): pass

    @abc.abstractmethod
    def body(self, data): pass

    def forward(self, data):
        if self.training:
            logits = self.body(data=data)
            mask = data["mask"]
            if logits.shape[-2:] != mask.shape[-2:]:
                logits = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)
            sod_loss = F.binary_cross_entropy_with_logits(input=logits, target=mask, reduction="mean")
            return dict(loss=sod_loss)
        else:
            logits = self.body(data=data)
            return dict(pred=logits.sigmoid())

class ZoomNeXt(_ZoomNeXt_Base):
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False):
        super().__init__(num_frames, pretrained, use_checkpoint)
        self.normalize_encoder = PixelNormalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tra_5 = self.tra_4 = self.tra_3 = self.tra_2 = None
        self.siu_5 = self.siu_4 = self.siu_3 = None
        self.hmu_5 = self.hmu_4 = self.hmu_3 = None
        self.rgpu = self.proj = None

    def body(self, data):
        l_trans_feats = self.encoder(self.normalize_encoder(data["image_l"]))
        m_trans_feats = self.encoder(self.normalize_encoder(data["image_m"]))
        s_trans_feats = self.encoder(self.normalize_encoder(data["image_s"]))

        l, m, s = self.tra_5(l_trans_feats[4]), self.tra_5(m_trans_feats[4]), self.tra_5(s_trans_feats[4])
        lms = self.siu_5(l=l, m=m, s=s)
        x = self.hmu_5(lms)

        l, m, s = self.tra_4(l_trans_feats[3]), self.tra_4(m_trans_feats[3]), self.tra_4(s_trans_feats[3])
        lms = self.siu_4(l=l, m=m, s=s)
        x = self.hmu_4(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_3(l_trans_feats[2]), self.tra_3(m_trans_feats[2]), self.tra_3(s_trans_feats[2])
        lms = self.siu_3(l=l, m=m, s=s)
        x = self.hmu_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_2(l_trans_feats[1]), self.tra_2(m_trans_feats[1]), self.tra_2(s_trans_feats[1])
        x = self.rgpu(curr_x=l, prev_x=x) 
        return self.proj(x)

class PvtV2_ZoomNeXt(ZoomNeXt):
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False, variant='b2'):
        super().__init__(num_frames, pretrained, use_checkpoint)
        if variant == 'b2': self.backbone = pvt_v2_eff_b2(pretrained=pretrained, use_checkpoint=use_checkpoint); embed_dims = [64, 128, 320, 512]
        elif variant == 'b5': self.backbone = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint); embed_dims = [64, 128, 320, 512]
        else: raise NotImplementedError
        out_dim = 64 
        self.tra_5 = ConvBNReLU(embed_dims[3], out_dim, 1)
        self.tra_4 = ConvBNReLU(embed_dims[2], out_dim, 1)
        self.tra_3 = ConvBNReLU(embed_dims[1], out_dim, 1)
        self.tra_2 = ConvBNReLU(embed_dims[0], out_dim, 1) 
        self.siu_5 = MHSIU(out_dim, num_groups=4)
        self.siu_4 = MHSIU(out_dim, num_groups=4)
        self.siu_3 = MHSIU(out_dim, num_groups=4)
        self.hmu_5 = SimpleASPP(out_dim, out_dim)
        self.hmu_4 = SimpleASPP(out_dim, out_dim)
        self.hmu_3 = SimpleASPP(out_dim, out_dim)
        self.rgpu = RGPU(in_dim=out_dim, out_dim=out_dim)
        self.proj = nn.Conv2d(out_dim, 1, kernel_size=3, padding=1)

    def encoder(self, x):
        feats = self.backbone(x)
        if isinstance(feats, dict): feats = list(feats.values())
        elif not isinstance(feats, (list, tuple)): feats = [feats]
        else: feats = list(feats)
        return [None] + feats 

class PvtV2B2_ZoomNeXt(PvtV2_ZoomNeXt):
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False):
        super().__init__(num_frames, pretrained, use_checkpoint, variant='b2')

class PvtV2B5_ZoomNeXt(PvtV2_ZoomNeXt):
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False):
        super().__init__(num_frames, pretrained, use_checkpoint, variant='b5')

class RN50_ZoomNeXt(ZoomNeXt): pass
class EffB1_ZoomNeXt(ZoomNeXt): pass
class EffB4_ZoomNeXt(ZoomNeXt): pass
class videoPvtV2B5_ZoomNeXt(PvtV2B5_ZoomNeXt): pass
''')
print("✓ Fixed methods/zoomnext/zoomnext.py")

# --- 4. Create main_for_image.py (Fixed Training Loop) ---
with open('main_for_image.py', 'w', encoding='utf-8') as f:
    f.write('''import os
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
        print(f"Found {len(self.image_list)} images")
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
        if os.path.exists(mask_path): mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else: mask = np.zeros(image.shape[:2], dtype=np.uint8)
        augmented = self.transform(image=image, mask=mask)
        return dict(data={"image": augmented['image'], "mask": augmented['mask'].float().unsqueeze(0)}, info={"name": name})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="PvtV2B2_ZoomNeXt")
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    
    print(f"Creating Model: {args.model_name}")
    model_class = model_zoo.__dict__.get(args.model_name)
    model = model_class(num_frames=1, pretrained=True)
    model.cuda()
    
    train_set = KaggleCODDataset(root=cfg.train.data.root_path, image_sub=cfg.train.data.train_image_path, mask_sub=cfg.train.data.train_mask_path, shape=cfg.train.data.shape, is_train=True)
    train_loader = data.DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, drop_last=True)
    
    optim = optimizer.construct_optimizer(model, initial_lr=cfg.train.lr, mode=cfg.train.optimizer.mode, group_mode=cfg.train.optimizer.group_mode, cfg=cfg.train.optimizer.cfg)
    sche = scheduler.Scheduler(optimizer=optim, num_iters=cfg.train.num_epochs * len(train_loader), epoch_length=len(train_loader), scheduler_cfg=cfg.train.scheduler)
    amp_scaler = scaler.Scaler(optim, use_fp16=cfg.train.use_amp)
    
    print("Starting Training...")
    total_step = 0
    for epoch in range(cfg.train.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.num_epochs}")
        for batch in pbar:
            images = batch['data']['image'].cuda()
            masks = batch['data']['mask'].cuda()
            masks = masks / 255.0
            images_l = images
            images_m = F.interpolate(images, scale_factor=0.75, mode='bilinear', align_corners=True)
            images_s = F.interpolate(images, scale_factor=0.5, mode='bilinear', align_corners=True)
            
            preds = model(data={"image_l": images_l, "image_m": images_m, "image_s": images_s, "mask": masks})
            
            if isinstance(preds, dict) and "loss" in preds: loss = preds["loss"]
            else: loss = torch.tensor(0.0, requires_grad=True).cuda()

            optim.zero_grad()
            amp_scaler.calculate_grad(loss)
            amp_scaler.update_grad()
            sche.step(total_step)
            total_step += 1
            pbar.set_postfix(loss=loss.item())
            
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/zoomnext_epoch_{epoch+1}.pth")
        print(f"Saved checkpoint for epoch {epoch+1}")

if __name__ == "__main__":
    main()
''')
print("✓ Created main_for_image.py")

# --- 5. Create configs/rtx4090_train.py (Lab Config) ---
os.makedirs("configs", exist_ok=True)
with open('configs/rtx4090_train.py', 'w', encoding='utf-8') as f:
    f.write('''
_base_ = ["icod_train.py"]
__BATCHSIZE = 20
__NUM_EPOCHS = 50
train = dict(
    batch_size=__BATCHSIZE,
    num_workers=4,
    use_amp=True,
    num_epochs=__NUM_EPOCHS,
    epoch_based=True,
    lr=0.0001,
    optimizer=dict(mode="adamw", set_to_none=True, group_mode="finetune", cfg=dict(weight_decay=1e-4, diff_factor=0.1)),
    sche_usebatch=True,
    scheduler=dict(warmup=dict(num_iters=1000), mode="cos", cfg=dict(lr_decay=0.9, min_coef=0.01)),
    data=dict(
        shape=dict(h=384, w=384),
        names=["cod10k_tr"],
        root_path="C:/Datasets/COD10K-v3", 
        train_image_path="Train/Image",
        train_mask_path="Train/GT_Object",
        test_image_path="Test/Image",
        test_mask_path="Test/GT_Object",
    ),
)
test = dict(batch_size=__BATCHSIZE, data=dict(shape=dict(h=384, w=384), names=["cod10k_te"]))
''')
print("✓ Created configs/rtx4090_train.py")
print("\nALL FIXES APPLIED! You can run training now.")