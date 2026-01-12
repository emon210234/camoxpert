import os

print("Restoring missing utility files for ZoomNeXt...")

# Ensure utils directory exists
os.makedirs("utils", exist_ok=True)

# --- 1. utils/tensor_ops.py ---
with open("utils/tensor_ops.py", "w", encoding="utf-8") as f:
    f.write('''import torch
import torch.nn.functional as F

def rescale_2x(x: torch.Tensor, scale_factor=2):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)

def resize_to(x: torch.Tensor, tgt_hw: tuple):
    return F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)

def clip_grad(params, mode, clip_cfg: dict):
    if mode == "norm":
        if "max_norm" not in clip_cfg:
            raise ValueError("`clip_cfg` must contain `max_norm`.")
        torch.nn.utils.clip_grad_norm_(
            params,
            max_norm=clip_cfg.get("max_norm"),
            norm_type=clip_cfg.get("norm_type", 2.0),
        )
    elif mode == "value":
        if "clip_value" not in clip_cfg:
            raise ValueError("`clip_cfg` must contain `clip_value`.")
        torch.nn.utils.clip_grad_value_(params, clip_value=clip_cfg.get("clip_value"))
    else:
        raise NotImplementedError
''')
print("✓ Created utils/tensor_ops.py")

# --- 2. utils/array_ops.py ---
with open("utils/array_ops.py", "w", encoding="utf-8") as f:
    f.write('''import os
import cv2
import numpy as np

def minmax(data_array: np.ndarray, up_bound: float = None) -> np.ndarray:
    if up_bound is not None:
        data_array = data_array / up_bound
    max_value = data_array.max()
    min_value = data_array.min()
    if max_value != min_value:
        data_array = (data_array - min_value) / (max_value - min_value)
    return data_array
''')
print("✓ Created utils/array_ops.py")

# --- 3. utils/image.py ---
with open("utils/image.py", "w", encoding="utf-8") as f:
    f.write('''import cv2
import numpy as np
from .array_ops import minmax

def read_gray_array(path, div_255=False, to_normalize=False, thr=-1, dtype=np.float32) -> np.ndarray:
    assert path.endswith(".jpg") or path.endswith(".png"), path
    gray_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert gray_array is not None, f"Image Not Found: {path}"
    if div_255: gray_array = gray_array / 255
    if to_normalize: gray_array = minmax(gray_array, up_bound=255)
    if thr >= 0: gray_array = gray_array > thr
    return gray_array.astype(dtype)

def read_color_array(path: str):
    bgr_array = cv2.imread(path)
    assert bgr_array is not None, f"Image Not Found: {path}"
    return cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
''')
print("✓ Created utils/image.py")

# --- 4. utils/params.py ---
with open("utils/params.py", "w", encoding="utf-8") as f:
    f.write('''import os
import torch

def save_weight(save_path, model):
    print(f"Saving weight '{save_path}'")
    if isinstance(model, dict):
        model_state = model
    else:
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(model_state, save_path)
    print(f"Saved weight '{save_path}'")

def load_weight(load_path, model, *, strict=True, skip_unmatched_shape=False):
    assert os.path.exists(load_path), load_path
    model_params = model.state_dict()
    for k, v in torch.load(load_path, map_location="cpu").items():
        if k.endswith("module."): k = k[7:]
        if skip_unmatched_shape and k in model_params and v.shape != model_params[k].shape:
            continue
        model_params[k] = v
    model.load_state_dict(model_params, strict=strict)
''')
print("✓ Created utils/params.py")

# --- 5. utils/scheduler.py ---
with open("utils/scheduler.py", "w", encoding="utf-8") as f:
    f.write('''import math
from bisect import bisect_right

def linear_increase(low_bound, up_bound, percentage):
    return low_bound + (up_bound - low_bound) * percentage

def cos_anneal(low_bound, up_bound, percentage):
    cos_percentage = (1 + math.cos(math.pi * percentage)) / 2.0
    return linear_increase(low_bound, up_bound, percentage=cos_percentage)

def poly_anneal(low_bound, up_bound, percentage, lr_decay):
    poly_percentage = pow((1 - percentage), lr_decay)
    return linear_increase(low_bound, up_bound, percentage=poly_percentage)

def linear_anneal(low_bound, up_bound, percentage):
    return linear_increase(low_bound, up_bound, percentage=1 - percentage)

class Scheduler:
    def __init__(self, optimizer, num_iters, epoch_length, scheduler_cfg):
        self.optimizer = optimizer
        self.num_iters = num_iters
        self.epoch_length = epoch_length
        self.scheduler_cfg = scheduler_cfg
        self.warmup_cfg = scheduler_cfg["warmup"]
        self.mode = self.scheduler_cfg["mode"]
        self.learning_rates = [group["lr"] for group in self.optimizer.param_groups]

    def step(self, curr_idx):
        if curr_idx < self.warmup_cfg["num_iters"]:
            percentage = curr_idx / self.warmup_cfg["num_iters"]
            self._update_lr(self.warmup_cfg.get("mode", "linear"), self.warmup_cfg.get("initial_coef", 0.01), 1.0, percentage)
        else:
            percentage = (curr_idx - self.warmup_cfg["num_iters"]) / (self.num_iters - self.warmup_cfg["num_iters"])
            self._update_lr(self.mode, 1.0, self.scheduler_cfg["cfg"].get("gamma", 0.1), percentage)

    def _update_lr(self, mode, factor, target_factor, percentage):
        if mode == "linear": factor = linear_anneal(target_factor, factor, percentage)
        elif mode == "cos": factor = cos_anneal(target_factor, factor, percentage)
        elif mode == "constant": factor = self.scheduler_cfg["cfg"]["coef"]
        
        for group, initial_lr in zip(self.optimizer.param_groups, self.learning_rates):
            group["lr"] = initial_lr * factor
''')
print("✓ Created utils/scheduler.py")

# --- 6. utils/optimizer.py ---
with open("utils/optimizer.py", "w", encoding="utf-8") as f:
    f.write('''import types
from torch.optim import SGD, Adam, AdamW

def get_optimizer(mode, params, initial_lr, optim_cfg):
    if mode == "adamw":
        return AdamW(params, lr=initial_lr, betas=optim_cfg.get("betas", (0.9, 0.999)), weight_decay=optim_cfg.get("weight_decay", 0))
    elif mode == "adam":
        return Adam(params, lr=initial_lr, betas=optim_cfg.get("betas", (0.9, 0.999)), weight_decay=optim_cfg.get("weight_decay", 0))
    return SGD(params, lr=initial_lr, momentum=optim_cfg["momentum"], weight_decay=optim_cfg["weight_decay"])

def group_params(model, group_mode, initial_lr, optim_cfg):
    if group_mode == "finetune":
        backbone_params, other_params = [], []
        for name, param in model.named_parameters():
            if "backbone" in name: backbone_params.append(param)
            else: other_params.append(param)
        return [{"params": backbone_params, "lr": initial_lr * optim_cfg.get("diff_factor", 0.1)}, {"params": other_params, "lr": initial_lr}]
    return model.parameters()

def construct_optimizer(model, initial_lr, mode, group_mode, cfg):
    params = group_params(model, group_mode, initial_lr, cfg)
    return get_optimizer(mode, params, initial_lr, cfg)
''')
print("✓ Created utils/optimizer.py")

# --- 7. utils/__init__.py (Crucial Fix) ---
with open("utils/__init__.py", "w", encoding="utf-8") as f:
    f.write('''from .image import read_color_array, read_gray_array
from .params import load_weight, save_weight
from . import array_ops
from . import tensor_ops
''')
print("✓ Created utils/__init__.py")

print("All utilities restored successfully!")