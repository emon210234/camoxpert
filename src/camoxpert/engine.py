"""Training helpers: seeding, optimizer, learning-rate schedule, checkpoints, run metadata."""

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys

import numpy as np
import torch


# ----------------------------------------------------------------------------------- seeding
def seed_everything(seed: int):
    """Seeds Python, NumPy and PyTorch (CPU + all GPUs). cudnn.benchmark stays on for speed, so
    two runs with the same seed are close but not bit-identical; the paper says so."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """Each DataLoader worker gets its own, reproducible seed for `random` and NumPy
    (albumentations draws its random numbers from them). torch already gives every worker
    base_seed + worker_id; we reuse that number."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------------- optimizer
def build_optimizer(model, opt_cfg: dict):
    """AdamW with two parameter groups:
    backbone (pretrained PVTv2-B5): lr * backbone_lr_factor;  everything else: lr.
    Weight decay is applied to all parameters (as in the legacy code)."""
    prefix = model.backbone_prefix
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if name.startswith(prefix) else head).append(p)
    lr = opt_cfg["lr"]
    groups = [
        {"params": backbone, "lr": lr * opt_cfg["backbone_lr_factor"], "name": "backbone"},
        {"params": head, "lr": lr, "name": "decoder"},
    ]
    for g in groups:
        g["initial_lr"] = g["lr"]
    if opt_cfg["name"] != "adamw":
        raise ValueError("only adamw is configured for the paper runs")
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(opt_cfg["betas"]), weight_decay=opt_cfg["weight_decay"])


def lr_factor(it: int, total: int, warmup_iters: int, warmup_start: float, min_factor: float) -> float:
    """Multiplier of each group's initial lr at iteration `it` (0-based).

    warmup (it < W):   f = a + (1 - a) * it / W                   linear from a to 1
    afterwards:        f = m + (1 - m) * (1 + cos(pi * q)) / 2,   q = (it - W) / (T - W)
                       cosine from 1 down to m at the last iteration
    (a = warmup_start, m = min_factor, W = warmup_iters, T = total iterations.)
    """
    if it < warmup_iters:
        return warmup_start + (1.0 - warmup_start) * it / warmup_iters
    q = (it - warmup_iters) / max(1, total - warmup_iters)
    q = min(max(q, 0.0), 1.0)
    return min_factor + (1.0 - min_factor) * (1 + math.cos(math.pi * q)) / 2


def set_lr(optimizer, factor):
    for g in optimizer.param_groups:
        g["lr"] = g["initial_lr"] * factor


# ------------------------------------------------------------------------------- checkpoints
def atomic_save(obj, path):
    """Write to path.tmp then rename: a crash or power cut mid-save never corrupts the old file."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------ run metadata
def git_info(repo_dir):
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def env_info():
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import albumentations
        import cv2
        import py_sod_metrics  # noqa: F401
        import timm

        info.update(albumentations=albumentations.__version__, opencv=cv2.__version__, timm=timm.__version__)
    except Exception:
        pass
    return info


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


class JsonlLogger:
    """One JSON object per line; easy to load later with pandas or a 5-line script."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def log(self, **record):
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
