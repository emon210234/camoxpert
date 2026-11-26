"""Datasets for the standard COD protocol.

Expected layout (the SINet-V2 release; checked on the lab VM on 2026-11-26):
    <root>/TrainDataset/Imgs/*.jpg   <root>/TrainDataset/GT/*.png     4,040 = 3,040 COD10K + 1,000 CAMO
    <root>/TestDataset/<NAME>/Imgs/*.jpg, <root>/TestDataset/<NAME>/GT/*.png
        NAME: CAMO (250), CHAMELEON (76), COD10K (2,026), NC4K (4,121)
Junk from zip files (__MACOSX/, "._*" files) is ignored.

Images are returned as float tensors in [0, 1]; the model does the ImageNet mean/std
normalisation itself (once). The legacy loader also applied A.Normalize, so the input was
normalised twice (CHANGES.md, finding C10).
"""

import hashlib
import os

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
TEST_SETS = ("CAMO", "CHAMELEON", "COD10K", "NC4K")


def _is_junk(name):
    return name.startswith("._") or name.startswith(".") or name == "__MACOSX"


def list_pairs(img_dir, gt_dir):
    """Sorted list of (image_path, mask_path). Every image must have a mask with the same stem
    (.png preferred, .jpg accepted); a missing mask raises an error instead of silently using an
    all-zero mask as the legacy loader did."""
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(img_dir)
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(gt_dir)
    gts = {}
    for f in os.listdir(gt_dir):
        stem, ext = os.path.splitext(f)
        if _is_junk(f) or ext.lower() not in IMG_EXTS:
            continue
        if stem not in gts or ext.lower() == ".png":
            gts[stem] = os.path.join(gt_dir, f)
    pairs, missing = [], []
    for f in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(f)
        if _is_junk(f) or ext.lower() not in IMG_EXTS:
            continue
        if stem in gts:
            pairs.append((os.path.join(img_dir, f), gts[stem]))
        else:
            missing.append(f)
    if missing:
        raise FileNotFoundError(f"{len(missing)} images in {img_dir} have no mask, e.g. {missing[:3]}")
    if not pairs:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return pairs


def train_pairs(root):
    return list_pairs(os.path.join(root, "TrainDataset", "Imgs"), os.path.join(root, "TrainDataset", "GT"))


def test_pairs(root, name):
    base = os.path.join(root, "TestDataset", name)
    return list_pairs(os.path.join(base, "Imgs"), os.path.join(base, "GT"))


def pairs_fingerprint(pairs):
    """SHA-256 of the sorted image file names: proves later which exact file list was used."""
    names = "\n".join(os.path.basename(p[0]) for p in pairs)
    return hashlib.sha256(names.encode()).hexdigest()


def read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_gray(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise IOError(f"Cannot read mask {path}")
    return g


def to_tensor_image(img_uint8):
    return torch.from_numpy(np.ascontiguousarray(img_uint8)).permute(2, 0, 1).float().div(255.0)


class TrainDataset(Dataset):
    """Augmentation (same operations and probabilities as the legacy CamoXpert loader):
    resize to S x S (bilinear for the image, nearest for the mask), horizontal flip p=0.5,
    vertical flip p=0.5, rotation in [-15, 15] degrees p=0.5 (border: reflect-101, applied to image
    and mask alike), ColorJitter (brightness/contrast/saturation 0.2, hue 0.2) p=0.3, image only.
    Mask target = gray value / 255 (soft target in [0, 1], as in the legacy code)."""

    def __init__(self, pairs, size):
        self.pairs = pairs
        self.size = size
        self.transform = A.Compose(
            [
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(p=0.3),
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]
        image, mask = read_rgb(img_path), read_gray(gt_path)
        if image.shape[:2] != mask.shape[:2]:  # a few public masks differ in size: follow the mask
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        out = self.transform(image=image, mask=mask)
        mask_t = torch.from_numpy(np.ascontiguousarray(out["mask"])).float().div(255.0).unsqueeze(0)
        return {"image": to_tensor_image(out["image"]), "mask": mask_t}


class TestDataset(Dataset):
    """Image resized to S x S (bilinear), no augmentation. The GT is read at its original size by
    the evaluator; predictions are upsampled to it."""

    def __init__(self, pairs, size):
        self.pairs = pairs
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]
        image = read_rgb(img_path)
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return {
            "image": to_tensor_image(image),
            "gt_path": gt_path,
            "name": os.path.splitext(os.path.basename(img_path))[0],
            "hw": torch.tensor([h, w]),
        }
