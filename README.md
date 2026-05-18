# CamoXpert: Camouflaged Object Detection

CamoXpert is a camouflaged object detection (COD) research codebase built on the ZoomNeXt architecture, with a newer CamoXpertV12 model that adds boundary refinement, adaptive scale weighting, and an enhanced loss.

## Highlights
- **Models**: CamoXpertV11 (baseline) and CamoXpertV12 (improved)
- **Evaluation**: `test_v11.py`, `test_v12.py` (S-measure + MAE on CAMO/CHAMELEON/COD10K/NC4K)
- **Training**: `main_for_image.py` (image COD), `main_for_video.py` (video COD)
- **Verification**: `verify_architecture.py` (sanity checks for V12)

## Environment Setup

**Python**: 3.9+ recommended

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2
pip install -r requirements.txt
```

## Data Preparation

### 1) Image COD test datasets (used by `test_v11.py` / `test_v12.py`)
Default expected path:
```
./data/COD-TestDataset/COD-TestDataset/
  CAMO/Image, CAMO/Mask
  CHAMELEON/Image, CHAMELEON/Mask
  COD10K/Image, COD10K/Mask
  NC4K/Imgs, NC4K/GT
```
If your data lives elsewhere, update the `DATA_ROOT` variable at the top of `test_v11.py` and `test_v12.py`.

### 2) Video COD datasets (used by `main_for_video.py`)
Create a `dataset.yaml` in the repo root (this file is gitignored). Use the structure below and adjust paths:
```yaml
moca_mask_tr:
  { root: "<VCOD_ROOT>/MoCA-Mask/MoCA_Video/TrainDataset_per_sq",
    image: { path: "*/Imgs", suffix: ".jpg" },
    mask:  { path: "*/GT",   suffix: ".png" },
    start_idx: 0, end_idx: 0 }

moca_mask_te:
  { root: "<VCOD_ROOT>/MoCA-Mask/MoCA_Video/TestDataset_per_sq",
    image: { path: "*/Imgs", suffix: ".jpg" },
    mask:  { path: "*/GT",   suffix: ".png" },
    start_idx: 0, end_idx: -2 }

cad:
  { root: "<VCOD_ROOT>/CamouflagedAnimalDataset",
    image: { path: "original_data/*/frames",     suffix: ".png" },
    mask:  { path: "converted_mask/*/groundtruth", suffix: ".png" },
    start_idx: 0, end_idx: 0 }

cod10k_tr:
  { root: "<ICOD_ROOT>/Train/COD10K-TR",
    image: { path: "Image", suffix: ".jpg" },
    mask:  { path: "Mask",  suffix: ".png" } }

camo_tr:
  { root: "<ICOD_ROOT>/Train/CAMO-TR",
    image: { path: "Image", suffix: ".jpg" },
    mask:  { path: "Mask",  suffix: ".png" } }

cod10k_te:
  { root: "<ICOD_ROOT>/Test/COD10K-TE",
    image: { path: "Image", suffix: ".jpg" },
    mask:  { path: "Mask",  suffix: ".png" } }

camo_te:
  { root: "<ICOD_ROOT>/Test/CAMO-TE",
    image: { path: "Image", suffix: ".jpg" },
    mask:  { path: "Mask",  suffix: ".png" } }

chameleon:
  { root: "<ICOD_ROOT>/Test/CHAMELEON",
    image: { path: "Image", suffix: ".jpg" },
    mask:  { path: "Mask",  suffix: ".png" } }

nc4k:
  { root: "<ICOD_ROOT>/Test/NC4K",
    image: { path: "Imgs", suffix: ".jpg" },
    mask:  { path: "GT",   suffix: ".png" } }
```

## Weights
- V11 default: `checkpoints_v11/zoomnext_epoch_120.pth`
- V12 default: `checkpoints_v12/camoxpert_v12_epoch_120.pth`

Update the `CHECKPOINT_PATH` in `test_v11.py` or `test_v12.py` if your weights live elsewhere.

## Quick Evaluation (Recommended for Paper Metrics)
These scripts print S-measure and MAE tables for CAMO, CHAMELEON, COD10K, and NC4K.

```bash
# Baseline (V11)
python test_v11.py

# Improved (V12)
python test_v12.py
```

## Training (Image COD)
`main_for_image.py` uses a config file that includes dataset root paths. Copy and edit `configs/camoxpert_v11.py` for your local paths, then run:

```bash
# CamoXpertV12 (base)
python main_for_image.py \
  --config configs/camoxpert_v11.py \
  --model-name CamoXpertV12_Base \
  --save-dir checkpoints_v12

# Resume from an existing checkpoint
python main_for_image.py \
  --config configs/camoxpert_v11.py \
  --model-name CamoXpertV12_Base \
  --load-from checkpoints_v11/zoomnext_epoch_120.pth \
  --save-dir checkpoints_v12
```

## Training / Evaluation (Video COD)
```bash
# Pretrain (image) then finetune (video)
python main_for_image.py --config configs/icod_pretrain.py --pretrained --model-name PvtV2B5_ZoomNeXt
python main_for_video.py --config configs/vcod_finetune.py --model-name videoPvtV2B5_ZoomNeXt --load-from <PRETRAINED_WEIGHT>

# Evaluate video model
python main_for_video.py --config configs/vcod_finetune.py --model-name videoPvtV2B5_ZoomNeXt --evaluate --load-from <FINETUNED_WEIGHT>
```

## Architecture Verification (V12)
```bash
python verify_architecture.py
```

## Notes for Reproducibility
- `test_v11.py` and `test_v12.py` use multi-scale test-time augmentation (3 scales × 2 views).
- Metrics reported: **S-measure (↑)** and **MAE (↓)**.
- Keep dataset splits consistent between runs to ensure fair comparisons.

## Reference
This repository builds on ZoomNeXt (TPAMI 2024). Please cite the original paper when appropriate.
