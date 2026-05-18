# CamoXpert: Camouflaged Object Detection (Image + Video)

CamoXpert is a research codebase for camouflaged object detection (COD). It includes the **CamoXpertV11** baseline and **CamoXpertV12** improvements (adaptive scale weighting, boundary refinement, enhanced loss, optional progressive refinement). The project supports:

- **Image COD (ICOD)** training and evaluation
- **Video COD (VCOD)** finetuning and evaluation
- Multi-scale test-time augmentation (MSTTA) benchmarks

> This repository reflects the post–Jan 13, 2026 codebase updates.

## Key Models

- **CamoXpertV11**: PVTv2-B5 + Mamba + MoE (baseline)
- **CamoXpertV12_Base**: V11 + Adaptive Scale Weighting + Boundary Refinement + Enhanced Loss
- **CamoXpertV12_Progressive**: V12_Base + Progressive Refinement (slower, higher accuracy)

## Environment Setup

**Recommended:** Python 3.9–3.11 with a CUDA-capable GPU.

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) Install PyTorch (choose your CUDA build)
# Use the PyTorch selector to match your CUDA version (wheel tags like cu118/cu121):
# https://pytorch.org/get-started/locally/
# Example (CUDA 12.1):
# pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
# Example (CPU-only):
# pip install torch==2.1.2 torchvision==0.16.2

# 3) Install remaining dependencies
pip install -r requirements.txt
```

## Data Setup

### Image COD (ICOD)
Training with `main_for_image.py` uses paths defined in your config file.
Update these fields in **configs/camoxpert_v11.py** (or your custom config):

```python
train = dict(
    data=dict(
        root_path="/path/to/COD10K-v3",
        train_image_path="Train/Image",
        train_mask_path="Train/Mask",   # or Train/GT_Object depending on your dataset
        test_image_path="Test/Image",
        test_mask_path="Test/GT_Object",
    )
)
```

### Video COD (VCOD)
`main_for_video.py` expects a dataset YAML. Create `dataset.yaml` in the repo root (or pass `--data-cfg`).

Minimal template:
```yaml
# VCOD
moca_mask_tr:
  { root: "/path/to/MoCA-Mask/MoCA_Video/TrainDataset_per_sq", image: { path: "*/Imgs", suffix: ".jpg" }, mask: { path: "*/GT", suffix: ".png" }, start_idx: 0, end_idx: 0 }
moca_mask_te:
  { root: "/path/to/MoCA-Mask/MoCA_Video/TestDataset_per_sq",  image: { path: "*/Imgs", suffix: ".jpg" }, mask: { path: "*/GT", suffix: ".png" }, start_idx: 0, end_idx: -2 }
cad:
  { root: "/path/to/CamouflagedAnimalDataset", image: { path: "original_data/*/frames", suffix: ".png" }, mask: { path: "converted_mask/*/groundtruth", suffix: ".png" }, start_idx: 0, end_idx: 0 }

# ICOD
cod10k_tr:
  { root: "/path/to/COD10K-v3/Train/COD10K-TR", image: { path: "Image", suffix: ".jpg" }, mask: { path: "Mask", suffix: ".png" } }
camo_tr:
  { root: "/path/to/CAMO/Train/CAMO-TR", image: { path: "Image", suffix: ".jpg" }, mask: { path: "Mask", suffix: ".png" } }
cod10k_te:
  { root: "/path/to/COD10K-v3/Test/COD10K-TE", image: { path: "Image", suffix: ".jpg" }, mask: { path: "Mask", suffix: ".png" } }
camo_te:
  { root: "/path/to/CAMO/Test/CAMO-TE", image: { path: "Image", suffix: ".jpg" }, mask: { path: "Mask", suffix: ".png" } }
chameleon:
  { root: "/path/to/CHAMELEON", image: { path: "Image", suffix: ".jpg" }, mask: { path: "Mask", suffix: ".png" } }
nc4k:
  { root: "/path/to/NC4K", image: { path: "Imgs", suffix: ".jpg" }, mask: { path: "GT", suffix: ".png" } }
```

## Training

### Image COD (ICOD)
```bash
# V11 baseline
python main_for_image.py --config configs/camoxpert_v11.py --model-name CamoXpertV11 --save-dir checkpoints_v11

# V12 base (copy config for clarity; same data fields)
cp configs/camoxpert_v11.py configs/camoxpert_v12.py
# Update root_path/train_* paths and (optionally) batch size for memory
python main_for_image.py --config configs/camoxpert_v12.py --model-name CamoXpertV12_Base --save-dir checkpoints_v12

# V12 progressive refinement
python main_for_image.py --config configs/camoxpert_v12.py --model-name CamoXpertV12_Progressive --save-dir checkpoints_v12
```

### Video COD (VCOD)
```bash
# Pretrain (ICOD backbone init)
python main_for_image.py --config configs/icod_pretrain.py --model-name PvtV2B5_ZoomNeXt --pretrained --save-dir checkpoints_icod

# Finetune (VCOD) from ICOD checkpoint
python main_for_video.py --config configs/vcod_finetune.py --data-cfg dataset.yaml --model-name videoPvtV2B5_ZoomNeXt --load-from checkpoints_icod/<PRETRAIN_CKPT>.pth
```

> If you hit OOM, reduce batch size in the config or add `--use-checkpoint` to enable gradient checkpointing.

## Evaluation (Paper Metrics)

### V11 (baseline)
```bash
# Update CHECKPOINT_PATH and DATA_ROOT in test_v11.py before running
python test_v11.py
```

### V12 (improved)
```bash
# Update CHECKPOINT_PATH and DATA_ROOT in test_v12.py before running
python test_v12.py
```

Both scripts compute **S-measure** and **MAE** for **CAMO**, **CHAMELEON**, **COD10K**, and **NC4K** with MSTTA.

## Architecture Verification

```bash
python verify_architecture.py
```

## Project Layout (Key Files)

```
configs/               # Training configs
methods/zoomnext/      # Model implementations (V11/V12)
main_for_image.py      # ICOD training script
main_for_video.py      # VCOD training/eval script (uses dataset.yaml)
test_v11.py            # V11 benchmarking script
test_v12.py            # V12 benchmarking script
verify_architecture.py # Sanity check for V12 model
```

## Notes

- `test_v11.py` uses `DEVICE = 'cuda'` by default. Switch to `'cpu'` if needed.
- Verify dataset folder names (`Image/Mask` vs `Imgs/GT`) and update configs accordingly.
- For reproducibility, keep a record of checkpoint paths and data versions used in your paper.
- `--pretrained` loads backbone weights; use `--load-from` to resume a full CamoXpert checkpoint.

## License

No explicit license is provided. Contact the repository owner for usage permissions.
