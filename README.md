# CamoXpert: Camouflaged Object Detection (ICOD + VCOD)

CamoXpert is a research codebase for camouflaged object detection (COD) with image and video pipelines. The current code reflects the post–Jan 13, 2026 updates and includes the CamoXpertV11 baseline and the improved CamoXpertV12 variants.

## Highlights
- **CamoXpertV11**: PVTv2-B5 + Mamba-style selective scan + Spectral MoE (baseline)
- **CamoXpertV12_Base**: V11 + Adaptive Scale Weighting + Boundary Refinement + Enhanced Loss
- **CamoXpertV12_Progressive**: V12_Base + Progressive Refinement (higher accuracy, slower)
- **Metrics**: S-measure and MAE on CAMO, CHAMELEON, COD10K, NC4K
- **MSTTA**: Multi-scale test-time augmentation for final benchmarks

## Requirements
- **Python**: 3.9–3.11
- **GPU**: CUDA-capable GPU recommended (CPU possible but slow)
- **PyTorch**: Install a CUDA-matched build from https://pytorch.org/get-started/locally/

## Installation
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install PyTorch (select the CUDA build that matches your system)
# Example (CUDA 11.8):
# pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

## Data Setup
### Image COD (ICOD)
Update dataset paths in your config (e.g., `configs/camoxpert_v11.py`, or copy it to `configs/camoxpert_v12.py`).

Key fields:
- `root_path`
- `train_image_path`
- `train_mask_path`
- `test_image_path`
- `test_mask_path`

Ensure the folder names match your dataset layout (e.g., `Image/Mask` vs `Imgs/GT`).

### Video COD (VCOD)
Create a `dataset.yaml` in the repo root (or pass `--data-cfg`) with dataset definitions. Example structure:

```yaml
moca_mask_tr:
  { root: "/path/to/MoCA-Mask/MoCA_Video/TrainDataset_per_sq", image: { path: "*/Imgs", suffix: ".jpg" }, mask: { path: "*/GT", suffix: ".png" }, start_idx: 0, end_idx: 0 }

moca_mask_te:
  { root: "/path/to/MoCA-Mask/MoCA_Video/TestDataset_per_sq", image: { path: "*/Imgs", suffix: ".jpg" }, mask: { path: "*/GT", suffix: ".png" }, start_idx: 0, end_idx: -2 }

cad:
  { root: "/path/to/CamouflagedAnimalDataset", image: { path: "original_data/*/frames", suffix: ".png" }, mask: { path: "converted_mask/*/groundtruth", suffix: ".png" }, start_idx: 0, end_idx: 0 }

cod10k_tr:
  { root: "/path/to/COD10K-v3/Train/COD10K-TR", image: { path: "Image", suffix: ".jpg" }, mask: { path: "Mask", suffix: ".png" } }
```

## Training
### Image COD (ICOD)
```bash
# V11 baseline
python main_for_image.py --config configs/camoxpert_v11.py --model-name CamoXpertV11 --save-dir checkpoints_v11

# V12 base
python main_for_image.py --config configs/camoxpert_v12.py --model-name CamoXpertV12_Base --save-dir checkpoints_v12

# V12 progressive (slower, higher accuracy)
python main_for_image.py --config configs/camoxpert_v12.py --model-name CamoXpertV12_Progressive --save-dir checkpoints_v12

# Resume or transfer learning
python main_for_image.py --config configs/camoxpert_v12.py --model-name CamoXpertV12_Base --load-from checkpoints_v11/zoomnext_epoch_120.pth --save-dir checkpoints_v12
```

### Video COD (VCOD)
```bash
# Pretrain backbone on ICOD
python main_for_image.py --config configs/icod_pretrain.py --model-name PvtV2B5_ZoomNeXt --pretrained --save-dir checkpoints_icod

# Finetune on VCOD
python main_for_video.py --config configs/vcod_finetune.py --data-cfg dataset.yaml --model-name videoPvtV2B5_ZoomNeXt --load-from checkpoints_icod/<PRETRAIN_CKPT>.pth
```

## Evaluation (Paper Metrics)
Update `DATA_ROOT` and `CHECKPOINT_PATH` inside the test scripts, then run:

```bash
# V11 baseline metrics
python test_v11.py

# V12 metrics
python test_v12.py
```

Both scripts report **S-measure** and **MAE** for **CAMO**, **CHAMELEON**, **COD10K**, and **NC4K** using MSTTA.

## Architecture Verification
```bash
python verify_architecture.py
```
This validates that V12 instantiates correctly and runs a forward pass.

## Reproducibility Tips
- Record dataset versions, config files, and checkpoint paths.
- Keep the exact PyTorch/CUDA versions used for final results.
- Report MSTTA scales and evaluation settings in the paper.

## Project Layout
```
configs/               # Training configs
methods/zoomnext/      # Model implementations (V11/V12)
main_for_image.py      # ICOD training script
main_for_video.py      # VCOD training/eval script
test_v11.py            # V11 benchmarking (S-measure/MAE)
test_v12.py            # V12 benchmarking (S-measure/MAE)
verify_architecture.py # V12 architecture sanity check
```

## License
No explicit license is provided. Treat the code as proprietary until you obtain written permission from the repository owner.
