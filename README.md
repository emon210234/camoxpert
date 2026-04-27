# CamoXpert

Modern camouflaged object detection built on ZoomNeXt, with custom CamoXpert variants (including V11 and V12).

This repository includes:
- training entrypoints for image/video COD
- evaluation scripts for V11/V12
- architecture validation utilities

---

## 1) Environment Setup (Fresh Machine)

### Recommended
- OS: Linux/WSL (CUDA-capable GPU recommended)
- Python: 3.10
- CUDA: compatible with your installed PyTorch build

### Create environment
```bash
cd /home/runner/work/camoxpert/camoxpert
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### Install dependencies
```bash
pip install torch==2.1.2 torchvision==0.16.2
pip install -r requirements.txt
```

---

## 2) Dataset Preparation

You need two styles of data config:

### A) Image COD training (`main_for_image.py`)
`configs/camoxpert_v11.py` contains `root_path`, `train_image_path`, `train_mask_path`, etc.
Update these to your local paths before training.

### B) Video + general eval pipeline (`main_for_video.py`)
Create `/home/runner/work/camoxpert/camoxpert/dataset.yaml`:

```yaml
# VCOD
moca_mask_tr:
  { root: "YOUR-VCOD-ROOT/MoCA-Mask/MoCA_Video/TrainDataset_per_sq", image: { path: "*/Imgs", suffix: ".jpg" }, mask: { path: "*/GT", suffix: ".png" }, start_idx: 0, end_idx: 0 }
moca_mask_te:
  { root: "YOUR-VCOD-ROOT/MoCA-Mask/MoCA_Video/TestDataset_per_sq", image: { path: "*/Imgs", suffix: ".jpg" }, mask: { path: "*/GT", suffix: ".png" }, start_idx: 0, end_idx: -2 }
cad:
  { root: "YOUR-VCOD-ROOT/CamouflagedAnimalDataset", image: { path: "original_data/*/frames", suffix: ".png" }, mask: { path: "converted_mask/*/groundtruth", suffix: ".png" }, start_idx: 0, end_idx: 0 }

# ICOD
cod10k_tr:
  { root: "YOUR-ICOD-ROOT/Train/COD10K-TR", image: { path: "Image", suffix: ".jpg" }, mask: { path: "Mask", suffix: ".png" } }
camo_tr:
  { root: "YOUR-ICOD-ROOT/Train/CAMO-TR", image: { path: "Image", suffix: ".jpg" }, mask: { path: "Mask", suffix: ".png" } }
cod10k_te:
  { root: "YOUR-ICOD-ROOT/Test/COD10K-TE", image: { path: "Image", suffix: ".jpg" }, mask: { path: "Mask", suffix: ".png" } }
camo_te:
  { root: "YOUR-ICOD-ROOT/Test/CAMO-TE", image: { path: "Image", suffix: ".jpg" }, mask: { path: "Mask", suffix: ".png" } }
chameleon:
  { root: "YOUR-ICOD-ROOT/Test/CHAMELEON", image: { path: "Image", suffix: ".jpg" }, mask: { path: "Mask", suffix: ".png" } }
nc4k:
  { root: "YOUR-ICOD-ROOT/Test/NC4K", image: { path: "Imgs", suffix: ".jpg" }, mask: { path: "GT", suffix: ".png" } }
```

---

## 3) Quick Sanity Checks

### Check architecture (no dataset required)
```bash
python verify_architecture.py
```

### Verify model imports
Available CamoXpert classes are exported from:
`/home/runner/work/camoxpert/camoxpert/methods/__init__.py`

---

## 4) Training

## A) Image COD (CamoXpert V11 baseline)

```bash
python main_for_image.py \
  --config configs/camoxpert_v11.py \
  --model-name CamoXpertV11 \
  --save-dir checkpoints_v11
```

Resume from a checkpoint:
```bash
python main_for_image.py \
  --config configs/camoxpert_v11.py \
  --model-name CamoXpertV11 \
  --load-from checkpoints_v11/zoomnext_epoch_120.pth \
  --save-dir checkpoints_v11
```

## B) Image COD (CamoXpert V12)

Use one of:
- `CamoXpertV12`
- `CamoXpertV12_Base`
- `CamoXpertV12_Progressive`

```bash
python main_for_image.py \
  --config configs/camoxpert_v11.py \
  --model-name CamoXpertV12_Base \
  --save-dir checkpoints_v12
```

> Note: there is currently no committed `configs/camoxpert_v12.py`; reusing `configs/camoxpert_v11.py` is the fastest path, then tune hyperparameters as needed.

## C) Video COD (MoCA/CAD finetuning)

```bash
python main_for_video.py \
  --config configs/vcod_finetune.py \
  --data-cfg dataset.yaml \
  --model-name videoPvtV2B5_ZoomNeXt \
  --pretrained
```

Evaluate mode:
```bash
python main_for_video.py \
  --config configs/vcod_finetune.py \
  --data-cfg dataset.yaml \
  --model-name videoPvtV2B5_ZoomNeXt \
  --load-from <CHECKPOINT_PATH> \
  --evaluate
```

---

## 5) Evaluation & Metric Extraction (for paper tables)

### V11 evaluation (Sm + MAE on CAMO/CHAMELEON/COD10K/NC4K)
```bash
python test_v11.py | tee results_v11.txt
```

### V12 evaluation (Sm + MAE on CAMO/CHAMELEON/COD10K/NC4K)
```bash
python test_v12.py | tee results_v12.txt
```

Both scripts print a final table:
- `S-measure (↑)`
- `MAE (↓)`

Use these values directly for IEEE result tables.

---

## 6) Reproducibility Checklist (Publication-Oriented)

- Fix random seed where possible.
- Keep dataset versions and directory layout fixed.
- Record:
  - git commit hash
  - config file used
  - checkpoint path
  - GPU model / driver / CUDA
  - exact metric table output (`results_*.txt`)
- Run each final evaluation at least 3 times and report mean ± std.

---

## 7) Common Issues

### Out-of-memory
- lower `batch_size` in config
- reduce input size (`shape`)
- use fewer workers

### Missing checkpoints in test scripts
Edit:
- `CHECKPOINT_PATH`
- `DATA_ROOT`
inside `test_v11.py` / `test_v12.py`.

### Dataset not found / empty
Check exact folder names:
- `Image` vs `Imgs`
- `Mask` vs `GT`

---

## 8) Useful Files

- `/home/runner/work/camoxpert/camoxpert/main_for_image.py` — image COD training
- `/home/runner/work/camoxpert/camoxpert/main_for_video.py` — video COD training/eval
- `/home/runner/work/camoxpert/camoxpert/test_v11.py` — V11 benchmark script
- `/home/runner/work/camoxpert/camoxpert/test_v12.py` — V12 benchmark script
- `/home/runner/work/camoxpert/camoxpert/verify_architecture.py` — architecture sanity checks
- `/home/runner/work/camoxpert/camoxpert/configs/` — experiment configs
