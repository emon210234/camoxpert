# CamoXpert

CamoXpert is a camouflaged object detection project built on ZoomNeXt, with the current production architecture centered on **CamoXpertV12**.

---

## 1. What this repository contains

- Image COD training entrypoint: `main_for_image.py`
- Video COD training/evaluation entrypoint: `main_for_video.py`
- Evaluation scripts for paper metrics:
  - `test_v11.py`
  - `test_v12.py`
- Architecture sanity script: `verify_architecture.py`
- Model implementations:
  - `methods/zoomnext/zoomnext.py` (V11 and prior models)
  - `methods/zoomnext/camoxpert_v12.py` (V12 family)

---

## 2. Fast restart (from a clean machine)

Set your local repository root:
`export CAMOXPERT_ROOT=/path/to/camoxpert`

If you want this to persist across sessions, add that line to your shell profile (for example `~/.bashrc` or `~/.zshrc`).

### 2.1 Create environment

```bash
cd "$CAMOXPERT_ROOT"
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2.2 Install dependencies

Install a PyTorch build that matches your CUDA/CPU environment first:

```bash
# Example (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install project dependencies
pip install -r requirements.txt
```

> If you use a different CUDA version (or CPU-only), select the correct command at https://pytorch.org/get-started/locally/ and then install `requirements.txt`.

---

## 3. Data configuration

### 3.1 Image COD training config (used by `main_for_image.py`)

Edit:
`$CAMOXPERT_ROOT/configs/camoxpert_v11.py`

Set:
- `train.data.root_path`
- `train.data.train_image_path`
- `train.data.train_mask_path`
- `train.data.test_image_path`
- `train.data.test_mask_path`

This same config file is currently used by both V11 and V12 image-training commands in this repository.

### 3.2 Video/benchmark YAML (used by `main_for_video.py`)

Create:
`$CAMOXPERT_ROOT/dataset.yaml`

Minimal template:

```yaml
moca_mask_tr:
  { root: "YOUR-VCOD-ROOT/MoCA-Mask/MoCA_Video/TrainDataset_per_sq", image: { path: "*/Imgs", suffix: ".jpg" }, mask: { path: "*/GT", suffix: ".png" }, start_idx: 0, end_idx: 0 }
moca_mask_te:
  { root: "YOUR-VCOD-ROOT/MoCA-Mask/MoCA_Video/TestDataset_per_sq", image: { path: "*/Imgs", suffix: ".jpg" }, mask: { path: "*/GT", suffix: ".png" }, start_idx: 0, end_idx: -2 }
cad:
  { root: "YOUR-VCOD-ROOT/CamouflagedAnimalDataset", image: { path: "original_data/*/frames", suffix: ".png" }, mask: { path: "converted_mask/*/groundtruth", suffix: ".png" }, start_idx: 0, end_idx: 0 }
cod10k_tr:
  { root: "YOUR-ICOD-ROOT/Train/COD10K-TR", image: { path: "Image", suffix: ".jpg" }, mask: { path: "Mask", suffix: ".png" } }
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

## 4. Sanity check before long runs

```bash
cd "$CAMOXPERT_ROOT"
python verify_architecture.py
```

This validates model construction and forward pass behavior for the V12 stack.

---

## 5. Train models

### 5.1 Train V11 baseline (image COD)

```bash
cd "$CAMOXPERT_ROOT"
python main_for_image.py \
  --config configs/camoxpert_v11.py \
  --model-name CamoXpertV11 \
  --save-dir checkpoints_v11
```

Note: `main_for_image.py` currently constructs models with `pretrained=True` internally.

### 5.2 Train V12 (recommended for paper)

```bash
cd "$CAMOXPERT_ROOT"
python main_for_image.py \
  --config configs/camoxpert_v11.py \
  --model-name CamoXpertV12_Base \
  --save-dir checkpoints_v12
```

Available V12 names:
- `CamoXpertV12`
- `CamoXpertV12_Base`
- `CamoXpertV12_Progressive`

---

## 6. Evaluate and extract paper metrics

Set `CHECKPOINT_PATH` and `DATA_ROOT` in:
- `$CAMOXPERT_ROOT/test_v11.py`
- `$CAMOXPERT_ROOT/test_v12.py`

Then run:

```bash
cd "$CAMOXPERT_ROOT"
python test_v11.py | tee results_v11.txt
python test_v12.py | tee results_v12.txt
```

Primary outputs:
- `S-measure (↑)`
- `MAE (↓)`

Datasets covered by default:
- CAMO
- CHAMELEON
- COD10K
- NC4K

For publication tables, keep all raw run logs and report mean/std from repeated runs.

---

## 7. Video COD workflow (optional)

Train/finetune:

```bash
cd "$CAMOXPERT_ROOT"
python main_for_video.py \
  --config configs/vcod_finetune.py \
  --data-cfg dataset.yaml \
  --model-name videoPvtV2B5_ZoomNeXt \
  --pretrained
```

Evaluate:

```bash
cd "$CAMOXPERT_ROOT"
python main_for_video.py \
  --config configs/vcod_finetune.py \
  --data-cfg dataset.yaml \
  --model-name videoPvtV2B5_ZoomNeXt \
  --load-from /path/to/checkpoint.pth \
  --evaluate
```

---

## 8. Reproducibility checklist

For every reported result, record:
- Commit hash
- Model name
- Config file path
- Checkpoint path
- GPU + CUDA versions
- Full command used
- Raw metric output file

Run final evaluations multiple times and report aggregate statistics.

---

## 9. Troubleshooting

- **`ModuleNotFoundError: torch`**  
  Install PyTorch first, then reinstall requirements.
- **`prettytable` import error in test scripts**  
  Re-run `pip install -r requirements.txt` in the active environment.
- **OOM during V12 training**  
  Lower batch size, reduce input shape, keep AMP enabled.
- **Dataset not found**  
  Verify folder names (`Image` vs `Imgs`, `Mask` vs `GT`) and absolute roots.
