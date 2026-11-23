# CamoXpert: Camouflaged Object Detection
 
CamoXpert is a research codebase for image-based camouflaged object detection (COD). It includes the **CamoXpertV11** baseline and **CamoXpertV12** improvements (adaptive scale weighting, boundary refinement, enhanced loss, optional progressive refinement). The project supports:
 
- **Image COD** training and evaluation
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
# Example (tested CUDA 11.8 build):
# pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
# Example (CPU-only, tested):
# pip install torch==2.1.2 torchvision==0.16.2
#
# Note: the versions above are tested with this repo; use pytorch.org for newer stable releases if desired.
 
# 3) Install remaining dependencies
pip install -r requirements.txt
```
 
## Data Setup
 
Training with `main_for_image.py` uses paths defined in your config file.
Update these fields in **configs/camoxpert_v11.py** (or your custom config):
 
```python
train = dict(
    data=dict(
        root_path="/path/to/COD10K-v3",
        train_image_path="Train/Image",
        train_mask_path="Train/GT_Object",
        test_image_path="Test/Image",
        test_mask_path="Test/GT_Object",
    )
)
```
The training script does not auto-detect folder conventions—make sure these paths match your dataset layout.
If you are using COD10K-v3, masks are typically under `Train/GT_Object`; older releases may use `Train/Mask`.
 
## Training
 
```bash
# V11 baseline
python main_for_image.py --config configs/camoxpert_v11.py --model-name CamoXpertV11 --save-dir checkpoints_v11
 
# V12 base (reuse V11 config or create a separate one)
python main_for_image.py --config configs/camoxpert_v11.py --model-name CamoXpertV12_Base --save-dir checkpoints_v12
# Optional: copy for experiment tracking
cp configs/camoxpert_v11.py configs/camoxpert_v12.py
# Update root_path/train_* paths and (optionally) batch size for memory
python main_for_image.py --config configs/camoxpert_v12.py --model-name CamoXpertV12_Base --save-dir checkpoints_v12
 
# V12 progressive refinement
python main_for_image.py --config configs/camoxpert_v12.py --model-name CamoXpertV12_Progressive --save-dir checkpoints_v12
```
V12 uses the same config schema as V11; set dataset paths in whichever config you use and only change hyperparameters if needed.
 
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
 
## Results
 
Quantitative comparison with state-of-the-art COD methods on four benchmarks.
 
**Metrics:** S<sub>α</sub> = S-measure, F<sup>ω</sup><sub>β</sub> = weighted F-measure, E<sub>φ</sub> = E-measure, M = mean absolute error (MAE). ↑ higher is better, ↓ lower is better.
The best result in each column is in **bold** and the second best is <ins>underlined</ins>. "–" means the result is not reported.
 
### COD10K and CAMO
 
| Method | Year | COD10K<br>S<sub>α</sub> ↑ | COD10K<br>F<sup>ω</sup><sub>β</sub> ↑ | COD10K<br>E<sub>φ</sub> ↑ | COD10K<br>M ↓ | CAMO<br>S<sub>α</sub> ↑ | CAMO<br>F<sup>ω</sup><sub>β</sub> ↑ | CAMO<br>E<sub>φ</sub> ↑ | CAMO<br>M ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ANet-SRM | 2019 | – | – | – | – | 0.682 | 0.484 | 0.722 | 0.126 |
| SINet | 2021 | 0.776 | 0.631 | 0.874 | 0.043 | 0.745 | 0.644 | 0.829 | 0.092 |
| SLSR | 2021 | 0.804 | 0.673 | 0.892 | 0.037 | 0.787 | 0.696 | 0.854 | 0.080 |
| MGL-R | 2021 | 0.814 | 0.666 | 0.890 | 0.035 | 0.775 | 0.673 | 0.842 | 0.088 |
| PFNet | 2021 | 0.800 | 0.660 | 0.890 | 0.040 | 0.782 | 0.695 | 0.855 | 0.085 |
| UJSC | 2021 | 0.809 | 0.684 | 0.891 | 0.035 | 0.800 | 0.728 | 0.873 | 0.073 |
| MirrorNet | 2021 | – | – | – | – | 0.785 | 0.719 | 0.850 | 0.077 |
| C²F-Net | 2021 | 0.813 | 0.686 | 0.900 | 0.036 | 0.796 | 0.719 | 0.864 | 0.080 |
| UGTR | 2021 | 0.817 | 0.666 | 0.890 | 0.036 | 0.784 | 0.684 | 0.851 | 0.086 |
| ZoomNet | 2022 | 0.838 | 0.729 | 0.911 | 0.029 | 0.820 | 0.753 | 0.892 | 0.066 |
| HitNet | 2023 | 0.868 | <ins>0.784</ins> | <ins>0.925</ins> | <ins>0.024</ins> | 0.844 | 0.801 | **0.920** | 0.056 |
| ZoomNeXt | 2024 | **0.878** | **0.845** | **0.938** | **0.023** | 0.842 | <ins>0.810</ins> | 0.915 | 0.060 |
| CamoXpert v11 | – | 0.862 | 0.692 | 0.836 | 0.033 | 0.876 | 0.775 | 0.906 | 0.057 |
| **CamoXpert v12** | – | 0.874 | 0.753 | 0.891 | 0.028 | **0.882** | **0.812** | <ins>0.918</ins> | **0.050** |
| v12-Progressive | – | <ins>0.875</ins> | 0.755 | 0.894 | 0.028 | <ins>0.879</ins> | 0.808 | 0.916 | <ins>0.051</ins> |
 
### CHAMELEON and NC4K
 
| Method | Year | CHAMELEON<br>S<sub>α</sub> ↑ | CHAMELEON<br>F<sup>ω</sup><sub>β</sub> ↑ | CHAMELEON<br>E<sub>φ</sub> ↑ | CHAMELEON<br>M ↓ | NC4K<br>S<sub>α</sub> ↑ | NC4K<br>F<sup>ω</sup><sub>β</sub> ↑ | NC4K<br>E<sub>φ</sub> ↑ | NC4K<br>M ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ANet-SRM | 2019 | – | – | – | – | – | – | – | – |
| SINet | 2021 | 0.872 | 0.806 | 0.946 | 0.034 | 0.808 | 0.723 | 0.883 | 0.058 |
| SLSR | 2021 | 0.890 | 0.822 | 0.948 | 0.030 | 0.840 | 0.766 | 0.907 | 0.048 |
| MGL-R | 2021 | 0.893 | 0.812 | 0.941 | 0.031 | 0.833 | 0.739 | 0.893 | 0.053 |
| PFNet | 2021 | 0.882 | 0.810 | 0.945 | 0.033 | 0.829 | 0.745 | 0.898 | 0.053 |
| UJSC | 2021 | 0.891 | 0.833 | <ins>0.955</ins> | 0.030 | 0.842 | 0.771 | 0.907 | 0.047 |
| MirrorNet | 2021 | – | – | – | – | – | – | – | – |
| C²F-Net | 2021 | 0.888 | 0.828 | 0.946 | 0.032 | 0.838 | 0.762 | 0.904 | 0.049 |
| UGTR | 2021 | 0.888 | 0.794 | 0.940 | 0.031 | 0.839 | 0.746 | 0.899 | 0.052 |
| ZoomNet | 2022 | 0.902 | <ins>0.845</ins> | **0.958** | **0.023** | 0.853 | 0.784 | 0.912 | 0.043 |
| HitNet | 2023 | 0.899 | 0.835 | 0.950 | 0.028 | 0.875 | <ins>0.818</ins> | <ins>0.927</ins> | **0.036** |
| ZoomNeXt | 2024 | <ins>0.909</ins> | **0.871** | **0.958** | <ins>0.024</ins> | **0.879** | **0.842** | **0.931** | <ins>0.038</ins> |
| CamoXpert v11 | – | 0.901 | 0.779 | 0.902 | 0.035 | 0.869 | 0.752 | 0.886 | 0.049 |
| **CamoXpert v12** | – | **0.912** | 0.840 | 0.947 | 0.026 | <ins>0.878</ins> | 0.794 | 0.908 | 0.042 |
| v12-Progressive | – | **0.912** | 0.837 | 0.939 | 0.026 | 0.874 | 0.789 | 0.911 | 0.043 |
 
## Architecture Verification
 
```bash
python verify_architecture.py
```
 
## Project Layout (Key Files)
 
```
configs/               # Training configs
methods/zoomnext/      # Model implementations (V11/V12)
main_for_image.py      # Training script
test_v11.py            # V11 benchmarking script
test_v12.py            # V12 benchmarking script
verify_architecture.py # Sanity check for V12 model
```
 
## Notes
 
- `test_v11.py` hardcodes `DEVICE = 'cuda'`. Switch to `'cpu'` in the script if needed (no CLI flag).
- `test_v12.py` auto-selects CUDA if available; edit `DEVICE` to force CPU.
- Verify dataset folder names (`Image/Mask` vs `Imgs/GT`) and update configs accordingly.
- For reproducibility, keep a record of checkpoint paths and data versions used in your paper.
- `--pretrained` loads backbone weights; use `--load-from` to resume a full CamoXpert checkpoint.
## License
 
No explicit license is provided. Treat the code as proprietary until you obtain written permission from the repository owner.
 
