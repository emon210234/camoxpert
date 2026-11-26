# CamoXpert

Code for the manuscript on CamoXpert, a camouflaged object detection (COD) network with a
PVTv2-B5 encoder, a three-scale input pyramid, and a decoder that fuses the scales with
directional GRU scans and a two-expert mixture, plus optional adaptive scale weighting (ASW),
a boundary refinement module (BRM) and progressive refinement (PR).

**Built on ZoomNeXt.** This code base started from ZoomNeXt (Pang et al., TPAMI 2024,
https://github.com/lartpang/ZoomNeXt). The PVTv2 backbone file, the basic ops and the ZoomNeXt
baseline network are copied from it (see `THIRD_PARTY_NOTICES.md`). Please cite ZoomNeXt and
PVTv2 if you use this code.

## Models

| Name in the paper | Config | What it is |
|---|---|---|
| CamoXpert-Base | `configs/r1_camoxpert_base.py` | PVTv2-B5 + scan/MoE fusion decoder, BCE + MoE balance loss |
| CamoXpert | `configs/r5_camoxpert.py` | Base + ASW + BRM, BCE + IoU + SSIM + edge loss |
| CamoXpert-PR | `configs/r6_camoxpert_pr.py` | CamoXpert + 2-step progressive refinement |
| ZoomNeXt-B5 (our pipeline) | `configs/r0_zoomnext_b5.py` | upstream ZoomNeXt network and loss, trained with our protocol |
| ablations | `configs/r2_*.py`, `r3_*.py`, `r4_*.py`, `r1a_*.py`, `r1b_*.py` | one component at a time |

All runs share the protocol in `configs/_base.py`; only the model and loss differ.

## Install (Windows or Linux, NVIDIA GPU)

Requires [uv](https://docs.astral.sh/uv/). From the repository folder:

```powershell
uv sync                     # Python 3.11 + PyTorch 2.7.1 (CUDA 12.6) + pinned dependencies -> .venv, writes uv.lock
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest -q            # 29 CPU tests, about a minute
```

The ImageNet weights of PVTv2-B5 are downloaded on first use from the official PVT release
into the torch hub cache. Offline machines: download `pvt_v2_b5.pth` from
https://github.com/whai362/PVT/releases/tag/v2 and add `--set train.pretrained_path=C:/path/pvt_v2_b5.pth`.

## Data

Standard COD split (the SINet-V2 release):

```
C:/data/COD/TrainDataset/{Imgs,GT}                 4,040 (3,040 COD10K + 1,000 CAMO)
C:/data/COD/TestDataset/CAMO/{Imgs,GT}               250
C:/data/COD/TestDataset/CHAMELEON/{Imgs,GT}           76
C:/data/COD/TestDataset/COD10K/{Imgs,GT}           2,026
C:/data/COD/TestDataset/NC4K/{Imgs,GT}             4,121
```

```powershell
uv run python tools/check_data.py --root C:/data/COD --out C:/runs/data_manifest.json --full
```

## Train, evaluate, profile

```powershell
# one run (seed 0); writes config.json, run_meta.json, train_log.jsonl, last.pth, model_final.pth
uv run python tools/train.py --config configs/r5_camoxpert.py --out C:/runs/R5_camoxpert_s0
# after a crash or reboot
uv run python tools/train.py --config configs/r5_camoxpert.py --out C:/runs/R5_camoxpert_s0 --resume
# other seeds
uv run python tools/train.py --config configs/r5_camoxpert.py --out C:/runs/R5_camoxpert_s1 --set seed=1

# evaluation of the final checkpoint (single pass), with prediction maps
uv run python tools/eval.py --run C:/runs/R5_camoxpert_s0 --save-maps
# multi-scale + flip TTA, reported separately
uv run python tools/eval.py --run C:/runs/R5_camoxpert_s0 --tta

# parameters, FLOPs, speed
uv run python tools/profile_model.py --config configs/r5_camoxpert.py --out C:/runs/profile

# LaTeX tables from the JSON results (see the docstring for the spec format)
# (copy tables/*.example.json first and fill in the run folders / published numbers)
uv run python tools/make_tables.py --spec tables/main_table.json --out tables/main.tex
```

Protocol rules the scripts enforce: the standard 4,040-image training set, a fixed number of
epochs, evaluation of the last checkpoint only (no test data is used during training), and one
evaluation code path for every model.

## Verification

```powershell
uv run python tests/e2e_smoke.py                                    # fake data, whole pipeline, ~5 min on CPU
uv run python tools/verify/against_legacy.py --legacy-root ../_archive/camoxpert_pre_P2
uv run python tools/verify/zoomnext_port.py --upstream-root ../ZoomNeXt_upstream   # git clone of lartpang/ZoomNeXt
```

The last two show that the CamoXpert classes reproduce the pre-refactor classes and that the
ZoomNeXt port reproduces upstream ZoomNeXt exactly (same outputs and losses for the same weights).
`CHANGES.md` lists every change from the pre-refactor code and why it was made.
