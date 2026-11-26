# CHANGES — P2 code rework (2026-11-26)

This file explains every change from the pre-P2 code (archived unchanged in
`../_archive/camoxpert_pre_P2/`) to this version, so that you can defend each one to a reviewer.
Read it next to the code: every module's docstring states its maths.

## 0. The one-paragraph summary

The network is unchanged. The new `CamoXpert` class, with switches, reproduces the legacy
`CamoXpertV11`, `CamoXpertV12_Base` and `CamoXpertV12_Progressive` **bit for bit** (same weights,
same input → difference 0.0 in logits and in the training loss; `tools/verify/against_legacy.py`).
What changed is everything around the network: the data protocol, input normalisation,
learning-rate schedule, mixed precision, checkpointing, logging, evaluation and reporting. The
"ZoomNeXt" class in the old repo was not ZoomNeXt; the real upstream network is now ported and
verified bit for bit against the upstream repo (`tools/verify/zoomnext_port.py`).

## 1. How the equivalence was verified

`tools/verify/against_legacy.py` builds each legacy class with random weights, copies the weights
into the new class, feeds both the same 224×224 input (CPU, float32) and compares:

| Legacy class → new config | params legacy | params new | max \|Δ logits\| | loss legacy = new |
|---|---|---|---|---|
| CamoXpertV11 → CamoXpert(asw=False, brm=False) | 82,648,775 | 82,636,103 | 0.0 | 0.893838 = 0.893838 |
| CamoXpertV12_Base → CamoXpert(asw, brm) | 82,827,073 | 82,814,401 | 0.0 | 1.675030 = 1.675030 |
| CamoXpertV12_Progressive → CamoXpert(…, progressive_iters=2) | 83,050,307 | 83,037,058 | 0.0 | 3.119442 = 3.119442 |

(Loss values are for random weights and a random mask: they only show the two codes agree.)
The parameter difference is exactly the dead weights removed (C15). The same check for ZoomNeXt:
84,773,821 parameters in both, Δ logits 0.0, loss identical at training progress 0, 0.3, 0.7, 1.0.
Both scripts were run with PyTorch 2.15 and with 2.7.1 (the version pinned for the VM).

Two more checks: `tests/test_units.py` (29 CPU tests: losses, LR schedule, configs, every model
forwards and backwards, metrics on a perfect prediction) and `tests/e2e_smoke.py` (fake data:
train → same seed gives an identical loss curve → resume → eval single/TTA → profile → table).

## 2. Findings in the legacy code (C1–C8 from PLAN.md, C9–C18 new)

**C1 Training data.** Legacy trained on COD10K-v3 `Train/Image` (6,000 images incl.
non-camouflaged ones with empty masks) + CAMO. Now: the standard 3,040 + 1,000 = 4,040 set;
`train.py` refuses to start with a different count.

**C2 Metrics.** `test_v12.py` computed only S-measure and MAE. Now `eval.py` stores S_α,
F_β^w, MAE, and the mean / max / adaptive variants of E_φ and F_β, so the paper can name the
variant it reports (M3).

**C3 Test-time augmentation.** Legacy evaluation averaged 6 passes (3 scales × flip). Now
single-pass is the default; `--tta` reproduces the legacy TTA and writes a separate file.

**C4 Per-image min-max (correction of the Round-0 review).** PySODMetrics already min-max
normalises every prediction and binarises the GT at 128 before computing anything, like the MATLAB
toolbox used by SINet-V2 and most COD papers. The extra normalisation in `test_v12.py` was
therefore redundant and harmless. Not a protocol problem.

**C5 Training bookkeeping.** Legacy: no seed, weights saved every epoch, no log file, no
optimizer/scheduler state. Now: seeded (Python, NumPy, PyTorch, DataLoader shuffle, per-worker
seeds), `last.pth` with the full state for `--resume`, `model_final.pth` for evaluation,
`train_log.jsonl`, `config.json`, `run_meta.json` (git commit, versions, GPU, data fingerprint).

**C6 Ablation switches.** Now `asw`, `brm`, `progressive_iters`, `use_scan`, `use_moe` and
the loss weights are config values, so every ablation is a config file, never a code edit.

**C7 Dead code.** CamoXpertV2–V10 placeholders, moe_v3/v4/v5, apply_fixes.py, duplicate utils,
video scripts, LaTeX build files: all archived, none needed for the paper.

**C8 The legacy `PvtV2B5_ZoomNeXt` was not ZoomNeXt.** Its `body()` was `pass` (it returned
`None`, so it could not even be trained), and its `MHSIU`, `RGPU`, `SimpleASPP` were simplified
blocks that only share names with ZoomNeXt's. The real network is now in
`models/zoomnext.py`, copied from upstream and verified (section 1).

**C9 The committed v12 config cannot be parsed.** `configs/camoxpert_v12.py` (commit 83cabb8)
contains `datasets[` instead of `datasets=[` → `SyntaxError: positional argument follows keyword
argument`. Whatever produced the old v12 numbers was not this file as committed. One more reason
the old numbers are void.

**C10 Double input normalisation.** The legacy loader applied `A.Normalize()` (ImageNet mean/std)
and the model applied `PixelNormalizer` (the same mean/std) again:
x' = ((x − μ)/σ − μ)/σ. With μ, σ of the red channel (0.485, 0.229), a pixel in [0, 1] became a
value in [−11.4, 7.7] instead of [−2.12, 2.25] (all channels: about [−11.4, 9.9]). The pretrained
PVTv2 was trained on the once-normalised range, so its first layers received inputs about 4–5×
too large. Training still works (the network adapts), but it wastes the pretraining. Now the
loader returns [0, 1] and only the model normalises. (This changes training, so it is one of the
reasons all runs are new.)

**C11 Learning-rate schedule.** Two bugs in `utils/scheduler.py`: (a) the cosine phase decayed
to `gamma` (default 0.1), ignoring the config's `min_coef=0.01`, so the lr ended at 10 %, not
1 %; (b) the scheduler was called *after* `optimizer.step()`, so iteration 0 used the full lr
instead of the warm-up start and every later iteration used the previous iteration's value. Now
`engine.lr_factor` (formula in its docstring) is applied before every step and ends at
`sched.min_factor` (0.01).

**C12 "AMP" was never on.** `use_amp=True` created a GradScaler but nothing ever entered
`autocast`, so legacy training ran in float32. Now `train.amp="bf16"` really uses bf16 autocast
(RTX 4090 supports it; no loss scaling needed). Losses are always computed in float32 outside
autocast, and the GRU scan runs in float32 inside autocast (rounding would accumulate along
the 42-step recurrence). `train.amp="none"` gives plain float32 if you prefer.

**C13 Train/test resize mismatch.** The legacy training built the input pyramid with
`align_corners=True`, the test script with `align_corners=False`. Now one function
(`models.camoxpert.make_pyramid`, `align_corners=False`) does both.

**C14 Missing masks became empty masks.** The legacy loader silently used an all-zero mask when
a GT file was missing. Now `data.list_pairs` raises an error naming the files.

**C15 Dead parameters.** Legacy `SimpleASPP` built `conv1x1_2` but never used it
(3 × 4,224 = 12,672 parameters), and progressive mode built an unused `proj` head (577). They
inflated the parameter count; now removed.

**C16 The upstream PVT file cannot be built without a GPU.** It calls
`torch.cuda.get_device_properties` in `Attention.__init__`, and uses the deprecated
`torch.backends.cuda.sdp_kernel`. Two marked patches fix this without changing GPU results
(THIRD_PARTY_NOTICES.md).

**C17 Edge loss under autocast.** The edge loss is a BCE on probabilities
(`F.binary_cross_entropy`), which PyTorch refuses inside autocast. It worked in legacy only
because autocast was never on (C12). Now it runs in float32 outside autocast, with the legacy
order of operations kept: sigmoid at the edge-head resolution → bilinear upsampling → BCE
(upsampling logits first would give a different number, because sigmoid is nonlinear).

**C18 ZoomNeXt has no licence file.** See THIRD_PARTY_NOTICES.md: ask the authors before the
repo is made public; credit them in the paper (F5).

## 3. Names: old → new

| Legacy | New | Why |
|---|---|---|
| `CamoXpertV11` | `CamoXpert(asw=False, brm=False)` = **CamoXpert-Base** | version numbers are lab history (m3) |
| `CamoXpertV12` / `_Base` | `CamoXpert()` = **CamoXpert** | |
| `CamoXpertV12_Progressive` | `CamoXpert(progressive_iters=2)` = **CamoXpert-PR** | |
| `MambaMoE` | `ScanMoEFusion` | no Mamba/SSM inside (F2) |
| `CrossScaleSelectiveScan` | `RecurrentScanFusion` | it is two GRUCells, not a selective scan |
| `SimpleASPP` (CamoXpert's) | `DilatedContextBlock` | differs from ZoomNeXt's SimpleASPP (no global branch) |
| `RGPU` (CamoXpert's) | `LowLevelGuidance` | unrelated to ZoomNeXt's RGPU |
| `SpectralRouter`, `ConvExpert`, `AdaptiveScaleWeighting`, `BoundaryRefinementModule`, `ProgressiveRefinement` | same names | |

Attribute names inside the model (`siu_5`, `hmu_4`, `rgpu`, …) are kept, which is what lets the
legacy weights load directly into the new classes.

What the code really does, for the methodology section (all from the docstrings):
- **Fusion** (`ScanMoEFusion`): l and s resized to m; 1×1 conv over the concatenation; one GRU scan
  down every column and one along every row (2 directions only, F3); sum, sigmoid gate, 1×1 conv,
  residual; then a soft mixture of 2 conv experts weighted by an *image-level* router fed with
  global-average-pooled features and pooled |Laplacian| responses; load-balancing loss
  0.1 × 10 × mean_e(d_e − 1/2)².
- **ASW**: per-image softmax weight per scale from pooled features (a scalar per scale).
- **BRM**: edge head → sigmoid → concatenated back → 3 convs; edge supervised with Sobel edges of
  the GT (threshold 0.1), not with the dataset's Edge folders.
- **PR**: 2 heads in sequence, each re-reads the same decoder features plus the detached previous
  prediction; loss weights 1.0 and 1.2.
- **Output** at stride 4 (112×112 for a 448 input), upsampled bilinearly for loss and evaluation.

## 4. The protocol now fixed for every run (`configs/_base.py`)

| Item | Value | Legacy |
|---|---|---|
| Train set | TrainDataset, 4,040 pairs (3,040 COD10K + 1,000 CAMO) | COD10K-v3 Train/Image (6,000, incl. non-camouflaged) + CAMO-V.1.0 Train (count not recorded) |
| Test sets | CAMO 250, CHAMELEON 76, COD10K 2,026, NC4K 4,121 | same names, other copies |
| Input | 448 (pyramid 448/336/224); masks supervised at 448 | same |
| Augmentation | resize, h-flip, v-flip, rotate ±15°, ColorJitter p=0.3 | same |
| Normalisation | once, in the model | twice (C10) |
| Optimiser | AdamW, lr 5e-5 decoder / 5e-6 backbone, wd 1e-4 | same |
| Schedule | 1,000-iter linear warm-up from 1 %, cosine to 1 % | cosine to 10 %, off by one (C11) |
| Batch / epochs | 8 / 120 (epochs PROVISIONAL until the P3 timing run) | 8 / 120 |
| Precision | bf16 autocast (scan + losses in fp32) | fp32 (C12) |
| Seed | 0 (R5b: 1, 2) | none |
| Model selection | none: last epoch only, no test data during training | unknown |
| Evaluation | single pass, sigmoid, bilinear to GT size, PySODMetrics | 6-pass TTA, S/MAE only |

`tests/test_units.py::test_protocol_identical_across_runs` fails if any config changes
anything other than the model, the loss, or R0's input size.

## 5. Decisions taken in P2 that you should confirm (all are one config line)

1. **bf16 vs float32.** bf16 autocast is usually faster on an RTX 4090 and usually changes
   results only at noise level, but neither is measured yet for this model. The P3 timing run
   measures both (`--set train.amp=none` for float32); then one setting is fixed in `_base.py`
   before the first paper run and kept for all runs.
2. **ZoomNeXt (R0) input size.** R0 uses ZoomNeXt's own pyramid (576/384/192). Alternative:
   `--set data.size=448` to give it exactly CamoXpert's input. The paper reports which one.
3. **Cosine floor** 1 % (`sched.min_factor=0.01`), as the legacy config intended.
4. **Optional ablations** R1a (no GRU scans) and R1b (single expert, no router) test the fusion
   block itself; they cost two more runs. Without them a reviewer may ask whether the scan or the
   MoE does anything.
5. **GT for metrics**: raw GT into PySODMetrics (binarised at 128 inside). ZoomNeXt's own
   evaluator first set every GT pixel > 0 to 255; the two differ only on GT pixels with gray
   values 1–128 (`tools/check_data.py` counts masks that have such values).

## 6. File map

| New | Replaces |
|---|---|
| `src/camoxpert/models/camoxpert.py`, `camoxpert_layers.py` | `methods/zoomnext/zoomnext.py` (V11 part), `camoxpert_v12.py`, `improved_modules.py`, `layers.py` |
| `src/camoxpert/models/zoomnext.py`, `zoomnext_layers.py` | the broken `PvtV2*_ZoomNeXt` classes |
| `src/camoxpert/models/backbone/pvt_v2_eff.py`, `ops.py` | same files (upstream) |
| `src/camoxpert/losses.py` | `CamouflageDetectionLoss` + losses inside `forward()` |
| `src/camoxpert/data.py` | `KaggleCODDataset` in `main_for_image.py` |
| `src/camoxpert/metrics.py`, `tools/eval.py` | `test_v11.py`, `test_v12.py` |
| `src/camoxpert/engine.py`, `config.py`, `tools/train.py` | `main_for_image.py`, `utils/*`, mmengine configs |
| `tools/profile_model.py` | none (M2) |
| `tools/check_data.py`, `tools/make_tables.py` | none |
| `configs/r*.py` | `configs/camoxpert_v*.py`, `rtx4090_train.py`, `icod_*.py`, `vcod_*.py` |

## 7. Reading order (to learn the code)

1. `models/camoxpert.py` — the forward pass, top to bottom.
2. `models/camoxpert_layers.py` — each block, with its equations.
3. `losses.py` — what is minimised.
4. `engine.py` — `lr_factor` and the optimizer groups.
5. `tools/train.py` — the loop: set lr → forward (autocast) → loss (fp32) → backward → step.
6. `metrics.py`, `tools/eval.py` — how a number in the paper is produced.
