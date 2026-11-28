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
2. **ZoomNeXt (R0) input size — decided 2026-11-27: same pyramid as CamoXpert (448/336/224).**
   Then data, recipe and input are identical and only the decoder and loss differ, so R0
   isolates the decoder. ZoomNeXt's native 576/384/192 setting is covered by citing its published
   numbers; it can be run as an extra (command in the r0 config docstring).
3. **Cosine floor** 1 % (`sched.min_factor=0.01`), as the legacy config intended.
4. **Ablations R1a (no GRU scans) and R1b (single expert, no router) — decided 2026-11-27: run
   both.** They test whether the scan and the mixture of experts contribute anything.
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

## 8. P3 addendum (2026-11-27): first timing run crashed on the lab VM

`tools/train.py` died on the first forward pass with `RuntimeError: CUDA error: unknown error`
(inside a PVTv2 depthwise conv). The legacy code did train on this VM, and differs in three
GPU-side settings: float32 (no autocast), cuDNN benchmark off, no pinned memory.
- New config keys `train.cudnn_benchmark` and `train.pin_memory` (defaults True = unchanged
  behaviour), so each setting can be switched off without a code edit.
- New `tools/diag_gpu.py`: tests each setting in a separate process and reports which crashes.
The fix chosen from its output is recorded here before any paper run.

Result (2026-11-27, lab VM, driver 610.88, torch 2.7.1+cu126):
- `diag_gpu.py`: all 11 cases passed, including 3 bf16 and 3 fp32 training steps with
  cudnn.benchmark on (random weights and random data, no DataLoader).
- Real R5 timing run (bf16, benchmark ON, pin_memory ON): crashed again in the first forward pass,
  same error, same PVTv2 dwconv. That makes 2 crashes out of 2 real runs with benchmark on.
- Same run with `--set train.cudnn_benchmark=False` (bf16, pin_memory ON): 300 iterations
  without error, 0.562 s/it, peak 11.6 GB (C:/runs/timing_R5_bf16_nobench/timing.json).
- The Windows System log has no nvlddmkm or Display events for the day, so a TDR driver reset
  is NOT confirmed. Why benchmark mode crashes only in real training is still unknown.
Decision: `train.cudnn_benchmark=False` in configs/_base.py for every run. This is the legacy
setting and does not change the maths of training: cuDNN only chooses its conv algorithm by a
fixed rule instead of timing them. pin_memory stays on (it worked).

Bug found by the fp32 timing run (2026-11-27): `--set train.amp=none` crashed with `KeyError: None`.
The override parser turns the text "none" into Python `None`, but `autocast_ctx` in tools/train.py
only tested for the string "none". The CPU smoke test never reached that line (on CPU autocast is
skipped first), so the tests did not catch it. Fix: `autocast_ctx` accepts both `None` and "none",
and rejects any other unknown value with a clear message. No effect on bf16 runs.

Training protocol frozen (2026-11-27), from the P3 timing runs (300 iterations, first 50 not timed,
batch 8, benchmark off; files C:/runs/timing_*/timing.json on the VM):
- Precision: `train.amp="none"` (float32). R5: fp32 0.507 s/it vs bf16 0.562 s/it in training;
  batch-1 forward 131.8 ms fp32 vs 165.0 ms bf16 (tools/profile_model.py). Peak 18.1 GB fits 24 GB.
  The code sets no TF32 flag, so PyTorch defaults apply: cuDNN convolutions may use TF32,
  matmuls use full float32. The paper must state precision this way.
- Epochs: 120 (legacy), the same for every run, last checkpoint, chosen before any test result.
- num_workers: 2. With 4 workers, free RAM on the 8 GB VM fell from 4.88 GB to 0.70 GB at the
  first iteration; the run was stopped by hand (C:/runs/timing_R5_fp32_w4, not used).
- With 2 workers GPU-Util averaged 69% (R5 fp32), so data loading partly limits speed. Accepted.


## 9. After R5 and R0 (2026-11-28): R7 = CamoXpert + UAL
Same protocol, single pass, R5 minus R0 (4 decimals from eval output):
| set | Sm | wFm | MAE (lower better) | meanEm | maxEm | adpEm |
|---|---|---|---|---|---|---|
| CAMO | +0.0035 | -0.0109 | +0.0031 | -0.0027 | +0.0013 | -0.0063 |
| CHAMELEON | +0.0046 | -0.0106 | +0.0033 | -0.0013 | +0.0051 | -0.0078 |
| COD10K | +0.0039 | -0.0137 | +0.0022 | -0.0029 | +0.0044 | -0.0071 |
| NC4K | +0.0035 | -0.0103 | +0.0026 | -0.0047 | +0.0004 | -0.0065 |
Pattern on all 4 sets: CamoXpert wins the structure metric (Sm) and the best-threshold metric (maxEm);
ZoomNeXt wins every metric that uses the raw grey values (wFm, MAE) or a fixed adaptive threshold (adpEm).
Map check (saved uint8 maps, share of pixels with value 26..229, i.e. probability 0.1..0.9):
CAMO R5 4.80% vs R0 2.05%; CHAMELEON 4.00% vs 1.55%; COD10K (first 810 of 2026 maps) 3.12% vs 1.41%.
So CamoXpert's maps are less confident. ZoomNeXt's UAL term, c(t) * mean(1 - |2p-1|^2), is largest at
p = 0.5 and zero at p = 0 or 1, so it pushes pixels toward a confident answer; R5 had ual = 0.
R7 adds it with R0's weight (1.0) and ramp; no value was tuned. Disclosure for the paper: this change was
motivated by test-set results, it is one declared variant, and R5, R0 and R7 are all reported.

## 10. R8 = ZoomNeXt-B5 + ASW + BRM (2026-11-28, declared before training)
Question: do CamoXpert's two own modules improve a strong baseline? (R8 vs R0, same protocol.)
Code: `models/zoomnext.py` class `ZoomNeXtB5Plus` (registry name `zoomnext_b5_plus`), config
`configs/r8_zoomnext_asw_brm.py`. Forward pass, per decoder stage k = 5, 4, 3, 2 (deep to shallow):
    l, m, s = tra_k(features of each scale)            # 64 channels each (ZoomNeXt, unchanged)
    [ASW]   l, m, s = w_l*l, w_m*m, w_s*s             # w = softmax(MLP(GAP(l), GAP(m), GAP(s)))
    x = RGPU_k( MHSIU_k(l, m, s) + upsample(x) )       # ZoomNeXt's fusion + refinement, unchanged
then [BRM] x, edge = BRM(x); logits = predictor(tra_1(x)).
Placement choices (state them in the paper):
* ASW before every multi-scale fusion. ZoomNeXt fuses at 4 stages (CamoXpert at 3), so 4 ASW blocks.
* BRM at the same place in the decoder as in CamoXpert (just before the prediction head). ZoomNeXt
  fuses at the medium scale's resolution, so BRM runs at 84 x 84 for S = 448 (CamoXpert: 112 x 112).
  ZoomNeXt's logits are at 0.75 S (336 x 336), same as R0; loss and evaluation resize them.
* Loss = R0's (BCE 1.0 + UAL 1.0) + edge 0.2 (R5's weight; BRM's edge head needs supervision).
  No IoU/SSIM, so R8 - R0 measures the modules (+ their edge supervision), nothing else.
* Nothing tuned. Params: R0 84,773,821, R8 84,966,650 (+192,829).
Checks (CPU, PyTorch 2.15): 35 unit tests pass, incl. `test_zoomnext_plus_without_modules_equals_zoomnext`
(asw=False, brm=False gives bit-identical logits to ZoomNeXtB5 for the same seed) and output shapes.
Not yet measured: GPU memory and speed on the VM (R0 peaked at 19.5 GB; the extra modules are small,
but watch the first log lines).
