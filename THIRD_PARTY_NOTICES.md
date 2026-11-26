# Third-party code in this repository

| File(s) here | Origin | Licence (as found) |
|---|---|---|
| `src/camoxpert/models/backbone/pvt_v2_eff.py` | ZoomNeXt, `methods/backbone/pvt_v2_eff.py` (commit `614af43`), itself adapted from PVTv2 | PVTv2: Apache-2.0 ([whai362/PVT](https://github.com/whai362/PVT), LICENSE checked at `57e2dfa`). ZoomNeXt: see note below |
| `src/camoxpert/models/ops.py` | ZoomNeXt `methods/zoomnext/ops.py`, verbatim | see note below |
| `src/camoxpert/models/zoomnext_layers.py` | ZoomNeXt `methods/zoomnext/layers.py`, verbatim | see note below |
| `src/camoxpert/models/zoomnext.py` (`_UpstreamPvtV2ZoomNeXt`) | ZoomNeXt `methods/zoomnext/zoomnext.py`, class `PvtV2B2_ZoomNeXt` | see note below |
| UAL term in `src/camoxpert/losses.py` | ZoomNeXt `_ZoomNeXt_Base.forward` / `get_coef` | see note below |
| Metrics (dependency, not copied) | [PySODMetrics](https://github.com/lartpang/PySODMetrics) | MIT (package metadata) |

CamoXpert's own decoder (`camoxpert_layers.py`, `camoxpert.py`) was written on top of the
ZoomNeXt code base: it keeps ZoomNeXt's overall layout (shared encoder on a three-scale input
pyramid, top-down decoder over stages 5 to 2) and the PVT/ops files, and replaces the decoder
blocks.

**Original work**

- ZoomNeXt: Y. Pang, X. Zhao, T.-Z. Xiang, L. Zhang, H. Lu, "ZoomNeXt: A Unified Collaborative
  Pyramid Network for Camouflaged Object Detection", IEEE TPAMI, 2024,
  doi:10.1109/TPAMI.2024.3417329. Code: https://github.com/lartpang/ZoomNeXt
- PVTv2: W. Wang et al., "PVT v2: Improved Baselines with Pyramid Vision Transformer",
  Computational Visual Media, 2022. Code: https://github.com/whai362/PVT

**Note on the ZoomNeXt licence.** The ZoomNeXt repository had no LICENSE file at commit
`614af43` (checked 2026-11-26). Without a licence, copyright law does not grant reuse rights
by default. Before this repository is made public with these files in it, ask the ZoomNeXt
authors for permission (or for them to add a licence), and keep this notice and the
per-file headers.

**Modifications to third-party files**

- `pvt_v2_eff.py`: two blocks marked `[CamoXpert patch]`: (1) the GPU query in `Attention.__init__`
  only runs when CUDA is available, so the model can be built on a CPU; (2) the deprecated
  `torch.backends.cuda.sdp_kernel(...)` is replaced by `torch.nn.attention.sdpa_kernel(...)` with
  the same list of attention kernels. Neither changes the computed values on a GPU.
- Everything else listed above is unchanged apart from added provenance comments.
