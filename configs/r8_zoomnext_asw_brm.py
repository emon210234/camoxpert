"""R8 - ZoomNeXt-B5 (R0) + CamoXpert's ASW and BRM modules, trained exactly like R0.

Loss = R0's loss (BCE 1.0 + UAL 1.0, same cosine ramp) + R5's edge loss (0.2) for the BRM edge head.
IoU/SSIM are NOT added, so R8 - R0 measures the two modules (+ the edge supervision they need).
Declared 2026-11-28 before training (CHANGES.md 10); its result is reported whatever it is.
"""
from _base import base_config, merge

CFG = merge(base_config(), dict(run_name="R8_zoomnext_asw_brm"))
CFG["model"] = dict(name="zoomnext_b5_plus", scales=[1.0, 0.75, 0.5], use_checkpoint=False, asw=True, brm=True)
CFG["loss"] = dict(bce=1.0, iou=0.0, ssim=0.0, edge=0.2, ual=1.0, aux=0.0)
