"""R7 - CamoXpert (R5) + ZoomNeXt's uncertainty-aware loss (UAL), weight 1.0 with the same cosine ramp
as R0. Nothing else changes. Added 2026-11-28 AFTER R0/R5 were evaluated (see CHANGES.md 9): R5 beat R0 on
S-measure and maxEm but lost on wFm, MAE, meanEm and adpEm, and R5's maps had about 2x more uncertain pixels
(values in 0.1-0.9). Declared before training; its result is reported whatever it is.
"""
from _base import base_config, merge

CFG = merge(
    base_config(),
    dict(
        run_name="R7_camoxpert_ual",
        loss=dict(bce=1.0, iou=0.5, ssim=0.3, edge=0.2, ual=1.0, aux=0.1),
    ),
)
