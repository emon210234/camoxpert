"""R3 - ablation: Base + Boundary Refinement Module with its edge loss (no ASW, no IoU/SSIM)."""
from _base import base_config, merge

CFG = merge(
    base_config(),
    dict(
        run_name="R3_base_brm",
        model=dict(asw=False, brm=True),
        loss=dict(bce=1.0, iou=0.0, ssim=0.0, edge=0.2, ual=0.0, aux=0.1),
    ),
)
