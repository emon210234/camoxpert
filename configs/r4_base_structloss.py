"""R4 - ablation: Base architecture trained with BCE + 0.5 IoU + 0.3 SSIM (no edge head, no ASW)."""
from _base import base_config, merge

CFG = merge(
    base_config(),
    dict(
        run_name="R4_base_structloss",
        model=dict(asw=False, brm=False),
        loss=dict(bce=1.0, iou=0.5, ssim=0.3, edge=0.0, ual=0.0, aux=0.1),
    ),
)
