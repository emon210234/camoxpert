"""R2 - ablation: Base + Adaptive Scale Weighting only (loss as Base)."""
from _base import base_config, merge

CFG = merge(
    base_config(),
    dict(
        run_name="R2_base_asw",
        model=dict(asw=True, brm=False),
        loss=dict(bce=1.0, iou=0.0, ssim=0.0, edge=0.0, ual=0.0, aux=0.1),
    ),
)
