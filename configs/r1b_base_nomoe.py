"""R1b (optional) - ablation: Base with a single conv expert instead of the 2-expert routed mixture."""
from _base import base_config, merge

CFG = merge(
    base_config(),
    dict(
        run_name="R1b_base_nomoe",
        model=dict(asw=False, brm=False, use_moe=False),
        loss=dict(bce=1.0, iou=0.0, ssim=0.0, edge=0.0, ual=0.0, aux=0.0),
    ),
)
