"""R1a (optional) - ablation: Base without the GRU scans (fusion = 1x1 conv over concat[l, m, s])."""
from _base import base_config, merge

CFG = merge(
    base_config(),
    dict(
        run_name="R1a_base_noscan",
        model=dict(asw=False, brm=False, use_scan=False),
        loss=dict(bce=1.0, iou=0.0, ssim=0.0, edge=0.0, ual=0.0, aux=0.1),
    ),
)
