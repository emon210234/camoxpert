"""R1 - CamoXpert-Base (legacy v11 / CamoXpertV11): PVTv2-B5 + scan/MoE fusion decoder, BCE + MoE aux loss."""
from _base import base_config, merge

CFG = merge(
    base_config(),
    dict(
        run_name="R1_camoxpert_base",
        model=dict(asw=False, brm=False),
        loss=dict(bce=1.0, iou=0.0, ssim=0.0, edge=0.0, ual=0.0, aux=0.1),
    ),
)
