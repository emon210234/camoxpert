"""R0 - ZoomNeXt (PVTv2-B5), upstream network and loss (BCE + UAL), trained in OUR pipeline:
same data, augmentation, epochs, batch, optimizer, schedule, seed and evaluation as CamoXpert.

Input pyramid = ZoomNeXt's native one: 576 / 384 / 192 (the upstream 0.5x/1x/1.5x of a 384 base).
To run it at CamoXpert's input size instead:  --set data.size=448
The official ZoomNeXt numbers (their own recipe) are cited separately in the paper.
"""
from _base import base_config, merge

CFG = merge(base_config(), dict(run_name="R0_zoomnext_b5", data=dict(size=576)))
CFG["model"] = dict(name="zoomnext_b5", scales=[1.0, 2.0 / 3.0, 1.0 / 3.0], use_checkpoint=False)
CFG["loss"] = dict(bce=1.0, iou=0.0, ssim=0.0, edge=0.0, ual=1.0, aux=0.0)
