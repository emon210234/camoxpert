"""R0 - ZoomNeXt (PVTv2-B5), upstream network and loss (BCE + UAL), trained in OUR pipeline:
same data, augmentation, epochs, batch, optimizer, schedule, seed, evaluation AND the same input
pyramid as CamoXpert (448 / 336 / 224). The only differences from CamoXpert are the decoder and
the loss, so the comparison isolates them.

ZoomNeXt's own design uses 576 / 384 / 192 (0.5x / 1x / 1.5x of a 384 base). Its published
numbers (own recipe, own input size) are cited separately in the paper. To run that native
setting as an extra:  --set data.size=576 "model.scales=[1.0,0.6666666666666666,0.3333333333333333]"
"""
from _base import base_config, merge

CFG = merge(base_config(), dict(run_name="R0_zoomnext_b5"))
CFG["model"] = dict(name="zoomnext_b5", scales=[1.0, 0.75, 0.5], use_checkpoint=False)
CFG["loss"] = dict(bce=1.0, iou=0.0, ssim=0.0, edge=0.0, ual=1.0, aux=0.0)
