"""R6 - CamoXpert-PR (legacy CamoXpertV12_Progressive): CamoXpert + 2-step progressive refinement,
every step supervised (weights 1.0, 1.2), edge loss on the last step."""
from _base import base_config, merge

CFG = merge(base_config(), dict(run_name="R6_camoxpert_pr", model=dict(progressive_iters=2)))
