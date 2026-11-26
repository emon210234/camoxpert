"""R5 - CamoXpert (legacy v12 / CamoXpertV12_Base): Base + ASW + BRM + structure & edge loss. Main result."""
from _base import base_config, merge

CFG = merge(base_config(), dict(run_name="R5_camoxpert"))
