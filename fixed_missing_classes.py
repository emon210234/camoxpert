import os

file_path = "methods/zoomnext/zoomnext.py"

missing_classes = """
# --- Added missing classes to fix ImportError ---
class PvtV2B3_ZoomNeXt(PvtV2_ZoomNeXt):
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False):
        super().__init__(num_frames, pretrained, use_checkpoint, variant='b3')

class PvtV2B4_ZoomNeXt(PvtV2_ZoomNeXt):
    def __init__(self, num_frames, pretrained=False, use_checkpoint=False):
        super().__init__(num_frames, pretrained, use_checkpoint, variant='b4')
"""

with open(file_path, "a") as f:
    f.write(missing_classes)

print(f"Successfully added PvtV2B3 and PvtV2B4 to {file_path}")