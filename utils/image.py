import cv2
import numpy as np
from .array_ops import minmax

def read_gray_array(path, div_255=False, to_normalize=False, thr=-1, dtype=np.float32) -> np.ndarray:
    assert path.endswith(".jpg") or path.endswith(".png"), path
    gray_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert gray_array is not None, f"Image Not Found: {path}"
    if div_255: gray_array = gray_array / 255
    if to_normalize: gray_array = minmax(gray_array, up_bound=255)
    if thr >= 0: gray_array = gray_array > thr
    return gray_array.astype(dtype)

def read_color_array(path: str):
    bgr_array = cv2.imread(path)
    assert bgr_array is not None, f"Image Not Found: {path}"
    return cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
