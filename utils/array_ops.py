import os
import cv2
import numpy as np

def minmax(data_array: np.ndarray, up_bound: float = None) -> np.ndarray:
    if up_bound is not None:
        data_array = data_array / up_bound
    max_value = data_array.max()
    min_value = data_array.min()
    if max_value != min_value:
        data_array = (data_array - min_value) / (max_value - min_value)
    return data_array
