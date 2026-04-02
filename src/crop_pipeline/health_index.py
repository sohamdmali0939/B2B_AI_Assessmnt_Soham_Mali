import numpy as np

def compute_ndvi_proxy(img):
    r = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)

    ndvi = (g - r) / (g + r + 1e-6)
    return np.mean(ndvi)