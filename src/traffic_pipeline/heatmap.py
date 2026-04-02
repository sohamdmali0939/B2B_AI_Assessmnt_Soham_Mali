import cv2
import numpy as np

def generate_heatmap(density_map):
    """
    Generate visual heatmap from density map
    """

    if density_map.max() == 0:
        return np.zeros((*density_map.shape, 3), dtype=np.uint8)

    # Smoothing the maps
    heatmap = cv2.GaussianBlur(density_map, (21, 21), 0)

    # Normalize the maps
    heatmap = heatmap / (heatmap.max() + 1e-6)
    heatmap = (heatmap * 255).astype(np.uint8)

    # Applying color
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap