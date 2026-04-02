import numpy as np

def compute_density(tracks, shape):
    """
    Compute spatial density map from tracked objects

    Args:
        tracks: list of track dicts
        shape: frame shape (H, W, C)

    Returns:
        density_map (H x W)
    """

    h, w = shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)

    for t in tracks:
        x, y = t["center"]

        x = int(x)
        y = int(y)

        if 0 <= x < w and 0 <= y < h:
            density_map[y, x] += 1.0

    return density_map