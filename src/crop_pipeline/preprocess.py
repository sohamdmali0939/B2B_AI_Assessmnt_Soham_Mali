import cv2
import numpy as np
import os
import rasterio


def normalize(image):
    """
    Normalize image to [0,1] and scale back for saving.
    Keeps pipeline compatible with OpenCV.
    """
    image = image.astype(np.float32) / 255.0
    return (image * 255).astype(np.uint8)


def tile_image(image_path, tile_size=512, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)

    tiles = []

    
    try:
        with rasterio.open(image_path) as src:
            img = src.read([1, 2, 3])  # Read RGB bands
            img = np.transpose(img, (1, 2, 0))  # Convert to HWC
            transform = src.transform  # preserve for future use
            crs = src.crs
    except:
        
        img = cv2.imread(image_path)
        transform = None
        crs = None

    if img is None:
        raise ValueError(f"Image not found or unreadable: {image_path}")

    h, w, _ = img.shape

    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):

            tile = img[y:y + tile_size, x:x + tile_size]

            # Only keep full tiles
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:

                # ✅ NORMALIZATION (NO HARDCODING)
                tile = normalize(tile)

                tile_path = os.path.join(output_dir, f"tile_{x}_{y}.png")
                cv2.imwrite(tile_path, tile)

                tiles.append({
                    "path": tile_path,
                    "x": x,
                    "y": y,
                    "transform": transform,
                    "crs": crs
                })

    return tiles