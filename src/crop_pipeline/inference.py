import cv2
import numpy as np


def simple_segmentation(image):
    """
    Accepts image array (NOT path) for better pipeline integration
    """

    
    lower = np.array([25, 40, 20])
    upper = np.array([100, 255, 100])

    mask = cv2.inRange(image, lower, upper)

    # Confidence derived from mask coverage
    confidence = float(np.mean(mask) / 255.0)

    return mask, confidence


def run_inference(tile_input):
    """
    Standardized inference wrapper for agent compatibility

    Accepts:
    - tile path (string)
    - tile dict (from preprocess)

    Returns:
    - list of results (standard format)
    """

    
    if isinstance(tile_input, dict):
        image_path = tile_input["path"]
    else:
        image_path = tile_input

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    mask, confidence = simple_segmentation(img)

    
    results = [{
        "mask": mask,
        "confidence": confidence
    }]

    return results