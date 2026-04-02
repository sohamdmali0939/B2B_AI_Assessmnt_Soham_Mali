import rasterio
from shapely.geometry import Polygon, mapping
import json
import cv2
import os


def pixel_to_geo(transform, x, y):
    """
    Convert pixel coordinates → geographic coordinates using affine transform
    """
    if transform is None:
        raise ValueError("Missing transform — cannot convert to geo coordinates")

    lon, lat = rasterio.transform.xy(transform, y, x)
    return lon, lat


def mask_to_geojson(mask, transform, crs, crop_type, confidence):
    """
    Convert segmentation mask → GeoJSON FeatureCollection
    """

    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []

    for contour in contours:
        coords = []

        for point in contour:
            x, y = point[0]

            try:
                lon, lat = pixel_to_geo(transform, x, y)
                coords.append((lon, lat))
            except:
                continue  # skip invalid points

        
        if len(coords) > 3:
            polygon = Polygon(coords)

            
            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            features.append({
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "crop_type": crop_type,
                    "confidence": float(confidence)
                }
            })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    
    if crs is not None:
        geojson["crs"] = {
            "type": "name",
            "properties": {
                "name": str(crs)
            }
        }

    return geojson


def save_geojson(geojson, path="outputs/geojson/crops.geojson"):
    """
    Save GeoJSON to file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(geojson, f, indent=4)