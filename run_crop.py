import rasterio
import cv2

from src.crop_pipeline.preprocess import tile_image
from src.crop_pipeline.inference import run_inference
from src.crop_pipeline.geo_utils import mask_to_geojson, save_geojson
from src.crop_pipeline.health_index import compute_ndvi_proxy


def run_pipeline(image_path):
    tiles = tile_image(image_path)

    # Try reading geospatial info
    try:
        with rasterio.open(image_path) as src:
            transform = src.transform
            crs = src.crs
    except:
        transform = None
        crs = None

    all_features = []

    for tile in tiles:
        # ✅ FIX: Use run_inference (NOT simple_segmentation)
        results = run_inference(tile)

        img = cv2.imread(tile["path"])

        for r in results:
            mask = r["mask"]
            confidence = r["confidence"]

            # Geo conversion only if available
            if tile["transform"] is not None:
                geojson = mask_to_geojson(
                    mask,
                    tile["transform"],
                    tile["crs"],
                    crop_type="generic_crop",
                    confidence=confidence
                )
            else:
                geojson = {"type": "FeatureCollection", "features": []}

            # Health index
            health = compute_ndvi_proxy(img)

            for f in geojson["features"]:
                f["properties"]["health_index"] = float(health)

            all_features.extend(geojson["features"])

    final_geojson = {
        "type": "FeatureCollection",
        "features": all_features
    }

    save_geojson(final_geojson)


if __name__ == "__main__":
    run_pipeline("data/raw/sample.tif")