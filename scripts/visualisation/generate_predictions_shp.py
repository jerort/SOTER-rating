"""
Generate a shapefile with prediction attributes merged onto road centroids.

Steps
-----
1. Read the last PRED_XXX sheet from labels.xlsx → build an attribute table
   (Identifier, Label, Raw Estimate, Estimate, Subset), dropping duplicate IDs.
2. Merge with all_roads.csv on identifier.
3. Load every *centroids.shp from roads/all_clips, concatenate them, and merge
   with the attribute table to produce a single output shapefile.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.labels import build_attribute_table


# ======================================================================
# User-configurable paths
# ======================================================================
LABELS_XLSX   = Path(__file__).resolve().parent / "labels.xlsx"
ALL_ROADS_CSV = Path(__file__).resolve().parents[1] / "inspectroad" / "roads_qualified" / "all_roads.csv"
CENTROIDS_DIR = Path(__file__).resolve().parents[1] / "roads" / "all_clips"
OUTPUT_SHP    = Path(__file__).resolve().parent / "predictions_centroids.shp"
# ======================================================================


def load_all_centroids(centroids_dir: Path) -> gpd.GeoDataFrame:
    """Read and concatenate all *centroids.shp files."""
    shp_files = sorted(centroids_dir.glob("*centroids.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No centroid shapefiles in {centroids_dir}")

    target_crs = "EPSG:25830"
    gdfs = []
    for f in shp_files:
        gdf = gpd.read_file(f)
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        gdfs.append(gdf)

    centroids = pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(centroids, crs=target_crs)


def main() -> None:
    # 1 — attribute table from PRED sheet (uses latest PRED_* by default)
    attrs = build_attribute_table(LABELS_XLSX)
    print(f"  Attributes: {len(attrs)} unique identifiers")

    # 2 — merge with all_roads.csv
    roads = pd.read_csv(ALL_ROADS_CSV)
    merged = attrs.merge(roads, on="identifier", how="left")
    print(f"  After roads merge: {len(merged)} rows")

    # 3 — load centroids and merge
    centroids = load_all_centroids(CENTROIDS_DIR)
    print(f"  Centroids loaded: {len(centroids)} features")

    result = centroids.merge(merged, on="identifier", how="inner")
    print(f"  Final features: {len(result)}")

    # 4 — write shapefile
    result.to_file(OUTPUT_SHP)
    print(f"  Shapefile written to {OUTPUT_SHP}")


if __name__ == "__main__":
    main()
