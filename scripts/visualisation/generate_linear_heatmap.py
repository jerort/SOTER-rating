"""
Generate a linear-heatmap shapefile for QGIS visualisation.

For each road the layout LineString is split at the midpoints between
consecutive centroids.  Each resulting segment inherits the attributes of
its centroid (estimate, label, subset, road-quality columns …).

The first segment starts at the beginning of the layout line and the last
segment ends at its end.

Output
------
A single shapefile of ~4975 short LineString features (one per centroid),
ready to be styled in QGIS with Graduated symbology on ``estimate``.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from src.labels import build_attribute_table, load_calibration_probs


# ======================================================================
# User-configurable paths
# ======================================================================
LABELS_XLSX   = Path(__file__).resolve().parent / "labels.xlsx"
ALL_ROADS_CSV = Path(__file__).resolve().parents[1] / "inspectroad" / "roads_qualified" / "all_roads.csv"
CENTROIDS_DIR = Path(__file__).resolve().parents[1] / "roads" / "all_clips"
LAYOUTS_DIR   = Path(__file__).resolve().parents[1] / "roads" / "layouts"
OUTPUT_SHP    = Path(__file__).resolve().parent / "predictions_heatmap.shp"
TARGET_CRS    = "EPSG:25830"
# Calibration experiment folder (set to None to skip probability columns)
CALIBRATION_DIR = Path(r"D:\SOTER\SOTER\scripts\rating\experiments\experiment_20260308_142636_CET")
# ======================================================================


def substring_line(line: LineString, start: float, end: float) -> LineString:
    """Extract the sub-LineString between two linear distances along *line*.

    Handles Z coordinates if present.  Returns a 2-point line when the
    sub-section contains no intermediate vertices.
    """
    coords = []
    d_acc = 0.0  # accumulated distance along the line

    pts = list(line.coords)
    has_z = len(pts[0]) == 3

    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        seg_len = LineString([p0, p1]).length
        d_next = d_acc + seg_len

        # --- start interpolation ------------------------------------------
        if not coords and d_next >= start:
            frac = (start - d_acc) / seg_len if seg_len else 0.0
            if has_z:
                coords.append((
                    p0[0] + frac * (p1[0] - p0[0]),
                    p0[1] + frac * (p1[1] - p0[1]),
                    p0[2] + frac * (p1[2] - p0[2]),
                ))
            else:
                coords.append((
                    p0[0] + frac * (p1[0] - p0[0]),
                    p0[1] + frac * (p1[1] - p0[1]),
                ))

        # --- end interpolation --------------------------------------------
        if coords and d_next >= end:
            frac = (end - d_acc) / seg_len if seg_len else 0.0
            if has_z:
                coords.append((
                    p0[0] + frac * (p1[0] - p0[0]),
                    p0[1] + frac * (p1[1] - p0[1]),
                    p0[2] + frac * (p1[2] - p0[2]),
                ))
            else:
                coords.append((
                    p0[0] + frac * (p1[0] - p0[0]),
                    p0[1] + frac * (p1[1] - p0[1]),
                ))
            break

        # --- intermediate vertex inside [start, end] ----------------------
        if coords:
            coords.append(p1)

        d_acc = d_next

    if len(coords) < 2:
        # Degenerate: start ≈ end → return a tiny segment around that point
        pt = line.interpolate(start)
        return LineString([pt, pt])
    return LineString(coords)


def split_layout_at_midpoints(
    layout_line: LineString,
    centroid_positions: list[float],
) -> list[LineString]:
    """Split *layout_line* so each centroid sits at the midpoint of its segment.

    Cut points are the midpoints between consecutive centroid projections.
    The first segment starts at 0 and the last ends at the line length.
    """
    n = len(centroid_positions)
    line_len = layout_line.length

    # Compute cut distances (midpoints between consecutive centroids)
    cuts = [0.0]
    for i in range(n - 1):
        cuts.append((centroid_positions[i] + centroid_positions[i + 1]) / 2.0)
    cuts.append(line_len)

    segments = []
    for i in range(n):
        seg = substring_line(layout_line, cuts[i], cuts[i + 1])
        segments.append(seg)
    return segments


def road_name_from_identifier(identifier: str) -> str:
    """'A-490__123' → 'A-490'."""
    return identifier.rsplit("__", 1)[0]


def main() -> None:
    # 1 — attribute table from PRED sheet (uses latest PRED_* by default)
    attrs = build_attribute_table(LABELS_XLSX)
    print(f"  Attributes: {len(attrs)} unique identifiers")

    # 2 — merge with all_roads.csv
    roads = pd.read_csv(ALL_ROADS_CSV)
    merged = attrs.merge(roads, on="identifier", how="left")
    print(f"  After roads merge: {len(merged)} rows")

    # 2b — merge calibration probabilities if available
    if CALIBRATION_DIR is not None:
        cal_probs = load_calibration_probs(Path(CALIBRATION_DIR))
        if cal_probs is not None:
            merged = merged.merge(cal_probs, on="identifier", how="left")
            print(f"  Merged calibration probs: {len(cal_probs)} samples, "
                  f"cols: {[c for c in cal_probs.columns if c != 'identifier']}")
        else:
            print("  WARNING: Could not load calibration probs "
                  "(missing identifier column? Re-run calibrate.py)")

    # 3 — load centroids (with road grouping) and layouts
    centroid_files = sorted(CENTROIDS_DIR.glob("*centroids.shp"))
    layout_files = {f.stem: f for f in LAYOUTS_DIR.glob("*.shp")}

    all_segments = []

    for cf in centroid_files:
        road_name = cf.stem.replace("__centroids", "")
        lf = layout_files.get(road_name)
        if lf is None:
            print(f"  WARNING: no layout for {road_name}, skipping")
            continue

        # Load and reproject if needed
        gdf_c = gpd.read_file(cf)
        if gdf_c.crs != TARGET_CRS:
            gdf_c = gdf_c.to_crs(TARGET_CRS)

        gdf_l = gpd.read_file(lf)
        if gdf_l.crs != TARGET_CRS:
            gdf_l = gdf_l.to_crs(TARGET_CRS)

        layout_line = gdf_l.geometry.iloc[0]

        # Filter to centroids that have prediction data
        road_attrs = merged[merged["identifier"].isin(gdf_c["identifier"])]
        # Keep centroid order (already sorted along the line)
        road_attrs = road_attrs.set_index("identifier").reindex(gdf_c["identifier"]).dropna(subset=["estimate"]).reset_index()

        if road_attrs.empty:
            continue

        # Project centroids onto layout line
        centroid_geoms = gdf_c.set_index("identifier").reindex(road_attrs["identifier"])
        positions = [layout_line.project(pt) for pt in centroid_geoms.geometry]

        # Split layout at midpoints
        segments = split_layout_at_midpoints(layout_line, positions)

        # Boundary values for the Interpolated Line renderer in QGIS.
        # See: https://www.youtube.com/watch?v=FtnD-bXfV58
        # Each pair (raw_start/raw_end, est_start/est_end) holds the interpolated
        # value at the left and right boundary of each segment (midpoints between
        # consecutive centroids).  Adjacent segments share the same boundary value
        # → smooth colour gradient with no visible step.
        def make_boundaries(vals):
            n = len(vals)
            b = [vals[0]]
            for i in range(n - 1):
                b.append((vals[i] + vals[i + 1]) / 2.0)
            b.append(vals[-1])
            return b

        raw_bounds = make_boundaries(road_attrs["raw_est"].tolist())
        est_bounds = make_boundaries(road_attrs["estimate"].tolist())

        # Confidence boundaries (if calibration columns present)
        conf_bounds = {}
        for prefix in ("gau", "ord"):
            col = f"{prefix}_conf"
            if col in road_attrs.columns and road_attrs[col].notna().any():
                vals = road_attrs[col].fillna(0.2).tolist()  # 0.2 = uniform 1/5
                conf_bounds[prefix] = make_boundaries(vals)

        for i, (seg, (_, row)) in enumerate(zip(segments, road_attrs.iterrows())):
            rec = row.to_dict()
            rec["geometry"]   = seg
            rec["road"]       = road_name
            rec["raw_start"]  = raw_bounds[i]
            rec["raw_end"]    = raw_bounds[i + 1]
            rec["est_start"]  = est_bounds[i]
            rec["est_end"]    = est_bounds[i + 1]
            for prefix in ("gau", "ord"):
                if prefix in conf_bounds:
                    rec[f"{prefix}_cs"] = conf_bounds[prefix][i]
                    rec[f"{prefix}_ce"] = conf_bounds[prefix][i + 1]
            all_segments.append(rec)

        print(f"  {road_name}: {len(segments)} segments")

    # 4 — assemble and write
    result = gpd.GeoDataFrame(all_segments, crs=TARGET_CRS)
    print(f"\n  Total segments: {len(result)}")
    result.to_file(OUTPUT_SHP)
    print(f"  Shapefile written to {OUTPUT_SHP}")


if __name__ == "__main__":
    main()
