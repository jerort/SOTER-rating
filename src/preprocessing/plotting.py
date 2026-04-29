from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fiona
import geopandas as gpd
from shapely.affinity import rotate
from shapely.geometry import LineString, MultiLineString, Point, Polygon
import matplotlib.pyplot as plt


# ---------------------------
# Geometry
# ---------------------------

class RectangleBuilder:
    """Builds rectangles aligned to a local direction of a line."""
    @staticmethod
    def aligned_rectangle(point: Point, line: LineString | MultiLineString,
                          width: float, height: float) -> Optional[Polygon]:
        """
        Create a rectangle centered at `point`, aligned with the local orientation of `line`.
        width: along the line (longitudinal)
        height: perpendicular to the line (transversal)
        """
        if not isinstance(line, (LineString, MultiLineString)):
            print("[geom] Warning: provided 'line' is not LineString/MultiLineString.")
            return None

        # For MultiLineString, shapely handles project/interpolate on the merged geometry,
        # but GeoPandas recommends union_all() for general multi-part handling.
        # Here, we assume the input is already dissolved when needed.

        # Project point onto the (multi)line to get a curvilinear distance
        t = line.project(point)

        # Use a tiny window around the projected distance to estimate orientation
        eps = min(0.001, max(0.001, 0.001))
        p1d = max(0.0, t - eps)
        p2d = min(line.length, t + eps)

        if math.isclose(p1d, p2d):
            # Fallback: axis-aligned rectangle if no local direction can be estimated
            half_w, half_h = width / 2.0, height / 2.0
            x1, y1 = point.x - half_w, point.y - half_h
            x2, y2 = point.x + half_w, point.y + half_h
            return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        p1 = line.interpolate(p1d)
        p2 = line.interpolate(p2d)

        angle_rad = math.atan2(p2.y - p1.y, p2.x - p1.x)
        half_w, half_h = width / 2.0, height / 2.0

        # Base (unrotated) rectangle centered at point
        x1, y1 = point.x - half_w, point.y - half_h
        x2, y2 = point.x + half_w, point.y + half_h
        rect = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        # Rotate around the center point
        return rotate(rect, math.degrees(angle_rad), origin=point, use_radians=False)


# ---------------------------
# Core processing
# ---------------------------

@dataclass
class GenerationParams:
    longitudinal_size: float = 30.0      # meters (along the line)
    transversal_size: float = 51.0       # meters (perpendicular) – used if transversal_sizes is None
    overlap_fraction: float = 0.25       # fraction in [0, 1)
    transversal_sizes: Optional[List[float]] = None  # if provided, overrides transversal_size with multi-run


class RoadShapefileProcessor:
    """
    Loads a road (multi)line shapefile, generates aligned rectangles
    along it, and outputs rectangles and centroids as shapefiles.
    """
    def __init__(self, input_path: str, utm_fallback_epsg: str = "EPSG:25829") -> None:
        self.input_path = input_path
        self.input_stem = os.path.splitext(os.path.basename(input_path))[0]
        self.utm_fallback_epsg = utm_fallback_epsg
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.original_crs = None
        self._line: Optional[LineString | MultiLineString] = None

    # ----- I/O & CRS -----

    def load(self):
        """Load shapefile robustly with SHAPE_RESTORE_SHX=YES."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        with fiona.Env(SHAPE_RESTORE_SHX="YES"):
            try:
                self.gdf = gpd.read_file(self.input_path)
            except Exception as e:
                raise RuntimeError(f"Failed to read shapefile: {e}") from e

        if self.gdf is None or self.gdf.empty:
            raise RuntimeError("Loaded shapefile is empty.")
        self.original_crs = self.gdf.crs
        return self

    def ensure_projected(self):
        """
        Ensure geometry is in a projected CRS (meters).
        If not, reproject to the given UTM fallback.
        """
        if self.gdf is None:
            raise RuntimeError("Call load() first.")

        if self.gdf.crs is None or not self.gdf.crs.is_projected:
            print(f"[crs] Reprojecting to {self.utm_fallback_epsg} for metric operations...")
            try:
                self.gdf = self.gdf.to_crs(self.utm_fallback_epsg)
            except Exception as e:
                raise RuntimeError(f"Reprojection failed: {e}") from e
        else:
            print("[crs] Layer already projected. No reprojection needed.")
        return self

    # ----- Geometry Prep -----

    def union_line(self) -> LineString | MultiLineString:
        """Dissolve to a single (multi)line geometry for downstream operations."""
        if self.gdf is None:
            raise RuntimeError("Call load() first.")
        self._line = self.gdf.geometry.union_all()
        if self._line is None:
            raise RuntimeError("Could not union line geometry.")
        return self._line

    # ----- Generation -----
    def _points_for_params(self, params: GenerationParams) -> List[Point]:
        if self._line is None:
            self.union_line()
        step = params.longitudinal_size * (1.0 - params.overlap_fraction)
        return self._generate_points(step)

    @staticmethod
    def _size_label(s: float) -> str:
        return str(int(s)) if float(s).is_integer() else f"{s:g}"

    def _generate_points(self, step_distance: float) -> List[Point]:
        if self._line is None:
            raise RuntimeError("Call union_line() first.")
        pts = []
        d = 0.0
        length = float(self._line.length)
        step = float(step_distance)
        if step <= 0:
            raise ValueError("step_distance must be > 0")

        while d < length:
            pts.append(self._line.interpolate(d))
            d += step
        # Optionally ensure last point at the end of the line
        # pts.append(self._line.interpolate(L))
        return pts

    def make_rectangles(self, params: GenerationParams) -> gpd.GeoDataFrame:
        if self.gdf is None:
            raise RuntimeError("Call load() first.")
        pts = self._points_for_params(params)

        rectangles, ids = [], []
        counter = 1
        for p in pts:
            poly = RectangleBuilder.aligned_rectangle(
                p, self._line, params.longitudinal_size, params.transversal_size
            )
            if poly is not None:
                rectangles.append(poly)
                ids.append(f"{self.input_stem}__{counter}")
                counter += 1

        if not rectangles:
            raise RuntimeError("No rectangles were generated. Check inputs/parameters.")
        return gpd.GeoDataFrame({"identifier": ids, "geometry": rectangles}, crs=self.gdf.crs)

    @staticmethod
    def make_centroids(rectangles_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Return a new GDF with centroids of given rectangles and same identifiers."""
        if rectangles_gdf.empty:
            raise RuntimeError("Rectangles GeoDataFrame is empty.")
        out = rectangles_gdf.copy()  # keeps 'identifier'
        out["geometry"] = rectangles_gdf.centroid
        return out

    def generate_rectangles(
            self,
            params: GenerationParams,
            include_size_in_identifier: bool = True,
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Unified generator:
          - If params.transversal_sizes is None/empty -> returns {"default": single_gdf}
          - Else -> returns {"<size_label>": gdf_for_size, ...}
        Identifiers are stable; if include_size_in_identifier is True in the multi case,
        identifiers become '<stem>__<i>__<size>m'.
        """
        if self.gdf is None:
            raise RuntimeError("Call load() first.")

        sizes = params.transversal_sizes or []
        if not sizes:
            # Single-size path (keep your current behavior)
            return {"default": self.make_rectangles(params)}

        pts = self._points_for_params(params)
        out: Dict[str, gpd.GeoDataFrame] = {}

        for tsize in sizes:
            rectangles, ids = [], []
            counter = 1
            for p in pts:
                poly = RectangleBuilder.aligned_rectangle(
                    p, self._line, params.longitudinal_size, tsize
                )
                if poly is not None:
                    rectangles.append(poly)
                    if include_size_in_identifier:
                        ids.append(f"{self.input_stem}__{counter}__{self._size_label(tsize)}m")
                    else:
                        ids.append(f"{self.input_stem}__{counter}")
                    counter += 1

            if not rectangles:
                raise RuntimeError(f"No rectangles generated for transversal_size={tsize}.")

            gdf_rectangles = gpd.GeoDataFrame(
                {"identifier": ids, "geometry": rectangles}, crs=self.gdf.crs
            )
            out[self._size_label(tsize)] = gdf_rectangles

        return out

    # ----- Export -----

    def _maybe_reproject_back(self, gdf_out: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Reproject output back to original CRS if it existed."""
        if self.original_crs is not None:
            return gdf_out.to_crs(self.original_crs)
        print("[crs] No original CRS. Leaving output in projected CRS.")
        return gdf_out

    @staticmethod
    def _shp_sidecars(shp_path: str) -> List[str]:
        base, _ = os.path.splitext(shp_path)
        # Common shapefile sidecars
        return [f"{base}{ext}" for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]]

    def save_shapefile(self, gdf_out: gpd.GeoDataFrame, out_path: str) -> Tuple[str, List[str]]:
        """
        Save GeoDataFrame to ESRI Shapefile and optionally trigger Colab download.
        Returns (shp_path, list_of_written_paths_that_exist).
        """
        gdf_final = self._maybe_reproject_back(gdf_out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        try:
            gdf_final.to_file(out_path, driver="ESRI Shapefile")
        except Exception as e:
            raise RuntimeError(f"Failed saving shapefile to {out_path}: {e}") from e

        paths = self._shp_sidecars(out_path)
        existing = [p for p in paths if os.path.exists(p)]

        return out_path, existing

    def save_rectangles_and_centroids(
            self,
            params: GenerationParams,
            outdir: str,
            rectangles_template: str = "rectangles_{size}.shp",  # {size} is "default" or "51m"
            centroids_template: str = "centroids.shp",
            include_size_in_identifier: bool = True,
    ) -> Dict[str, Tuple[str, str]]:
        """
        Generate rectangles (single or multi), create centroids, and save everything.
        Returns a dict: {size_key: (rectangles_shp_path, centroids_shp_path)}
        """
        os.makedirs(outdir, exist_ok=True)
        generated = self.generate_rectangles(params, include_size_in_identifier)
        saved: Dict[str, Tuple[str, str]] = {}

        for size_key, rectangles_gdf in generated.items():
            size_suffix = f"{size_key}m" if size_key != "default" else "default"
            rectangles_path = os.path.join(outdir, self.input_stem + "__" + rectangles_template.format(size=size_suffix))
            centroids_path = os.path.join(outdir, self.input_stem + "__" + centroids_template)

            self.save_shapefile(rectangles_gdf, rectangles_path)
            centroids_gdf = self.make_centroids(rectangles_gdf)
            self.save_shapefile(centroids_gdf, centroids_path)

            saved[size_key] = (rectangles_path, centroids_path)

        return saved


class ResultChecker:
    """
    Helper to overlay the original road geometry with the generated rectangles
    in a chosen UTM CRS and save/show a quick-look plot.
    """
    @staticmethod
    def plot_overlay_utm(
        original_shp: str,
        rectangles_shp: str,
        utm_epsg: str = "EPSG:25829",
        out_path: Optional[str] = "check_overlay.png",
        show: bool = False,
        include_centroids_shp: Optional[str] = None,
    ) -> str:
        """
        Reads the original road shapefile and the rectangles shapefile, reprojects
        both to `utm_epsg`, and plots them overlaid. Optionally include centroids.

        Returns the path to the saved PNG (if out_path is not None).
        """
        if not os.path.exists(original_shp):
            raise FileNotFoundError(f"Original shapefile not found: {original_shp}")
        if not os.path.exists(rectangles_shp):
            raise FileNotFoundError(f"Rectangles shapefile not found: {rectangles_shp}")

        with fiona.Env(SHAPE_RESTORE_SHX="YES"):
            gdf_orig = gpd.read_file(original_shp)
            gdf_rect = gpd.read_file(rectangles_shp)
            gdf_cent = None
            if include_centroids_shp:
                if not os.path.exists(include_centroids_shp):
                    print(f"[checker] Centroids not found: {include_centroids_shp}")
                else:
                    gdf_cent = gpd.read_file(include_centroids_shp)

        # Reproject to UTM if needed
        def to_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            if gdf.crs is None or not gdf.crs.is_projected or gdf.crs.to_string() != utm_epsg:
                return gdf.to_crs(utm_epsg)
            return gdf

        gdf_orig_utm = to_utm(gdf_orig)
        gdf_rect_utm = to_utm(gdf_rect)
        gdf_cent_utm = to_utm(gdf_cent) if gdf_cent is not None else None

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        gdf_orig_utm.plot(ax=ax, linewidth=1.2, alpha=0.7, label="Road (UTM)")
        gdf_rect_utm.boundary.plot(ax=ax, linewidth=1.0, label="Rectangles (UTM)")
        if gdf_cent_utm is not None and not gdf_cent_utm.empty:
            gdf_cent_utm.plot(ax=ax, markersize=10, label="Centroids (UTM)")

        ax.set_aspect("equal")
        ax.set_title(f"Overlay (UTM: {utm_epsg})")
        ax.legend()

        saved = ""
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            saved = out_path
            print(f"[checker] Saved overlay: {saved}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return saved
