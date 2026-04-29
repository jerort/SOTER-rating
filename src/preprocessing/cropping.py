import os
import math
from abc import ABC, abstractmethod
import cv2
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import csv


class ImageCropperBase(ABC):
    def __init__(self, source_path, target_path, max_workers=8):
        self.source_path = source_path
        self.target_path = target_path
        self.max_workers = max_workers
        self.supported_extensions = None
        os.makedirs(self.target_path, exist_ok=True)

    @abstractmethod
    def process_image(self, filepath, filename, size):
        pass

    def is_supported(self, filename):
        return os.path.splitext(filename)[1].lower() in self.supported_extensions

class ConventionalImageCropper(ImageCropperBase):

    def __init__(self, source_path, target_path, max_workers=8):
        super().__init__(source_path, target_path, max_workers)
        self.supported_extensions = {".png", ".jpg", ".jpeg"}

    def _crop_and_save(self, image, base_name, ext, x, y, row, col, size):
        crop = image[y:y + size, x:x + size]
        crop_name = f"{base_name}__R{row}__C{col}{ext}"
        out_path = os.path.join(self.target_path, crop_name)
        cv2.imwrite(out_path, crop)

    def process_image(self, filepath, filename, size):
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[Invalid Image] Skipping: {filename}")
            return

        height, width = image.shape[:2]
        base_name, ext = os.path.splitext(filename)

        tasks = []
        for row, y in enumerate(range(0, height - size + 1, size)):
            for col, x in enumerate(range(0, width - size + 1, size)):
                tasks.append((image, base_name, ext, x, y, row, col, size))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(lambda args: self._crop_and_save(*args), tasks)


class GeoTIFFImageCropper(ImageCropperBase):
    def __init__(self, source_path, target_path, shapefile_path=None, max_workers=8):
        super().__init__(source_path, target_path, max_workers)
        self.supported_extensions = {".tif", ".tiff"}
        self.shapefile_path = shapefile_path

    def _crop_and_save(self, data, meta, transform, base_name, row, col, size):
        out_name = f"{base_name}__R{row // size}__C{col // size}.tif"
        out_path = os.path.join(self.target_path, out_name)

        meta = meta.copy()
        meta.update({
            "height": size,
            "width": size,
            "transform": transform
        })
        with rasterio.Env(GDAL_PAM_ENABLED=False):
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(data)

    def process_image(self, filepath, filename, size):
        if self.shapefile_path is None:
            raise ValueError("shapefile_path must be provided for polygon-based filtering.")

        gdf = gpd.read_file(self.shapefile_path)
        with rasterio.open(filepath) as raster:
            if raster.crs != gdf.crs:
                gdf = gdf.to_crs(raster.crs)
            polygon = gdf.unary_union  # Merge all geometries

            width, height = raster.width, raster.height
            base_name, _ = os.path.splitext(filename)
            meta = raster.meta.copy()
            tasks = []
            included = 0
            for row in range(0, height - size + 1, size):
                for col in range(0, width - size + 1, size):
                    window = Window(col, row, size, size)
                    transform = raster.window_transform(window)
                    bounds = rasterio.transform.array_bounds(size, size, transform)
                    crop_poly = box(*bounds)

                    if polygon.contains(crop_poly):
                        data = raster.read(window=window)
                        tasks.append((data, meta, transform, base_name, row, col, size))
                        included += 1

            if not tasks:
                print(f"[No Tiles] Skipped: {filename} — No tiles matched the polygon")
                return

            print(f"[{filename}] Accepted {included} tile(s) inside polygon.")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                executor.map(lambda args: self._crop_and_save(*args), tasks)

class CentralRoadCropper(ImageCropperBase):
    def __init__(self, source_path, target_path, max_workers=8):
        super().__init__(source_path, target_path, max_workers)
        self.supported_extensions = {".png"}

    @staticmethod
    def _crop_and_save(image, filename, out_dir, x0, x1):
        crop = image[:, x0:x1]
        out_path = os.path.join(out_dir, filename)
        cv2.imwrite(str(out_path), crop)

    def process_image(self, filepath, filename, target_width):
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[Invalid Image] Skipping: {filename}")
            return

        height, width = image.shape[:2]
        _, ext = os.path.splitext(filename)

        if not isinstance(target_width, int) or target_width <= 0:
            print(f"[Invalid Width] Skipping: {filename} — target_width must be a positive int")
            return

        if target_width > width:
            print(f"[Too Wide] Skipping: {filename} — target_width ({target_width}) > image width ({width})")
            return

        x0 = (width - target_width) // 2
        x1 = x0 + target_width

        self._crop_and_save(image, filename, self.target_path, x0, x1)

class RectanglePolygonCropper(ImageCropperBase):

    def __init__(self, source_path, target_path, shapefile_path=None, max_workers=8):
        super().__init__(source_path, target_path, max_workers)
        self.supported_extensions = {".tif", ".tiff"}
        self.shapefile_path = shapefile_path

    def process_image(self, filepath: str, filename: str, size=None):
        gdf, total_tiles = self._prepare_gdf()
        if gdf.empty:
            raise ValueError(f"Empty/invalid shapefile: {self.shapefile_path}")
        if "identifier" not in gdf.columns:
            raise ValueError("Shapefile must contain an 'identifier' attribute.")

        base_name, _ = os.path.splitext(filename)

        # we're assuming a 0,3 m/pixel resolution
            # 27 pixels for 8m-tiles and 171 pixels for 51m-tiles
            # 101 pixels for 30m-tiles
        half_pixel_width = 85
        half_pixel_height = 50
        tile_size, half_tile = 500, 250

        with rasterio.open(filepath) as raster:
            if raster.crs is None:
                raise ValueError(f"{filename}: raster has no CRS; cannot rotate with reproject.")
            if gdf.crs and raster.crs and gdf.crs != raster.crs:
                gdf = gdf.to_crs(raster.crs)
            gdf = gdf[gdf.intersects(box(*raster.bounds))].copy()
            if gdf.empty:
                print(f"[Warning] No polygons intersect raster in {self.shapefile_path}")
                return total_tiles

            csv_path = os.path.join(self.target_path, f"angles.csv")
            csv_exists = os.path.exists(csv_path)

            with open(csv_path, "a", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                if not csv_exists:
                    # header written only once when file is first created
                    csv_writer.writerow(["patch_name", "centroid_x", "centroid_y", "angle"])

                for _, row in gdf.iterrows():
                    rectangle = row.geometry
                    if rectangle is None or rectangle.is_empty or row["identifier"] is None:
                        continue

                    # 1) 500×500 window around centroid (pixel coords, clamped)
                    col, rowp = (~raster.transform) * (rectangle.centroid.x, rectangle.centroid.y)
                    col_off = max(0, min(int(round(col)) - half_tile, raster.width - tile_size))
                    row_off = max(0, min(int(round(rowp)) - half_tile, raster.height - tile_size))
                    win = Window(col_off, row_off, min(tile_size, raster.width), min(tile_size, raster.height))
                    tile = raster.read(window=win)  # (bands, H, W)
                    win_tx = raster.window_transform(win)
                    H, W = tile.shape[1], tile.shape[2]

                    # 2) rotate about window center (map coords)
                    angle = self._compute_rotation_angle(rectangle)
                    cx_map, cy_map = win_tx * (W / 2.0, H / 2.0)
                    rot_raster_tx = Affine.translation(cx_map, cy_map) * Affine.rotation(angle) * Affine.translation(
                        -cx_map, -cy_map) * win_tx

                    # 3) reproject directly into final crop of size (2*half+1), centered on same pivot
                    out_w, out_h = 2 * half_pixel_width + 1, 2 * half_pixel_height + 1
                    px_w, px_h = win_tx.a, abs(win_tx.e)
                    xmin = cx_map - half_pixel_width * px_w
                    ymax = cy_map + half_pixel_height * px_h
                    dst_tx = Affine.translation(xmin, ymax) * Affine.scale(px_w, -px_h)

                    final = np.zeros((tile.shape[0], out_h, out_w), dtype=tile.dtype)
                    for b in range(tile.shape[0]):
                        reproject(
                            source=tile[b], destination=final[b],
                            src_transform=rot_raster_tx, src_crs=raster.crs,
                            dst_transform=dst_tx, dst_crs=raster.crs,
                            resampling=Resampling.bilinear
                        )

                    # 4) write PNG (no CRS/transform) — 1 band or RGB
                    # Before RGB8 conversion, check for no_data points across all bands
                    if raster.nodata is not None:
                        no_data_mask = np.all(final == raster.nodata, axis=0)
                    else:
                        no_data_mask = np.all(final == 0, axis=0)

                    if np.any(no_data_mask):
                        print(f"[Skipping] {row['identifier']}: Contains pixels with no_data in all bands")
                        continue

                    rgb8 = self.to_rgb8(final, nodata=raster.nodata)  # final is your (bands,H,W) result
                    out_png = os.path.join(self.target_path, f"{row['identifier']}.png")
                    with rasterio.open(out_png, "w", driver="PNG", height=rgb8.shape[1], width=rgb8.shape[2],
                                       count=3, dtype="uint8") as dst:
                        dst.write(rgb8)

                    # Log patch name, centroid (map coords in raster CRS) and rotation angle
                    csv_writer.writerow([row["identifier"], cx_map, cy_map, angle])

            return total_tiles

    def _prepare_gdf(self):
        gdf = gpd.read_file(self.shapefile_path)
        if gdf.empty:
            return gdf, 0
        def rename_identifiers(identifier):
            return identifier.split("__")[1]
        total_tiles = gdf["identifier"].copy().apply(rename_identifiers).astype(int).max()
        gdf = gdf[~gdf.geometry.is_empty].copy()
        gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf.explode(index_parts=False, ignore_index=True)
        return gdf[gdf.geometry.geom_type == "Polygon"].copy(), total_tiles

    @staticmethod
    def _compute_rotation_angle(rectangle):
        c = list(rectangle.exterior.coords)[:4]
        dx, dy = min([((c[i][0] - c[i - 1][0]), (c[i][1] - c[i - 1][1])) for i in range(4)],
                     key=lambda e: e[0] ** 2 + e[1] ** 2)
        return (90.0 - math.degrees(math.atan2(dy, dx)) + 180.0) % 360.0 - 180.0

    @staticmethod
    def to_rgb8(data, nodata=None):
        # pick 3 bands (replicate if fewer)
        bands = data[:3] if data.shape[0] >= 3 else np.repeat(data[:1], 3, axis=0)
        rgb = np.empty((3, bands.shape[1], bands.shape[2]), dtype=np.uint8)

        for i in range(3):
            b = bands[i].astype(np.float32, copy=False)
            if nodata is not None:
                mask = (b == nodata)
                b = np.where(mask, np.nan, b)

            # decide scaling range
            if np.isfinite(b).any():
                if np.issubdtype(bands.dtype, np.floating) and (np.nanmin(b) >= 0.0) and (np.nanmax(b) <= 1.0):
                    lo, hi = 0.0, 1.0
                else:
                    lo, hi = np.nanmin(b), np.nanmax(b)
                    if hi == lo: hi = lo + 1.0
                s = (255.0 * (b - lo) / (hi - lo))
                s = np.clip(s, 0, 255)
            else:
                s = np.zeros_like(b, dtype=np.float32)

            if nodata is not None:
                s = np.where(np.isnan(b), 0.0, s)

            rgb[i] = s.astype(np.uint8)

        return rgb  # (3, H, W)