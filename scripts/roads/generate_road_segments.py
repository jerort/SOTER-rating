from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import traceback
from src.preprocessing.cropping import RectanglePolygonCropper

# --- CONFIGURE THESE VARIABLES DIRECTLY ---
shapefiles_folder = r""  # Folder containing multiple .shp files
tiff_image_path = r""  # Single .tif image to crop
output_path = r""  # Output folder for cropped images


# --- TASK FUNCTION ---
def process_one_shp(shp_path: Path, tiff_path: Path, output_dir: Path):
    """
    Process a single shapefile with the given TIFF image.
    Returns the number of crops produced.
    """
    try:
        # Create output subfolder named after the shapefile (without extension)
        shp_output_dir = output_dir / shp_path.stem
        shp_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the cropper
        cropper = RectanglePolygonCropper(
            source_path=str(tiff_path.parent),
            target_path=str(shp_output_dir),
            shapefile_path=str(shp_path),
            max_workers=1  # Single worker per shapefile to avoid conflicts
        )
        total_tiles = cropper.process_image(str(tiff_path), tiff_path.name)

        return total_tiles, len(list(shp_output_dir.glob("*.png")))

    except Exception as e:
        raise RuntimeError(f"Error processing {shp_path.name}: {str(e)}")


shapefiles_dir = Path(shapefiles_folder)
tiff_file = Path(tiff_image_path)
output_dir = Path(output_path)

if not shapefiles_dir.exists():
    raise FileNotFoundError(f"Shapefiles folder not found: {shapefiles_folder}")

if not tiff_file.exists():
    raise FileNotFoundError(f"TIFF image not found: {tiff_image_path}")

if not tiff_file.suffix.lower() in {'.tif', '.tiff'}:
    raise ValueError(f"Input must be a .tif/.tiff file, got: {tiff_file.suffix}")

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Find .shp files
shp_files = list(shapefiles_dir.glob("*.shp"))
if not shp_files:
    raise RuntimeError(f"No .shp files found in: {shapefiles_folder}")

print(f"[info] Found {len(shp_files)} shp files to process with {tiff_file.name}")
print(f"[info] Output directory: {output_path}")

# Process files with ThreadPoolExecutor
errors = {}

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(process_one_shp, shp, tiff_file, output_dir): shp
        for shp in shp_files
    }

    for future in as_completed(futures):
        shp = futures[future]
        try:
            total, crop_count = future.result()
            print(f"[ok] {shp.name}: produced {crop_count} crops from {total} tiles")
        except Exception as e:
            tb = traceback.format_exc()
            errors[shp] = tb
            print(f"[ERR] {shp.name}: {e}")

if errors:
    print("\n==== ERRORS ====")
    for shp, tb in errors.items():
        print(f"\n{shp.name}:")
        print(tb)