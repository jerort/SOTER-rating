from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple
import traceback
from src.preprocessing.plotting import  GenerationParams, RoadShapefileProcessor

# --- CONFIGURE THESE VARIABLES DIRECTLY ---
input_path = r"layouts"
output_path = r"clips"

params = GenerationParams(longitudinal_size=30, overlap_fraction=0.25, transversal_sizes=[8.0, 51.0])

# --- RUN ---
def process_one_shp(shp_path: Path, outdir: Path) -> Dict[str, Tuple[str, str]]:
    proc = RoadShapefileProcessor(str(shp_path), utm_fallback_epsg="EPSG:25829")

    proc.load().ensure_projected().union_line()
    return proc.save_rectangles_and_centroids(params=params, outdir=output_path, include_size_in_identifier=False)

Path(output_path).mkdir(parents=True, exist_ok=True)

# Find .shp files (skip likely outputs to avoid reprocessing)
shp_files = [p for p in Path(input_path).glob("*.shp")]
if not shp_files:
    raise RuntimeError(f"[info] No .shp files found in: {input_path}")

print(f"[info] Found {len(shp_files)} shp files. Processing with ThreadPoolExecutor...")

results: Dict[Path, Dict[str, Tuple[str, str]]] = {}
errors: Dict[Path, str] = {}

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(process_one_shp, shp, Path(output_path)): shp for shp in shp_files}
    for fut in as_completed(futures):
        shp = futures[fut]
        try:
            saved = fut.result()
            results[shp] = saved
            size_keys = ", ".join(sorted(saved.keys()))
            print(f"[ok] {shp.name}: sizes -> {size_keys}")
        except Exception as e:
            tb = traceback.format_exc()
            errors[shp] = tb
            print(f"[ERR] {shp.name}: {e}\n{tb}")

# Summary
print("\n==== SUMMARY ====")
for shp, saved in results.items():
    print(f"\n{shp.name}")
    for size_key, (rects_path, cents_path) in saved.items():
        print(f"  - size={size_key:>7}  rects={Path(rects_path).name}  cents={Path(cents_path).name}")
if errors:
    print("\n==== ERRORS ====")
    for shp, tb in errors.items():
        print(f"\n{shp.name}\n{tb}")
