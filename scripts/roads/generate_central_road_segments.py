import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from src.preprocessing.cropping import CentralRoadCropper


def process_dataset(cropper, source_path, target_width, max_workers=8):
    source_dir = Path(source_path)
    files = [p.relative_to(source_dir) for p in source_dir.rglob("*.png") if p.is_file()]
    tasks = []

    for filename in files:
        # filename is now a relative Path (may include subfolders)
        filepath = str(source_dir / filename)
        if str(filename.parent):
            os.makedirs(os.path.join(cropper.target_path, str(filename.parent)), exist_ok=True)

        if cropper.is_supported(filename):
            tasks.append((filepath, filename))
        else:
            print(f"[Unsupported File] Skipping: {filename}")

    if not tasks:
        print("[No PNGs] Nothing to process.")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda args: cropper.process_image(args[0], args[1], target_width), tasks)


# ==== Instantiate and run ====
input_path  = r"D:\datasets\SOTER\road_crops"
output_path = r"D:\datasets\SOTER\road_crops_central"

# Central vertical crop target width (pixels)
# we're assuming a 0,3 m/pixel resolution
    # 27 pixels for 8m-tiles and 171 pixels for 51m-tiles
    # 101 pixels for 30m-tiles
tiles_8m_width = 27  # <- adjust to your desired width

center_cropper = CentralRoadCropper(input_path, output_path, max_workers=8)

process_dataset(center_cropper, input_path, tiles_8m_width)
