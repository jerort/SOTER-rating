"""
Generate train/val/test CSVs from labels.xlsx.

Reproducibility note
--------------------
This script is non-deterministic: image-path resolution runs through a
ThreadPoolExecutor, so the row order of the intermediate dataframe depends
on thread scheduling, and `train_test_split(..., random_state=RANDOM_SEED)`
picks different rows across runs even with a fixed seed. The CSVs shipped
in scripts/rating/datasets/ are a captured snapshot of one historical run
and are paired with the experiments under scripts/rating/experiments/, so
they are kept verbatim rather than regenerated.

Re-running this script does not reproduce those CSVs. Beyond the partition
shift, the published snapshot also predates a simplification of this script:
an earlier version set aside road A-495R as a candidate holdout, sampled
~100 of its images per a target class distribution, and discarded the rest.
That separation has since been removed — A-495R is now part of the main
pool — so a fresh run produces ~19 more rows than the published CSVs,
concentrated in the classes where the old per-class quota was below the
available pool.
"""

import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# Configuration
# =========================
LABELS_XLSX = r"D:\SOTER\SOTER\scripts\rating\data_inspection\labels.xlsx"
OUTPUT_DIR = r"D:\SOTER\SOTER\scripts\rating\datasets"  # SET THIS: Path to output directory (e.g., "D:\\datasets\\SOTER\\Carreteras\\HD15_8m")
IMAGE_ROOT = r"C:\data\Carreteras\HD15_8m"  # SET THIS: Path to folder with 4 region subfolders (e.g., "D:\\datasets\\SOTER\\Carreteras\\HD15_8m")

# Train/val/test split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42


def find_all_image_paths(identifier: str, image_root: str) -> list:
    if not image_root:
        return [f"{identifier}.png"]

    image_root_path = Path(image_root)
    if not image_root_path.exists():
        raise FileNotFoundError(f"Image root directory does not exist: {image_root}")

    filename = f"{identifier}.png"
    matches = list(image_root_path.rglob(filename))

    if not matches:
        raise FileNotFoundError(f"Image file not found: {filename}")

    return [str(match.relative_to(image_root_path)) for match in matches]


def process_single_identifier(identifier: str, _rating: int, image_root: str):
    try:
        filepaths = find_all_image_paths(identifier, image_root)
        return [(fp, _rating) for fp in filepaths], None
    except FileNotFoundError as e:
        return None, str(e)


def map_identifiers_to_paths(dataset: pd.DataFrame, image_root: str, set_name: str):
    print(f"   Processing {set_name} set with {len(dataset)} labels...")

    results = []
    missing_count = 0
    duplicate_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                process_single_identifier,
                row['Identifier'],
                row['Etiqueta'],
                image_root
            ): row['Identifier']
            for _, row in dataset.iterrows()
        }

        for future in as_completed(futures):
            identifier = futures[future]
            try:
                file_results, error = future.result()

                if error:
                    print(f"      Warning: {error}")
                    missing_count += 1
                    continue

                results.extend(file_results)

                if len(file_results) > 1:
                    duplicate_count += len(file_results) - 1

            except Exception as e:
                print(f"      Error processing {identifier}: {e}")
                missing_count += 1

    if duplicate_count > 0:
        print(f"      Found {duplicate_count} duplicate identifiers (images in multiple regions)")
    if missing_count > 0:
        print(f"      Warning: {missing_count} images not found")
    print(f"      Total samples after mapping: {len(results)}")

    return pd.DataFrame(results, columns=['filepath', 'rating'])


# Validate configuration
if not OUTPUT_DIR:
    raise ValueError("OUTPUT_DIR must be set in the script configuration")
if not IMAGE_ROOT:
    raise ValueError("IMAGE_ROOT must be set in the script configuration")

print("=" * 60)
print("Rating Dataset CSV Generator")
print("=" * 60)

# Load labels
print(f"\n1. Loading labels from {LABELS_XLSX}...")
df = pd.read_excel(LABELS_XLSX)
print(f"   Loaded {len(df)} samples")

# Round fractional ratings
print("\n2. Rounding fractional ratings to nearest integer...")
df['Etiqueta'] = df['Etiqueta'].round().astype(int)

# Remove any invalid ratings
df = df[df['Etiqueta'].isin([1, 2, 3, 4, 5])]
print(f"   Valid samples: {len(df)}")
print("\n   Class distribution:")
class_dist = df['Etiqueta'].value_counts().sort_index()
for rating, count in class_dist.items():
    pct = count / len(df) * 100
    print(f"   Class {rating}: {count:4d} ({pct:.1f}%)")

# Map identifiers to file paths
print("\n3. Mapping identifiers to file paths...")
print("   Note: Duplicate identifiers (same ID in multiple regions) will create separate samples")
df_all = map_identifiers_to_paths(df, IMAGE_ROOT, 'all labels')

print("\n   Class distribution after mapping:")
mapped_dist = df_all['rating'].value_counts().sort_index()
for rating, count in mapped_dist.items():
    pct = count / len(df_all) * 100
    print(f"   Class {rating}: {count:4d} ({pct:.1f}%)")

# Stratified split
print(f"\n4. Creating stratified train/val/test splits...")
print(f"   Total samples: {len(df_all)}")
print(f"   Ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")

# First split: train vs (val+test)
df_train, df_val_test = train_test_split(
    df_all,
    test_size=(VAL_RATIO + TEST_RATIO),
    stratify=df_all['rating'],
    random_state=RANDOM_SEED
)

# Second split: val vs test
val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
df_val, df_test = train_test_split(
    df_val_test,
    test_size=(1 - val_ratio_adjusted),
    stratify=df_val_test['rating'],
    random_state=RANDOM_SEED
)

print(f"\n   Train set: {len(df_train)} samples")
for rating, count in df_train['rating'].value_counts().sort_index().items():
    pct = count / len(df_train) * 100
    print(f"   Class {rating}: {count:4d} ({pct:.1f}%)")

print(f"\n   Validation set: {len(df_val)} samples")
for rating, count in df_val['rating'].value_counts().sort_index().items():
    pct = count / len(df_val) * 100
    print(f"   Class {rating}: {count:4d} ({pct:.1f}%)")

print(f"\n   Test set: {len(df_test)} samples")
for rating, count in df_test['rating'].value_counts().sort_index().items():
    pct = count / len(df_test) * 100
    print(f"   Class {rating}: {count:4d} ({pct:.1f}%)")

datasets = {'train': df_train, 'val': df_val, 'test': df_test}

# Create output CSVs
print(f"\n5. Saving CSV files to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for name in ('train', 'val', 'test'):
    output_csv = os.path.join(OUTPUT_DIR, f"{name}.csv")
    datasets[name].to_csv(output_csv, index=False)
    print(f"   Saved {output_csv} ({len(datasets[name])} samples)")

# Calculate and display class weights based on final training set
print(f"\n6. Calculating class weights for training...")
train_class_counts = datasets['train']['rating'].value_counts().sort_index()
total_samples = len(datasets['train'])
n_classes = len(train_class_counts)

class_weights = {}
for rating, count in train_class_counts.items():
    weight = total_samples / (n_classes * count)
    class_weights[rating] = weight

print("\n   Class weights (inverse frequency):")
for rating, weight in sorted(class_weights.items()):
    print(f"   Class {rating}: {weight:.4f}")

# Display final statistics
print("\n" + "=" * 60)
print("Final Dataset Statistics:")
print("=" * 60)
for name in ('train', 'val', 'test'):
    dataset = datasets[name]
    print(f"\n{name.upper()} SET: {len(dataset)} samples")
    class_dist = dataset['rating'].value_counts().sort_index()
    for rating, count in class_dist.items():
        pct = count / len(dataset) * 100
        print(f"  Class {rating}: {count:4d} ({pct:.1f}%)")

print("\n" + "=" * 60)
print("CSV generation completed successfully!")
print("=" * 60)
print(f"\nNext steps:")
print(f"1. Update train_regressor.py with:")
print(f"   - DATA_ROOT = r\"{IMAGE_ROOT}\"")
print(f"   - TRAIN_CSV = r\"{os.path.join(OUTPUT_DIR, 'train.csv')}\"")
print(f"   - VAL_CSV = r\"{os.path.join(OUTPUT_DIR, 'val.csv')}\"")
print(f"   - TEST_CSV = r\"{os.path.join(OUTPUT_DIR, 'test.csv')}\"")
print(f"2. Run training!")
