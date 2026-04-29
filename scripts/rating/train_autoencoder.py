import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler

import src.rating as rt


# =========================
# Configuration Constants
# =========================
DATA_ROOT    = r"C:\data\Carreteras\HD15_8m"
TRAIN_CSV    = r"D:\SOTER\SOTER\scripts\rating\datasets\train.csv"
VAL_CSV      = r"D:\SOTER\SOTER\scripts\rating\datasets\val.csv"
TEST_CSV     = r"D:\SOTER\SOTER\scripts\rating\datasets\test.csv"
RANDOM_SEED  = 42

EXPERIMENTS_DIR = r"D:\SOTER\SOTER\scripts\rating\experiments"

IMG_WIDTH  = 54
IMG_HEIGHT = 202

NUM_CLASSES              = 5
EPOCHS_PER_AUTOENCODER   = 50
BATCH_SIZE               = 32
LEARNING_RATE            = 1e-3
WEIGHT_DECAY             = 1e-5
AUTOENCODER_BASE_CHANNELS = 32
USE_UNET                 = True   # U-Net skip connections vs plain symmetric AE
MIN_EPOCH_SIZE           = 2000   # minimum samples per epoch (oversampling for small classes)
EARLY_STOP_PATIENCE      = 10

NUM_WORKERS = 0  # must be 0 on Windows with top-level scripts (no __main__ guard)


# =========================
# Training
# =========================

# --- Experiment setup ---
exp_folder      = rt.create_experiment_folder(EXPERIMENTS_DIR)
logger          = rt.Logger(exp_folder / "training_log.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.log(f"Using device: {device}")
logger.log(f"Experiment folder: {exp_folder}")

logger.log("\n--- Configuration ---")
logger.log(f"MODEL_TYPE                : autoencoder")
logger.log(f"NUM_CLASSES               : {NUM_CLASSES}")
logger.log(f"AUTOENCODER_BASE_CHANNELS : {AUTOENCODER_BASE_CHANNELS}")
logger.log(f"USE_UNET                  : {USE_UNET}")
logger.log(f"EPOCHS={EPOCHS_PER_AUTOENCODER}  BATCH_SIZE={BATCH_SIZE}  LR={LEARNING_RATE}  WD={WEIGHT_DECAY}")
logger.log(f"MIN_EPOCH_SIZE={MIN_EPOCH_SIZE}  EARLY_STOP_PATIENCE={EARLY_STOP_PATIENCE}")

# --- Transforms (no ImageNet normalization for autoencoders) ---
transform         = rt.build_autoencoder_transforms(IMG_HEIGHT, IMG_WIDTH)
augment_transform = rt.build_autoencoder_augmented_transforms(IMG_HEIGHT, IMG_WIDTH)

pin_memory = torch.cuda.is_available()

# --- Per-class training ---
logger.log("\n" + "=" * 60)
logger.log("Training one autoencoder per class")
logger.log("=" * 60)

df_train_full = pd.read_csv(TRAIN_CSV)
df_val_full   = pd.read_csv(VAL_CSV)

class_counts = df_train_full["rating"].round().astype(int).value_counts().to_dict()
logger.log(f"\nTraining class distribution: {dict(sorted(class_counts.items()))}")

best_model_paths = {}

for cls in range(1, NUM_CLASSES + 1):
    logger.log(f"\n--- Class {cls} ---")

    # Filter train and val CSVs for this class
    df_cls_train = df_train_full[df_train_full["rating"].round().astype(int) == cls]
    df_cls_val   = df_val_full[df_val_full["rating"].round().astype(int) == cls]
    n_train = len(df_cls_train)
    n_val   = len(df_cls_val)
    logger.log(f"  Samples: {n_train} train, {n_val} val")

    if n_train == 0:
        logger.log(f"  WARNING: No training samples for class {cls}, skipping.")
        continue

    # Write filtered CSVs for RatingsDataset
    import tempfile, os as _os
    _cls_dir = tempfile.mkdtemp(prefix=f"soter_ae_cls{cls}_")
    cls_train_csv = _os.path.join(_cls_dir, "train.csv")
    cls_val_csv   = _os.path.join(_cls_dir, "val.csv")
    df_cls_train.to_csv(cls_train_csv, index=False)
    df_cls_val.to_csv(cls_val_csv, index=False)

    # Use augmentation for all classes (autoencoders benefit from it)
    train_dataset = rt.RatingsDataset(
        cls_train_csv, DATA_ROOT,
        transform=transform,
        augment_transform=augment_transform,
        augmented_classes=list(range(1, NUM_CLASSES + 1)),
    )

    # Oversample small classes to MIN_EPOCH_SIZE
    num_samples = max(n_train, MIN_EPOCH_SIZE)
    sampler = WeightedRandomSampler(
        weights=[1.0] * n_train,
        num_samples=num_samples,
        replacement=(num_samples > n_train),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )

    if n_val > 0:
        val_dataset = rt.RatingsDataset(cls_val_csv, DATA_ROOT, transform=transform)
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=pin_memory,
        )
    else:
        val_loader = None
        logger.log(f"  WARNING: No validation samples for class {cls}, no early stopping.")

    # Build autoencoder
    AEClass = rt.UNetAutoencoder if USE_UNET else rt.ConvAutoencoder
    model = AEClass(
        in_channels=3,
        base_channels=AUTOENCODER_BASE_CHANNELS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
    ).to(device)
    logger.log(f"  Architecture: {'UNet' if USE_UNET else 'Conv'}  Parameters: {rt.count_trainable_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    best_val_mse   = float("inf")
    patience_count = 0
    best_model_path = exp_folder / f"best_autoencoder_class_{cls}.pth"

    for epoch in range(1, EPOCHS_PER_AUTOENCODER + 1):
        train_mse = rt.train_autoencoder_one_epoch(model, train_loader, optimizer, device)

        if val_loader is not None:
            val_mse = rt.validate_autoencoder(model, val_loader, device)
            scheduler.step(val_mse)
            logger.log(f"  Epoch {epoch}/{EPOCHS_PER_AUTOENCODER} | train_mse: {train_mse:.6f} | val_mse: {val_mse:.6f}")

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                patience_count = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mse": val_mse,
                    "class": cls,
                    "base_channels": AUTOENCODER_BASE_CHANNELS,
                    "architecture": "unet" if USE_UNET else "conv",
                }, best_model_path)
                logger.log(f"    -> Saved best model (val_mse: {val_mse:.6f})")
            else:
                patience_count += 1
                if patience_count >= EARLY_STOP_PATIENCE:
                    logger.log(f"    Early stopping at epoch {epoch}")
                    break
        else:
            scheduler.step(train_mse)
            logger.log(f"  Epoch {epoch}/{EPOCHS_PER_AUTOENCODER} | train_mse: {train_mse:.6f}")
            # No validation: always save latest
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "train_mse": train_mse,
                "class": cls,
                "base_channels": AUTOENCODER_BASE_CHANNELS,
                "architecture": "unet" if USE_UNET else "conv",
            }, best_model_path)

    best_model_paths[cls] = best_model_path
    logger.log(f"  Best val MSE for class {cls}: {best_val_mse:.6f}")


# =========================
# Classification by reconstruction
# =========================
logger.log("\n" + "=" * 60)
logger.log("Classification by minimum reconstruction MSE")
logger.log("=" * 60)

# Load all best autoencoders
autoencoders = {}
for cls, path in best_model_paths.items():
    ckpt = torch.load(path, weights_only=False)
    arch = ckpt.get("architecture", "conv")
    AEClass = rt.UNetAutoencoder if arch == "unet" else rt.ConvAutoencoder
    ae = AEClass(
        in_channels=3,
        base_channels=ckpt.get("base_channels", AUTOENCODER_BASE_CHANNELS),
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
    ).to(device)
    ae.load_state_dict(ckpt["model_state_dict"])
    autoencoders[cls] = ae
    logger.log(f"  Loaded class {cls} autoencoder ({arch}) from epoch {ckpt['epoch']}")

# Evaluate on all splits
val_dataset  = rt.RatingsDataset(VAL_CSV, DATA_ROOT, transform=transform)
test_dataset = rt.RatingsDataset(TEST_CSV, DATA_ROOT, transform=transform)
val_loader   = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=pin_memory)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=pin_memory)

datasets_to_eval = [("val", val_loader), ("test", test_loader)]

rows_5class: list = []
rows_3class: list = []
cms_5class:  dict = {}
cms_3class:  dict = {}
mse_matrices: dict = {}

for ds_name, loader in datasets_to_eval:
    logger.log(f"\nEvaluating on {ds_name} set...")
    all_true, all_pred, mse_matrix = rt.classify_by_reconstruction(
        autoencoders, loader, device, num_classes=NUM_CLASSES,
    )
    mse_matrices[ds_name] = mse_matrix

    # 5-class metrics
    clf_5 = rt.compute_classification_metrics(all_true, all_pred, class_labels=list(range(1, 6)))
    ranges_5 = rt.compute_range_distribution(all_true, all_pred, num_classes=5)
    rows_5class.append({"dataset": ds_name, **clf_5, **ranges_5})
    cm_5 = rt.sk_confusion_matrix(all_true, all_pred, labels=list(range(1, 6)))
    cms_5class[ds_name] = cm_5

    # 3-class metrics
    true_3 = rt.remap_to_3_classes(all_true)
    pred_3 = rt.remap_to_3_classes(all_pred)
    clf_3 = rt.compute_classification_metrics(true_3, pred_3, class_labels=[1, 2, 3])
    ranges_3 = rt.compute_range_distribution(true_3, pred_3, num_classes=3)
    rows_3class.append({"dataset": ds_name, **clf_3, **ranges_3})
    cms_3class[ds_name] = rt.sk_confusion_matrix(true_3, pred_3, labels=[1, 2, 3])

    rt.print_confusion_matrix(cm_5, f"{ds_name.upper()} confusion matrix (5-class)",
                              labels=[1, 2, 3, 4, 5], logger=logger)
    rt.print_confusion_matrix(cms_3class[ds_name], f"{ds_name.upper()} confusion matrix (3-class)",
                              labels=[1, 2, 3], logger=logger)

# --- Save CSVs ---
df_5 = pd.DataFrame(rows_5class)
df_3 = pd.DataFrame(rows_3class)
df_5.to_csv(exp_folder / "metrics_5class.csv", index=False)
df_3.to_csv(exp_folder / "metrics_3class.csv", index=False)
logger.log(f"\nSaved metrics_5class.csv and metrics_3class.csv to {exp_folder}")

for ds_name in cms_5class:
    pd.DataFrame(cms_5class[ds_name], index=[1,2,3,4,5], columns=[1,2,3,4,5]).to_csv(
        exp_folder / f"confusion_matrix_{ds_name}_5class.csv"
    )
    pd.DataFrame(cms_3class[ds_name], index=[1,2,3], columns=[1,2,3]).to_csv(
        exp_folder / f"confusion_matrix_{ds_name}_3class.csv"
    )
logger.log(f"Saved confusion matrix CSVs to {exp_folder}")

# --- Save MSE matrices for later calibration ---
import numpy as _np
for ds_name, mse_mat in mse_matrices.items():
    _np.save(exp_folder / f"mse_matrix_{ds_name}.npy", mse_mat)
logger.log(f"Saved MSE matrices (.npy) for calibration")

# --- Visualise reconstructions ---
logger.log("\nSaving reconstruction visualisations...")
rt.visualise_reconstructions(
    autoencoders, test_loader, device,
    output_path=exp_folder,
    num_classes=NUM_CLASSES,
    samples_per_class=4,
)
logger.log(f"Saved reconstruction grids to {exp_folder}")

# --- Print summaries ---
logger.log("\n" + "=" * 60)
logger.log("Metrics Summary (5-class)")
logger.log("=" * 60)
with pd.option_context("display.max_columns", None):
    logger.log(df_5.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

logger.log("\n" + "=" * 60)
logger.log("Metrics Summary (3-class)")
logger.log("=" * 60)
with pd.option_context("display.max_columns", None):
    logger.log(df_3.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
