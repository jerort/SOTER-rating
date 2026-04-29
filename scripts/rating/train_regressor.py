import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

import src.rating as rt


# =========================
# Configuration Constants
# =========================
BACKBONE        = "densenet121"  # torchvision: "resnet18" | "resnet50" | "resnet152"
                                 #              "densenet121" | "densenet201"
                                 #              "vit_b_16" | "vit_l_16"  (HVCA ignored; expects square input)
                                 # timm:        "darknet53" | "xception"
BACKBONE_BLOCKS = 1
DATA_ROOT    = r"C:\data\Carreteras\HD15_8m"
TRAIN_CSV    = r"D:\SOTER\SOTER\scripts\rating\datasets\train.csv"
VAL_CSV      = r"D:\SOTER\SOTER\scripts\rating\datasets\val.csv"
TEST_CSV     = r"D:\SOTER\SOTER\scripts\rating\datasets\test.csv"
RANDOM_SEED  = 42

EXPERIMENTS_DIR = r"D:\SOTER\SOTER\scripts\rating\experiments"

# Class weighting (mutually exclusive with USE_OVERSAMPLING)
USE_CLASS_WEIGHTS = False

# Weighted random sampling + augmentation proportional to class frequency.
# Minority classes (below average count) are augmented; majority class is not.
USE_OVERSAMPLING  = True

# HV Channel Attention
USE_CHANNEL_ATTENTION     = False
ATTENTION_REDUCTION_RATIO = 4

IMG_WIDTH  = 54
IMG_HEIGHT = 202

EPOCHS        = 30
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 0.01
SMOOTHL1_BETA = 1  # transition point between quadratic and linear regime

# Bayesian Ordinal NN loss (Lázaro & Figueiras-Vidal, PR 2023)
# Handles ordinal structure + class imbalance jointly in the loss function.
# When True, USE_CLASS_WEIGHTS is ignored (BONN handles imbalance internally).
USE_BONN   = False
BONN_MODE  = "amae"  # "mae" (sample-balanced) | "amae" (class-balanced, best for imbalance)
BONN_SIGMA = 0.388  # Gaussian Parzen window std dev (paper uses σ²=0.1507 → σ≈0.388)

NUM_WORKERS   = 0  # must be 0 on Windows with top-level scripts (no __main__ guard)


# =========================
# Training
# =========================

# --- Experiment setup ---
exp_folder      = rt.create_experiment_folder(EXPERIMENTS_DIR)
logger          = rt.Logger(exp_folder / "training_log.txt")
best_model_path = exp_folder / "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.log(f"Using device: {device}")
logger.log(f"Experiment folder: {exp_folder}")

logger.log("\n--- Configuration ---")
logger.log(f"BACKBONE              : {BACKBONE}  (blocks unfrozen: {BACKBONE_BLOCKS})")
logger.log(f"USE_CLASS_WEIGHTS     : {USE_CLASS_WEIGHTS}")
logger.log(f"USE_OVERSAMPLING      : {USE_OVERSAMPLING}")
logger.log(f"USE_CHANNEL_ATTENTION : {USE_CHANNEL_ATTENTION}")
logger.log(f"EPOCHS={EPOCHS}  BATCH_SIZE={BATCH_SIZE}  LR={LEARNING_RATE}  WD={WEIGHT_DECAY}  SMOOTHL1_BETA={SMOOTHL1_BETA}")
logger.log(f"USE_BONN={USE_BONN}  BONN_MODE={BONN_MODE}  BONN_SIGMA={BONN_SIGMA}")

# --- Mutual exclusion: oversampling disables class weights ---
use_weights = USE_CLASS_WEIGHTS
if USE_OVERSAMPLING and USE_CLASS_WEIGHTS:
    logger.log("\nWARNING: USE_OVERSAMPLING is True — class weights disabled.")
    use_weights = False
if USE_BONN and use_weights:
    logger.log("\nWARNING: USE_BONN is True — class weights disabled (BONN handles imbalance).")
    use_weights = False

class_weights = None
if use_weights:
    logger.log("\nCalculating class weights from training data...")
    class_weights = rt.calculate_class_weights(TRAIN_CSV)
    logger.log("Class weights (squared inverse frequency):")
    for r, w in sorted(class_weights.items()):
        logger.log(f"  Class {r}: {w:.4f}")

# --- Transforms & datasets ---
# Augmentation is tied to oversampling: only minority classes get it.
augmented_classes = rt.compute_oversampled_classes(TRAIN_CSV) if USE_OVERSAMPLING else None
logger.log(f"Augmented classes           : {augmented_classes}")

transform         = rt.build_transforms(IMG_HEIGHT, IMG_WIDTH)
augment_transform = rt.build_augmented_transforms(IMG_HEIGHT, IMG_WIDTH) if USE_OVERSAMPLING else None

train_dataset = rt.RatingsDataset(
    TRAIN_CSV, DATA_ROOT,
    transform=transform,
    class_weights=class_weights,
    augment_transform=augment_transform,
    augmented_classes=augmented_classes,
)
val_dataset = rt.RatingsDataset(VAL_CSV, DATA_ROOT, transform=transform)

pin_memory = torch.cuda.is_available()

if USE_OVERSAMPLING:
    sampler      = rt.build_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=sampler,          # mutually exclusive with shuffle=True
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )
    logger.log("Oversampling: WeightedRandomSampler active.")
else:
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=pin_memory,
)

# --- Model ---
model, unfrozen = rt.build_model(
    BACKBONE, BACKBONE_BLOCKS, USE_CHANNEL_ATTENTION, ATTENTION_REDUCTION_RATIO,
    img_size=(IMG_HEIGHT, IMG_WIDTH),
)
model = model.to(device)
logger.log(f"\nUnfrozen modules     : {unfrozen}")
logger.log(f"Trainable parameters : {rt.count_trainable_params(model):,}")

# --- Optimizer & loss ---
if USE_BONN:
    bonn_thresholds   = rt.BONNThresholds(num_classes=5).to(device)
    bonn_class_counts = rt.get_class_counts(TRAIN_CSV)
    criterion         = rt.BONNLoss(num_classes=5, mode=BONN_MODE, sigma=BONN_SIGMA)
    trainable_params  = (
        [p for p in model.parameters() if p.requires_grad]
        + list(bonn_thresholds.parameters())
    )
    logger.log(f"\nBONN class counts  : {bonn_class_counts}")
    logger.log(f"BONN init thresholds: {[f'{v:.3f}' for v in bonn_thresholds().tolist()]}")
else:
    bonn_thresholds   = None
    bonn_class_counts = None
    criterion         = nn.SmoothL1Loss(beta=SMOOTHL1_BETA, reduction="none" if use_weights else "mean")
    trainable_params  = (p for p in model.parameters() if p.requires_grad)

optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

best_val_loss    = float("inf")
epochs_class_log = []  # one dict per epoch: seen/augmented counts per class

logger.log(f"\nStarting training for {EPOCHS} epochs...")
for epoch in range(1, EPOCHS + 1):
    train_loss, class_counts, augmented_counts = rt.train_one_epoch(
        model, train_loader, criterion, optimizer, device, use_weights,
        augmented_classes=augmented_classes,
        bonn_thresholds=bonn_thresholds,
        bonn_class_counts=bonn_class_counts,
    )
    val_loss, val_mae = rt.validate(
        model, val_loader, criterion, device, use_weights,
        bonn_thresholds=bonn_thresholds,
        bonn_class_counts=bonn_class_counts,
    )
    scheduler.step(val_loss)

    logger.log(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train_loss: {train_loss:.4f} | "
        f"val_loss: {val_loss:.4f} | "
        f"val_mae: {val_mae:.4f}"
    )
    if bonn_thresholds is not None:
        thresholds_str = [f"{v:.3f}" for v in bonn_thresholds().tolist()]
        logger.log(f"  BONN thresholds: {thresholds_str}")

    # Log per-class distribution for this epoch
    dist_parts = [
        f"c{c}: {class_counts.get(c, 0)} ({augmented_counts.get(c, 0)} aug)"
        for c in range(1, 6)
    ]
    logger.log("  Classes seen: " + "  ".join(dist_parts))

    # Accumulate for CSV
    row = {"epoch": epoch}
    for c in range(1, 6):
        row[f"class_{c}_seen"]      = class_counts.get(c, 0)
        row[f"class_{c}_augmented"] = augmented_counts.get(c, 0)
    epochs_class_log.append(row)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        ckpt_dict = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss":             val_loss,
            "val_mae":              val_mae,
            "backbone":             BACKBONE,
            "class_weights":        class_weights,
        }
        if bonn_thresholds is not None:
            ckpt_dict["bonn_thresholds_state_dict"] = bonn_thresholds.state_dict()
        torch.save(ckpt_dict, best_model_path)
        logger.log(f"  → Saved best model (val_loss: {val_loss:.4f})")

# --- Save class distribution log ---
df_class_log = pd.DataFrame(epochs_class_log)
df_class_log.to_csv(exp_folder / "training_class_counts.csv", index=False)
logger.log(f"\nSaved per-epoch class counts to training_class_counts.csv")

# --- Log totals across all epochs ---
logger.log("\nTotal instances seen during training:")
logger.log(f"  {'Class':<8} {'Seen':>8} {'Augmented':>10}")
logger.log(f"  {'-'*28}")
for c in range(1, 6):
    seen = df_class_log[f"class_{c}_seen"].sum()
    aug  = df_class_log[f"class_{c}_augmented"].sum()
    logger.log(f"  {c:<8} {seen:>8} {aug:>10}")

# --- Final evaluation ---
logger.log("\n" + "=" * 60)
logger.log("Training completed! Evaluating on test sets...")
logger.log("=" * 60)

ckpt = torch.load(best_model_path, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
if bonn_thresholds is not None and "bonn_thresholds_state_dict" in ckpt:
    bonn_thresholds.load_state_dict(ckpt["bonn_thresholds_state_dict"])
    logger.log(f"  Restored BONN thresholds: {[f'{v:.3f}' for v in bonn_thresholds().tolist()]}")
logger.log(f"\nLoaded best model from epoch {ckpt['epoch']}")

test_dataset = rt.RatingsDataset(TEST_CSV, DATA_ROOT, transform=transform)
test_loader  = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=pin_memory,
)
# For evaluation we use BONN loss if active, otherwise SmoothL1 with mean reduction
criterion_eval   = criterion if USE_BONN else nn.SmoothL1Loss(beta=SMOOTHL1_BETA, reduction="mean")
datasets_to_eval = [("val", val_loader), ("test", test_loader)]

rows_5class: list = []
rows_3class: list = []
cms_5class:  dict = {}
cms_3class:  dict = {}

for ds_name, loader in datasets_to_eval:
    logger.log(f"\nEvaluating on {ds_name} set...")
    reg_row, cm_5, all_true, all_pred = rt.evaluate_dataset(
        ds_name, model, loader, device, criterion_eval,
        bonn_thresholds=bonn_thresholds,
        bonn_class_counts=bonn_class_counts,
    )
    mae  = reg_row["mae_clamped"]
    loss = reg_row["loss_clamped"]
    logger.log(f"  loss: {loss:.4f}  mae: {mae:.4f}")

    ranges = {k: v for k, v in reg_row.items() if k.startswith("range_")}

    clf_5 = rt.compute_classification_metrics(all_true, all_pred, class_labels=[1, 2, 3, 4, 5])
    rows_5class.append({"dataset": ds_name, **clf_5, "mae": mae, "loss": loss, **ranges})
    cms_5class[ds_name] = cm_5

    true_3   = rt.remap_to_3_classes(all_true)
    pred_3   = rt.remap_to_3_classes(all_pred)
    clf_3    = rt.compute_classification_metrics(true_3, pred_3, class_labels=[1, 2, 3])
    ranges_3 = rt.compute_range_distribution(true_3, pred_3, num_classes=3)
    rows_3class.append({"dataset": ds_name, **clf_3, "mae": mae, "loss": loss, **ranges_3})
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
