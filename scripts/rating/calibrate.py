"""
calibrate.py
------------
Post-hoc probability calibration for trained regressor or autoencoder experiments.

Regressor:  fits GaussianCalibrator and OrdinalLogisticCalibrator on val residuals.
Autoencoder: fits TemperatureCalibrator on val MSE matrix.

Usage:
    python calibrate.py <experiment_dir>
    python calibrate.py                      # uses the hardcoded EXPERIMENT_DIR below
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

import src.rating as rt


# =========================
# Configuration
# =========================
DATA_ROOT       = r"C:\data\Carreteras\HD15_8m"
TRAIN_CSV       = r"D:\SOTER\SOTER\scripts\rating\datasets\train.csv"
VAL_CSV         = r"D:\SOTER\SOTER\scripts\rating\datasets\val.csv"
TEST_CSV        = r"D:\SOTER\SOTER\scripts\rating\datasets\test.csv"
RANDOM_SEED     = 42

EXPERIMENT_DIR  = r"D:\SOTER\SOTER\scripts\rating\experiments\experiment_XXXXXXXX_XXXXXX_CET"
if len(sys.argv) > 1:
    EXPERIMENT_DIR = sys.argv[1]

IMG_WIDTH       = 54
IMG_HEIGHT      = 202
BATCH_SIZE      = 32
NUM_CLASSES     = 5
NUM_WORKERS     = 0


# =========================
# Helpers
# =========================

def detect_experiment_type(exp_dir: Path) -> str:
    """Detect whether the experiment is a regressor or autoencoder."""
    if (exp_dir / "best_model.pth").exists():
        return "regressor"
    if (exp_dir / "best_autoencoder_class_1.pth").exists():
        return "autoencoder"
    raise FileNotFoundError(
        f"Cannot detect experiment type in {exp_dir}: "
        "no best_model.pth or best_autoencoder_class_*.pth found."
    )


def extract_identifiers(dataset: rt.RatingsDataset) -> list:
    """Extract identifier (filename stem) from each filepath in the dataset."""
    return [Path(fp).stem for fp in dataset.df["filepath"]]


def parse_log_config(log_path: Path) -> dict:
    """Extract backbone and attention settings from training_log.txt."""
    cfg = {
        "backbone": "densenet121",
        "use_channel_attention": False,
        "attention_reduction_ratio": 4,
    }
    if not log_path.exists():
        return cfg
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if re.match(r"\s*BACKBONE\s*:", line):
            cfg["backbone"] = line.split(":", 1)[1].strip().split()[0]
        elif re.match(r"\s*USE_CHANNEL_ATTENTION\s*:", line):
            cfg["use_channel_attention"] = (
                line.split(":", 1)[1].strip().split()[0].lower() == "true"
            )
    return cfg


def build_inference_model(backbone: str, use_ca: bool, reduction_ratio: int = 4) -> nn.Module:
    """Reconstruct regressor architecture for inference (no pretrained weights).

    Supports all backbones used by rt.build_model: torchvision ResNets/DenseNets,
    timm CNNs (darknet53, xception, …), and timm ViTs (vit_b_16, vit_l_16).
    """
    if backbone == "resnet18":
        m = models.resnet18(weights=None);    m.fc = nn.Linear(m.fc.in_features, 1)
    elif backbone == "resnet50":
        m = models.resnet50(weights=None);    m.fc = nn.Linear(m.fc.in_features, 1)
    elif backbone == "resnet152":
        m = models.resnet152(weights=None);   m.fc = nn.Linear(m.fc.in_features, 1)
    elif backbone == "densenet121":
        m = models.densenet121(weights=None); m.classifier = nn.Linear(m.classifier.in_features, 1)
    elif backbone == "densenet201":
        m = models.densenet201(weights=None); m.classifier = nn.Linear(m.classifier.in_features, 1)
    elif backbone in rt._TIMM_CHANNELS:
        m = rt.TimmBackboneWithHVCA(
            backbone, rt._TIMM_CHANNELS[backbone],
            use_attention=use_ca, reduction_ratio=reduction_ratio,
        )
        return m  # HVCA already integrated
    elif backbone in rt._VIT_TIMM_NAMES:
        m = rt.TimmViTRegressor(
            rt._VIT_TIMM_NAMES[backbone], img_size=(IMG_HEIGHT, IMG_WIDTH),
            use_attention=use_ca,
        )
        return m  # no post-hoc HVCA for ViTs
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    if use_ca:
        if backbone in rt._RESNET_CHANNELS:
            m = rt.ResNetWithHVCA(m, backbone, reduction_ratio)
        elif backbone in rt._DENSENET_CHANNELS:
            m = rt.DenseNetWithHVCA(m, backbone, reduction_ratio)
    return m


@torch.no_grad()
def collect_regressor_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Run regressor inference, return (y_true, y_hat) as numpy arrays."""
    model.eval()
    all_true, all_hat = [], []
    for images, ratings, _weights in loader:
        images = images.to(device, non_blocking=True)
        preds = model(images).squeeze(1).cpu().numpy()
        all_hat.append(preds)
        all_true.append(ratings.numpy())
    return np.concatenate(all_true), np.concatenate(all_hat)


def compute_calibration_metrics(
    y_true_int: np.ndarray,
    probs: np.ndarray,
    num_classes: int = 5,
    n_bins: int = 10,
) -> dict:
    """Compute Expected Calibration Error (ECE) and Brier score."""
    # Brier score: mean squared error between one-hot and predicted probabilities
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true_int)), y_true_int - 1] = 1.0
    brier = float(((probs - one_hot) ** 2).sum(axis=1).mean())

    # ECE: bin by max predicted probability
    max_probs = probs.max(axis=1)
    pred_classes = probs.argmax(axis=1) + 1
    correct = (pred_classes == y_true_int).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (max_probs > bin_boundaries[i]) & (max_probs <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = max_probs[mask].mean()
        avg_acc = correct[mask].mean()
        ece += mask.sum() / len(y_true_int) * abs(avg_acc - avg_conf)

    return {"ece": float(ece), "brier": brier}


# =========================
# Main
# =========================

exp_dir = Path(EXPERIMENT_DIR)
exp_type = detect_experiment_type(exp_dir)
logger = rt.Logger(exp_dir / "calibration_log.txt")
logger.log(f"Experiment: {exp_dir.name}")
logger.log(f"Experiment type: {exp_type}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.log(f"Using device: {device}")

train_csv, val_csv, test_csv = TRAIN_CSV, VAL_CSV, TEST_CSV

pin_memory = torch.cuda.is_available()
datasets_to_eval = ["train", "val", "test"]
csv_map = {"train": train_csv, "val": val_csv, "test": test_csv}


# =====================================================================
# REGRESSOR CALIBRATION
# =====================================================================
if exp_type == "regressor":
    logger.log("\n" + "=" * 60)
    logger.log("Regressor calibration")
    logger.log("=" * 60)

    # Load model
    cfg = parse_log_config(exp_dir / "training_log.txt")
    model = build_inference_model(
        cfg["backbone"], cfg["use_channel_attention"], cfg["attention_reduction_ratio"],
    )
    ckpt = torch.load(exp_dir / "best_model.pth", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    logger.log(f"  Loaded model: backbone={cfg['backbone']}, epoch={ckpt['epoch']}")

    transform = rt.build_transforms(IMG_HEIGHT, IMG_WIDTH)

    # Collect predictions on val set
    val_dataset = rt.RatingsDataset(val_csv, DATA_ROOT, transform=transform)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin_memory)
    y_true_val, y_hat_val = collect_regressor_predictions(model, val_loader, device)
    logger.log(f"  Val predictions: N={len(y_true_val)}, "
               f"y_hat range=[{y_hat_val.min():.3f}, {y_hat_val.max():.3f}]")

    # Fit calibrators
    gaussian_cal = rt.GaussianCalibrator(num_classes=NUM_CLASSES).fit(y_true_val, y_hat_val)
    logger.log(f"\n  GaussianCalibrator: sigma={gaussian_cal.sigma:.4f}")

    ordinal_cal = rt.OrdinalLogisticCalibrator(num_classes=NUM_CLASSES).fit(y_true_val, y_hat_val)
    logger.log(f"  OrdinalLogisticCalibrator: beta={ordinal_cal.beta:.4f}, "
               f"alphas={[f'{a:.3f}' for a in ordinal_cal.alphas]}")

    # Evaluate on all splits
    for cal_name, calibrator in [("gaussian", gaussian_cal), ("ordinal", ordinal_cal)]:
        logger.log(f"\n--- {cal_name.upper()} calibration ---")
        rows = []

        for ds_name in datasets_to_eval:
            ds = rt.RatingsDataset(csv_map[ds_name], DATA_ROOT, transform=transform)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=pin_memory)
            y_true, y_hat = collect_regressor_predictions(model, loader, device)

            probs = calibrator.predict_proba(y_hat)
            pred_labels = calibrator.predict(y_hat)
            true_int = np.round(y_true).astype(int).clip(1, NUM_CLASSES)

            clf = rt.compute_classification_metrics(true_int, pred_labels, list(range(1, 6)))
            cal_metrics = compute_calibration_metrics(true_int, probs, NUM_CLASSES)
            rows.append({"dataset": ds_name, **clf, **cal_metrics})

            logger.log(f"  {ds_name}: balanced_acc={clf['balanced_accuracy']:.4f}  "
                        f"qwk={clf['qwk']:.4f}  ece={cal_metrics['ece']:.4f}  "
                        f"brier={cal_metrics['brier']:.4f}")

            # Save per-sample probabilities
            prob_df = pd.DataFrame(probs, columns=[f"prob_{k}" for k in range(1, 6)])
            prob_df["y_true"] = true_int
            prob_df["y_hat"] = y_hat
            prob_df["pred_class"] = pred_labels
            prob_df["identifier"] = extract_identifiers(ds)
            prob_df.to_csv(exp_dir / f"calibration_{cal_name}_{ds_name}.csv", index=False)

        df_cal = pd.DataFrame(rows)
        df_cal.to_csv(exp_dir / f"calibration_{cal_name}_metrics.csv", index=False)

    # Save calibration parameters
    params = {
        "gaussian": gaussian_cal.get_params(),
        "ordinal": ordinal_cal.get_params(),
    }
    with open(exp_dir / "calibration_params.json", "w") as f:
        json.dump(params, f, indent=2)
    logger.log(f"\nSaved calibration parameters to calibration_params.json")


# =====================================================================
# AUTOENCODER CALIBRATION
# =====================================================================
elif exp_type == "autoencoder":
    logger.log("\n" + "=" * 60)
    logger.log("Autoencoder calibration (temperature scaling)")
    logger.log("=" * 60)

    transform = rt.build_autoencoder_transforms(IMG_HEIGHT, IMG_WIDTH)

    # Load all autoencoders
    autoencoders = {}
    for cls in range(1, NUM_CLASSES + 1):
        path = exp_dir / f"best_autoencoder_class_{cls}.pth"
        if not path.exists():
            logger.log(f"  WARNING: {path.name} not found, skipping class {cls}")
            continue
        ckpt = torch.load(path, weights_only=False)
        arch = ckpt.get("architecture", "conv")
        AEClass = rt.UNetAutoencoder if arch == "unet" else rt.ConvAutoencoder
        ae = AEClass(
            in_channels=3,
            base_channels=ckpt.get("base_channels", 32),
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
        ).to(device)
        ae.load_state_dict(ckpt["model_state_dict"])
        autoencoders[cls] = ae
        logger.log(f"  Loaded class {cls} autoencoder from epoch {ckpt['epoch']}")

    # Collect MSE matrix on val set
    val_dataset = rt.RatingsDataset(val_csv, DATA_ROOT, transform=transform)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin_memory)
    y_true_val, _, mse_val = rt.classify_by_reconstruction(
        autoencoders, val_loader, device, num_classes=NUM_CLASSES,
    )

    # Fit temperature calibrator
    temp_cal = rt.TemperatureCalibrator(num_classes=NUM_CLASSES).fit(y_true_val, mse_val)
    logger.log(f"\n  TemperatureCalibrator: T={temp_cal.temperature:.6f}")

    # Evaluate on all splits
    rows = []
    for ds_name in datasets_to_eval:
        ds = rt.RatingsDataset(csv_map[ds_name], DATA_ROOT, transform=transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin_memory)
        y_true, _, mse_matrix = rt.classify_by_reconstruction(
            autoencoders, loader, device, num_classes=NUM_CLASSES,
        )

        probs = temp_cal.predict_proba(mse_matrix)
        pred_labels = temp_cal.predict(mse_matrix)
        true_int = np.round(y_true).astype(int).clip(1, NUM_CLASSES)

        clf = rt.compute_classification_metrics(true_int, pred_labels, list(range(1, 6)))
        cal_metrics = compute_calibration_metrics(true_int, probs, NUM_CLASSES)
        rows.append({"dataset": ds_name, **clf, **cal_metrics})

        logger.log(f"  {ds_name}: balanced_acc={clf['balanced_accuracy']:.4f}  "
                    f"qwk={clf['qwk']:.4f}  ece={cal_metrics['ece']:.4f}  "
                    f"brier={cal_metrics['brier']:.4f}")

        # Save per-sample probabilities
        prob_df = pd.DataFrame(probs, columns=[f"prob_{k}" for k in range(1, 6)])
        prob_df["y_true"] = true_int
        prob_df["pred_class"] = pred_labels
        prob_df["identifier"] = extract_identifiers(ds)
        prob_df.to_csv(exp_dir / f"calibration_temperature_{ds_name}.csv", index=False)

    df_cal = pd.DataFrame(rows)
    df_cal.to_csv(exp_dir / "calibration_temperature_metrics.csv", index=False)

    # Save calibration parameters
    params = {"temperature": temp_cal.get_params()}
    with open(exp_dir / "calibration_params.json", "w") as f:
        json.dump(params, f, indent=2)
    logger.log(f"\nSaved calibration parameters to calibration_params.json")


logger.log("\n" + "=" * 60)
logger.log("Calibration complete!")
logger.log("=" * 60)
