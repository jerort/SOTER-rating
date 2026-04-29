import re
from pathlib import Path

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
OUTPUT_XLSX     = Path(__file__).parent / "experiments_comparison.xlsx"

DATASETS = ["test"]

# ── helpers ────────────────────────────────────────────────────────────────

def parse_config(log_path: Path) -> dict:
    """Extract hyperparameters from training_log.txt."""
    config = {}
    if not log_path.exists():
        return config

    text = log_path.read_text(encoding="utf-8", errors="ignore")

    # Detect model type
    if re.search(r"MODEL_TYPE\s*:\s*autoencoder", text):
        config["model_type"] = "autoencoder"
    else:
        config["model_type"] = "regressor"

    if config["model_type"] == "regressor":
        # BACKBONE : resnet18  (blocks unfrozen: 3)
        m = re.search(r"BACKBONE\s*:\s*(\S+)\s*\(blocks unfrozen:\s*(\d+)\)", text)
        if m:
            config["backbone"]        = m.group(1)
            config["backbone_blocks"] = int(m.group(2))

        for flag in ("USE_CLASS_WEIGHTS", "USE_OVERSAMPLING", "USE_CHANNEL_ATTENTION", "USE_BONN"):
            m = re.search(rf"{flag}\s*:\s*(\S+)", text)
            if m:
                config[flag.lower()] = m.group(1) == "True"

        # EPOCHS=20  BATCH_SIZE=32  LR=0.0001  WD=0.01  SMOOTHL1_BETA=0.1
        m = re.search(
            r"EPOCHS=(\d+)\s+BATCH_SIZE=(\d+)\s+LR=([\d.eE+\-]+)\s+WD=([\d.eE+\-]+)\s+SMOOTHL1_BETA=([\d.eE+\-]+)",
            text,
        )
        if m:
            config["epochs"]        = int(m.group(1))
            config["batch_size"]    = int(m.group(2))
            config["lr"]            = float(m.group(3))
            config["weight_decay"]  = float(m.group(4))
            config["smoothl1_beta"] = float(m.group(5))

        # USE_BONN=False  BONN_MODE=amae  BONN_SIGMA=0.388
        m = re.search(r"BONN_MODE=(\S+)\s+BONN_SIGMA=([\d.eE+\-]+)", text)
        if m:
            config["bonn_mode"]  = m.group(1)
            config["bonn_sigma"] = float(m.group(2))

        # Trainable parameters : 1,234,567
        m = re.search(r"Trainable parameters\s*:\s*([\d,]+)", text)
        if m:
            config["trainable_params"] = int(m.group(1).replace(",", ""))

        # Augmented classes : [1, 4, 5]
        m = re.search(r"Augmented classes\s*:\s*(\[.*?\])", text)
        if m:
            config["augmented_classes"] = m.group(1)

    else:  # autoencoder
        m = re.search(r"AUTOENCODER_BASE_CHANNELS\s*:\s*(\d+)", text)
        if m:
            config["autoencoder_base_channels"] = int(m.group(1))

        # EPOCHS=50  BATCH_SIZE=32  LR=0.001  WD=1e-05
        m = re.search(
            r"EPOCHS=(\d+)\s+BATCH_SIZE=(\d+)\s+LR=([\d.eE+\-]+)\s+WD=([\d.eE+\-]+)",
            text,
        )
        if m:
            config["epochs"]       = int(m.group(1))
            config["batch_size"]   = int(m.group(2))
            config["lr"]           = float(m.group(3))
            config["weight_decay"] = float(m.group(4))

        m = re.search(r"MIN_EPOCH_SIZE=(\d+)", text)
        if m:
            config["min_epoch_size"] = int(m.group(1))

        m = re.search(r"EARLY_STOP_PATIENCE=(\d+)", text)
        if m:
            config["early_stop_patience"] = int(m.group(1))

        # Per-class param count (all identical) — take the first
        m = re.search(r"Parameters:\s*([\d,]+)", text)
        if m:
            config["trainable_params"] = int(m.group(1).replace(",", ""))

    return config


def load_metrics(csv_path: Path) -> dict[str, dict]:
    """Return {dataset: {metric: value}} from a metrics CSV."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    df["dataset"] = df["dataset"].str.strip()
    return {row["dataset"]: row.drop("dataset").to_dict() for _, row in df.iterrows()}


def build_sheet(experiments: list[Path], n_classes: int) -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(experiments):
        metrics_all = load_metrics(exp_dir / f"metrics_{n_classes}class.csv")
        config      = parse_config(exp_dir / "training_log.txt")

        row = {"experiment": exp_dir.name}

        for ds in DATASETS:
            ds_metrics = metrics_all.get(ds, {})
            for metric, value in ds_metrics.items():
                row[f"{ds}_{metric}"] = value

        # Calibration metrics (if available)
        for cal_method in ("gaussian", "ordinal", "temperature"):
            cal_csv = exp_dir / f"calibration_{cal_method}_metrics.csv"
            if cal_csv.exists():
                cal_data = load_metrics(cal_csv)
                for ds in DATASETS:
                    ds_cal = cal_data.get(ds, {})
                    for m_name in ("balanced_accuracy", "qwk", "ece", "brier"):
                        if m_name in ds_cal:
                            row[f"{ds}_{cal_method}_{m_name}"] = ds_cal[m_name]

        row.update(config)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Place model_type and trainable_params as early columns
    for col_name, pos in [("model_type", 1), ("trainable_params", 2)]:
        if col_name in df.columns:
            cols = df.columns.tolist()
            cols.remove(col_name)
            cols.insert(pos, col_name)
            df = df[cols]

    return df


experiments = [p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir()]
if not experiments:
    print(f"No experiment folders found under {EXPERIMENTS_DIR}")
else:
    print(f"Found {len(experiments)} experiment(s).")

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for n_classes in (3, 5):
            df = build_sheet(experiments, n_classes)
            sheet_name = f"metrics_{n_classes}class"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Auto-fit column widths
            ws = writer.sheets[sheet_name]
            for col_cells in ws.columns:
                max_len = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in col_cells
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)

            print(f"  Sheet '{sheet_name}': {len(df)} rows × {len(df.columns)} columns")

        print(f"\nSaved: {OUTPUT_XLSX}")
