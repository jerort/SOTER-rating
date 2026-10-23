r"""The raters of the paper, trained on CACHED backbone features (internal ids E04 and E11).

A rater is a frozen ImageNet backbone (E04 = DenseNet-201, paper E01; E11 = ViT-B/16, paper E02)
used as a fixed feature extractor, followed by ONE linear output and the BONN ordinal thresholds
(Lazaro & Figueiras-Vidal 2023). The input is only resized and normalised (no augmentation), so
every training run would push the same tiles through the same frozen network at every epoch.
This module therefore computes the backbone's features of every tile ONCE per (rater, input)
and trains the head on them, which takes seconds instead of minutes, on the CPU.

The backbone is a truly fixed feature extractor: eval mode, so DenseNet's BatchNorm layers keep
the ImageNet statistics (the usual linear-probe protocol). The ViT has only LayerNorm.

What is cached, per (rater, input), in runs/head_features/ (large, not in the repository):
  * the backbone's features of all 5,354 labelled tiles (eval mode, no gradient): the input of
    the head, obtained by replacing the head with an identity (E11 768 values, E04 1,920);
  * for each seed, the head's INITIAL weights and numpy's RNG state right after the full model
    is built with that seed: the head's initial weights depend on how much of the random stream
    the backbone's construction consumed, so this keeps the runs identical to training the
    full model (frozen backbone + head) end to end.

train_cached() trains the head: representative batches (every grade in every batch), BONN loss
with per-batch normalisation, AdamW with weight decay on the head and none on the thresholds,
constant learning rate, and after every check the grade-weighted BONN cost, AMAE, QWK and
within-1 (adjacent accuracy) on a fixed grade-balanced training subset and on the validation set.

    python scripts/rating/nested_cv/head_features.py build  [--exp E04] [--leg 15-otf]
    python scripts/rating/nested_cv/head_features.py bench  [--exp E04]

`build` needs the image tiles (SOTER_DATA_DIR, see common.py) and downloads the ImageNet weights
(torchvision for DenseNet-201, timm for ViT-B/16); a GPU helps but is not required.
"""
import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.rating as rt
from scripts.rating.nested_cv import common as cm
from scripts.rating.nested_cv.balanced_metrics import bonn_cost, rating_metrics

CACHE = cm.HERE / "runs" / "head_features"
EPOCHS = 10
# seeds used by the nested cross-validation: inner split (t, v) uses seed_of((t, v)) = 0..19; the
# refit of test group t uses REFIT_SEED + t = 101..105. Other seeds are added to the cache on first use.
DEFAULT_SEEDS = list(range(cm.N_CELLS)) + [cm.REFIT_SEED + t for t in cm.GROUPS]


def cache_path(exp: str, leg: str) -> Path:
    return CACHE / f"features_{exp}_{cm.leg_dir(leg)}.pt"


def head_cfg(exp, leg, train_csv="", seed=0, params=None) -> cm.HeadConfig:
    """The configuration of this rater and input (see common.HeadConfig)."""
    assert exp in cm.RATERS, f"{exp} is not one of the raters {cm.RATERS}"
    return cm.make_cfg(exp, leg, train_csv, seed, params)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_full(cfg):
    """The full model (frozen backbone + linear head), built on the CPU."""
    model, _ = rt.build_model(
        cfg.backbone, cfg.backbone_blocks, cfg.use_channel_attention, cfg.attention_reduction_ratio,
        img_size=(cfg.img_height, cfg.img_width), in_channels=cfg.in_channels,
        head_dropout=cfg.head_dropout)
    return model


def head_init(cfg, seed: int) -> dict:
    """The head's initial weights and numpy's global RNG state right after building the model
    with this seed (the batches are then drawn from that state)."""
    seed_everything(seed)
    model = build_full(cfg)
    _, head = rt.get_head_module(model)
    assert isinstance(head, torch.nn.Linear), f"head is {type(head)}, expected one linear layer"
    return {"head": {k: v.detach().clone() for k, v in head.state_dict().items()},
            "np_state": np.random.get_state()}


# ------------------------------------------------------------------ the cache
def build_cache(exp: str, leg: str, seeds=DEFAULT_SEEDS, device=None, batch_size=128) -> Path:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = head_cfg(exp, leg)
    CACHE.mkdir(parents=True, exist_ok=True)
    pool = cm.load_pool()
    pool_csv = CACHE / "pool.csv"                 # the 5,354 labelled tiles, in load_pool() order
    pool[["filepath", "rating"]].to_csv(pool_csv, index=False)

    t0 = time.time()
    inits = {int(s): head_init(cfg, int(s)) for s in seeds}
    model = build_full(cfg)
    head_name, head = rt.get_head_module(model)
    n_feat = head.in_features
    setattr(model, head_name, torch.nn.Identity())   # model(x) = the head's input
    model = model.to(device).eval()                  # eval: BatchNorm keeps the ImageNet statistics
    tf = rt.build_transforms(cfg.img_height, cfg.img_width, in_channels=cfg.in_channels)
    ds = rt.RatingsDataset(str(pool_csv), cfg.data_root, transform=tf, in_channels=cfg.in_channels)
    feats = []
    with torch.no_grad():
        for images, _r, _w in DataLoader(ds, batch_size=batch_size, shuffle=False):
            feats.append(model(images.to(device)).float().cpu())
    feats = torch.cat(feats)
    assert feats.shape == (len(pool), n_feat), feats.shape
    out = cache_path(exp, leg)
    torch.save({"filepaths": pool["filepath"].tolist(), "features": feats, "head_init": inits,
                "meta": {"exp": exp, "leg": leg, "backbone": cfg.backbone, "img": [cfg.img_height, cfg.img_width],
                         "data_root": cfg.data_root, "device": str(device), "batchnorm": "eval (ImageNet statistics)",
                         "torch": torch.__version__, "created": datetime.now().isoformat(timespec="seconds")}},
               out.with_suffix(".pt.part"))
    out.with_suffix(".pt.part").replace(out)
    print(f"{exp} {leg}: {feats.shape[0]} tiles x {feats.shape[1]} features, {len(inits)} seeds "
          f"-> {out.name} ({time.time() - t0:.0f} s)", flush=True)
    return out


_LOADED: dict = {}


def load_cache(exp: str, leg: str) -> dict:
    """The cache of one rater and input, kept in memory; adds a filepath -> row index."""
    if (exp, leg) not in _LOADED:
        if not cache_path(exp, leg).exists():
            raise SystemExit(f"no feature cache {cache_path(exp, leg)}; run "
                             f"'head_features.py build --exp {exp} --leg {leg}' first")
        c = torch.load(cache_path(exp, leg), weights_only=False)
        c["row"] = {fp: i for i, fp in enumerate(c["filepaths"])}
        _LOADED[(exp, leg)] = c
    return _LOADED[(exp, leg)]


def head_init_cached(exp: str, leg: str, seed: int) -> dict:
    """A seed's initial head weights; a seed not in the cache is computed and saved once."""
    c = load_cache(exp, leg)
    if seed not in c["head_init"]:
        c["head_init"][seed] = head_init(head_cfg(exp, leg), seed)
        saved = torch.load(cache_path(exp, leg), weights_only=False)
        saved["head_init"][seed] = c["head_init"][seed]
        torch.save(saved, cache_path(exp, leg).with_suffix(".pt.part"))
        cache_path(exp, leg).with_suffix(".pt.part").replace(cache_path(exp, leg))
    return c["head_init"][seed]


def features_of(exp: str, leg: str, csv_path) -> tuple[pd.DataFrame, torch.Tensor]:
    """The split's rows (CSV order) and their cached features."""
    c = load_cache(exp, leg)
    df = pd.read_csv(csv_path)
    idx = [c["row"][fp] for fp in df["filepath"]]
    return df, c["features"][idx]


# ------------------------------------------------------------------ training on the cache
def train_cached(exp, leg, train_csv, val_csv, seed, params, epochs=EPOCHS, checks_per_epoch=1,
                 device="cpu", return_head=False, n_checks=None):
    """Train the head for exactly `epochs` epochs at a constant learning rate (no early
    stopping), with `checks_per_epoch` checks per epoch. Returns the per-check curves and, if
    return_head, the trained head and thresholds. val_csv "" or None: no validation set (refit).
    n_checks: stop after that many checks instead (a refit trains for the E checks its search
    picked)."""
    device = torch.device(device)
    cfg = head_cfg(exp, leg, train_csv, seed, params)
    seed_everything(seed)

    train_df, Xtr = features_of(exp, leg, train_csv)
    ytr = torch.tensor(train_df["rating"].to_numpy(float), dtype=torch.float32)
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    sampler = rt.RepresentativeBatchSampler(train_df["rating"].values, cfg.batch_size, checks_per_epoch)

    init = head_init_cached(exp, leg, seed)
    np.random.set_state(init["np_state"])                 # the batches start from here
    head = torch.nn.Linear(Xtr.shape[1], 1)
    head.load_state_dict(init["head"])
    head = head.to(device)
    thresholds = rt.BONNThresholds(num_classes=5).to(device)
    counts = rt.get_class_counts(cfg.train_csv)
    criterion = rt.BONNLoss(num_classes=5, mode=cfg.bonn_mode, sigma=cfg.bonn_sigma, normalisation="batch")
    optimizer = torch.optim.AdamW(
        [{"params": list(head.parameters()), "weight_decay": cfg.weight_decay},
         {"params": list(thresholds.parameters()), "weight_decay": 0.0,
          "lr": cfg.threshold_lr or cfg.learning_rate}], lr=cfg.learning_rate)

    # a fixed grade-balanced training subset (k segments per grade, own RNG) for the train curves
    grades = train_df["rating"].round().astype(int).values
    k = np.bincount(grades, minlength=6)[1:].min()
    rng = np.random.RandomState(cfg.seed)
    sub = np.concatenate([rng.choice(np.flatnonzero(grades == g), k, replace=False) for g in range(1, 6)])
    evals = {"train": (Xtr[torch.as_tensor(sub, device=device)], grades[sub])}
    if val_csv:
        val_df, Xva = features_of(exp, leg, val_csv)
        evals["val"] = (Xva.to(device), val_df["rating"].round().astype(int).values)

    @torch.no_grad()
    def check_eval() -> dict:
        u = thresholds().detach()
        row = {}
        for name, (X, t) in evals.items():
            z = head(X).squeeze(1)
            pred = thresholds.predict(z).cpu().numpy()
            m = rating_metrics(t, pred, weighted=True)
            row.update({f"{name}_cost": bonn_cost(t, z.cpu().numpy(), u.cpu().numpy(), cfg.bonn_sigma, weighted=True),
                        f"{name}_amae": m["mae"], f"{name}_qwk": m["qwk"], f"{name}_within1": m["within1"]})
        return row

    rows = []
    for check in range(1, (n_checks or epochs * checks_per_epoch) + 1):
        running, total = 0.0, 0
        for batch in sampler:                             # one pass of the sampler = one check
            b = torch.as_tensor(batch, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(Xtr[b]).squeeze(1), ytr[b], thresholds(), counts)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch)
            total += len(batch)
        rows.append({"check": check, "epoch": check / checks_per_epoch,
                     "train_loss_trainmode": running / max(total, 1),
                     "lr": optimizer.param_groups[0]["lr"], **check_eval()})
    curves = pd.DataFrame(rows)
    if return_head:
        return curves, {k: v.detach().cpu() for k, v in head.state_dict().items()}, \
            {k: v.detach().cpu() for k, v in thresholds.state_dict().items()}
    return curves


# ------------------------------------------------------------------ main
def bench(exp: str, device: str) -> None:
    pool = cm.load_pool()
    split = cm.split_paths(1, 2, pool)
    train_cached(exp, "15-otf", split["train"], split["val"], 0, {"lr": 1e-4, "wd": 1.0}, 1, device=device)  # warm-up
    for checks in (1, 4):
        t0 = time.time()
        train_cached(exp, "15-otf", split["train"], split["val"], 0, {"lr": 1e-4, "wd": 1.0}, EPOCHS,
                     checks_per_epoch=checks, device=device)
        print(f"bench {exp} {device}: one 10-epoch run, {checks} check(s) per epoch: {time.time() - t0:.1f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "bench"])
    ap.add_argument("--exp", choices=cm.RATERS, help="one rater only (default both)")
    ap.add_argument("--leg", choices=cm.INPUTS, help="build: one input only (default all three)")
    ap.add_argument("--device", default="cpu", help="bench: device that trains the head")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True          # as in the paper's runs
    torch.backends.cudnn.allow_tf32 = True
    exps = [a.exp] if a.exp else list(cm.RATERS)
    if a.command == "build":
        for exp in exps:
            for leg in ([a.leg] if a.leg else cm.INPUTS):
                build_cache(exp, leg)
    else:
        for exp in exps:
            bench(exp, a.device)


if __name__ == "__main__":
    sys.exit(main())
