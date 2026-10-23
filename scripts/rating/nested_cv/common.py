r"""Shared definitions of the nested cross-validation: inputs, raters, road groups and splits.

Collected from the internal research scripts (run_pair_sweep.py, tanda6.py, run_fold_sweep.py,
run_holdout_sweep.py and experiments_registry.py) so that the final pipeline needs nothing else.
The values are the ones the paper's results were produced with; do not change them if you want
to reproduce those results.

Internal ids (used in file and folder names) -> names in the paper:
    rater  E04 -> E01, DenseNet-201        E11 -> E02, ViT-B/16
    input  30  -> native (0.30 m/px)       15-otf -> otf          15-paired -> paired
"""
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath

import pandas as pd

HERE = Path(__file__).resolve().parent              # scripts/rating/nested_cv
ROOT = HERE.parents[2]                              # repository root
SPLIT_DIR = HERE / "datasets_pairs"                 # the split CSVs (in the repository)
POOL_CSV = SPLIT_DIR / "pool.csv"                   # the 5,354 labelled segments (filepath, rating)
RESULTS = ROOT / "results"                          # the results reported in the paper

# Folder holding the three tile sets (rating_30, rating_15_otf, rating_15_paired). The tiles are
# derived from commercial imagery and are not in the repository (available on request).
DATA_DIR = os.environ.get("SOTER_DATA_DIR", str(ROOT / "data"))

# ------------------------------------------------------------------ names
RATERS = ["E04", "E11"]                             # internal ids of the two raters
INPUTS = ["30", "15-otf", "15-paired"]              # internal ids of the three inputs
BACKBONE = {"E04": "densenet201", "E11": "vit_b_16"}            # names in src/rating.py
PAPER_RATER = {"E04": "E01", "E11": "E02"}
PAPER_BACKBONE = {"E04": "DenseNet-201", "E11": "ViT-B/16"}
PAPER_INPUT = {"30": "native", "15-otf": "otf", "15-paired": "paired"}


def leg_dir(leg: str) -> str:
    """Input id as used in file names: '15-otf' -> '15_otf'."""
    return leg.replace("-", "_")


# ------------------------------------------------------------------ the three inputs
@dataclass(frozen=True)
class Leg:
    name: str            # "30" | "15-otf" | "15-paired"
    data_root: str
    img_height: int
    img_width: int
    sr_model: str        # provenance


LEGS: dict[str, Leg] = {
    # 29 px wide, not the 27 px that 8 m would give: DenseNet downsamples by 32 and cannot
    # process a narrower strip, so one extra pixel of shoulder is kept on each side (8.7 m).
    "30": Leg("30", os.path.join(DATA_DIR, "rating_30"), 101, 29,
              "none - native pansharpened Pleiades Neo at 0.30 m/px"),
    "15-otf": Leg("15-otf", os.path.join(DATA_DIR, "rating_15_otf"), 202, 54,
                  "ESRGAN otf_soft (on-the-fly degradation training)"),
    "15-paired": Leg("15-paired", os.path.join(DATA_DIR, "rating_15_paired"), 202, 54,
                     "ESRGAN paired_soft_2000 (paired Pleiades Neo -> Airbus HD15)"),
}


# ------------------------------------------------------------------ the rater's configuration
@dataclass(frozen=True)
class HeadConfig:
    """Everything head_features.py needs to build and train one rater: a frozen ImageNet
    backbone, one linear output and the BONN thresholds. The fixed values are those of the
    configuration the paper's runs used (train_regressor.Config in the internal repository)."""
    backbone: str
    data_root: str
    img_height: int
    img_width: int
    train_csv: str = ""
    seed: int = 0
    learning_rate: float = 1e-4                     # tuned by the search
    weight_decay: float = 1.0                       # tuned by the search (AdamW, head only)
    threshold_lr: float | None = None               # None: thresholds use the same learning rate
    batch_size: int = 32
    bonn_mode: str = "amae"
    bonn_sigma: float = 0.388
    backbone_blocks: str = "head"                   # only the head is trainable
    use_channel_attention: bool = False
    attention_reduction_ratio: int = 4
    head_dropout: float = 0.0
    in_channels: int = 3


def make_cfg(exp: str, leg: str, train_csv="", seed: int = 0, params: dict | None = None) -> HeadConfig:
    params = params or {"lr": 1e-4, "wd": 1.0}
    L = LEGS[leg]
    return HeadConfig(backbone=BACKBONE[exp], data_root=L.data_root, img_height=L.img_height,
                      img_width=L.img_width, train_csv=str(train_csv), seed=seed,
                      learning_rate=params["lr"], weight_decay=params["wd"])


# ------------------------------------------------------------------ road groups
# The five road groups: the unique k_min = 13 optimum of partition_search.py (7,508,501
# partitions), numbered by grade-5 count, largest first.
GROUPS = {
    1: ("A-495R", "A-492", "A-5053"),
    2: ("A-497", "A-5050"),
    3: ("A-490", "A-5058"),
    4: ("A-5052", "A-495", "A-495_1"),
    5: ("A-497_2C", "A-499", "A-5059"),
}
# expected segments per grade 1..5: a guard against a changed label file or group list
EXPECTED = {
    1: (80, 504, 257, 114, 22),
    2: (20, 315, 570, 113, 18),
    3: (105, 427, 648, 158, 13),
    4: (148, 477, 302, 48, 13),
    5: (43, 412, 439, 95, 13),
}
# inner splits (test group t, validation group v), t != v: 5 x 4 = 20, in row-major order
CELLS = [(t, v) for t in GROUPS for v in GROUPS if t != v]
N_CELLS = len(CELLS)
REFIT_SEED = 100                                    # the refit of test group t uses seed 100 + t


def seed_of(cell: tuple[int, int], rep: int = 0) -> int:
    """Seed of an inner split: its index 0..19 (shared by every rater and input)."""
    return CELLS.index(cell) + N_CELLS * rep


def road_of(filepath: str) -> str:
    return PureWindowsPath(filepath).parts[1]       # 'NE\A-490\A-490__1.png' -> 'A-490'


# ------------------------------------------------------------------ splits
def load_pool() -> pd.DataFrame:
    pool = pd.read_csv(POOL_CSV)
    assert len(pool) == 5354 and not pool.filepath.duplicated().any(), "label pool changed"
    return pool


def build_split(t: int, v: int, pool: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """TEST = group t, VALIDATION = group v, TRAIN = the other three groups."""
    roads = pool["filepath"].map(road_of)
    assert set(roads) == {r for g in GROUPS.values() for r in g}, "road list changed"
    test = pool[roads.isin(GROUPS[t])].reset_index(drop=True)
    val = pool[roads.isin(GROUPS[v])].reset_index(drop=True)
    train = pool[~roads.isin(GROUPS[t] + GROUPS[v])].reset_index(drop=True)
    # guards: exactly the documented groups, road-disjoint, nothing lost
    for g, df in ((t, test), (v, val)):
        got = tuple(int((df.rating == c).sum()) for c in range(1, 6))
        assert got == EXPECTED[g], f"group {g}: grades {got} != {EXPECTED[g]}"
    r_tr, r_va, r_te = (set(d.filepath.map(road_of)) for d in (train, val, test))
    assert not (r_tr & r_va or r_tr & r_te or r_va & r_te), "a road is in two sets"
    assert len(train) + len(val) + len(test) == len(pool)
    return {"train": train, "val": val, "test": test}


def _write_or_check(folder: Path, frames: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Write the CSVs once; afterwards verify them, never silently rewrite."""
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, df in frames.items():
        p = folder / f"{name}.csv"
        if p.exists():
            if not pd.read_csv(p).equals(df):
                raise SystemExit(f"{p} differs from the group definition; refusing to mix splits")
        else:
            tmp = p.with_suffix(".csv.part")
            df.to_csv(tmp, index=False)
            os.replace(tmp, p)
        out[name] = p
    return out


def split_paths(t: int, v: int, pool: pd.DataFrame) -> dict[str, Path]:
    """The CSVs of inner split (t, v): datasets_pairs/t{t}_v{v}/{train,val,test}.csv."""
    return _write_or_check(SPLIT_DIR / f"t{t}_v{v}", build_split(t, v, pool))


def refit_paths(t: int, pool: pd.DataFrame) -> dict[str, Path]:
    """The refit of outer fold t: train = the 4 non-test groups, test = group t."""
    roads = pool["filepath"].map(road_of)
    test = pool[roads.isin(GROUPS[t])].reset_index(drop=True)
    train = pool[~roads.isin(GROUPS[t])].reset_index(drop=True)
    got = tuple(int((test.rating == c).sum()) for c in range(1, 6))
    assert got == EXPECTED[t], f"group {t}: grades {got} != {EXPECTED[t]}"
    assert not set(train.filepath.map(road_of)) & set(test.filepath.map(road_of))
    assert len(train) + len(test) == len(pool)
    return _write_or_check(SPLIT_DIR / f"refit_t{t}", {"train": train, "test": test})


# ------------------------------------------------------------------ locks and logs
def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFO
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                                      # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@contextlib.contextmanager
def lock(name: str, base: Path):
    """Only one process per search (or per runner); a lock whose owner died is taken over."""
    base.mkdir(parents=True, exist_ok=True)
    f = base / f".lock_{name}"
    if f.exists():
        try:
            pid = json.loads(f.read_text())["pid"]
        except Exception:
            pid = -1
        if pid != os.getpid() and _pid_alive(pid):
            sys.exit(f"another process (pid {pid}) holds {f.name}; delete it if that process is dead")
    f.write_text(json.dumps({"pid": os.getpid(), "started": datetime.now().isoformat()}))
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            f.unlink()


class _Tee:
    """Write to the terminal and to a log file at the same time, flushing every write."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def tee_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old_out, fh), _Tee(old_err, fh)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err
