"""Write the nested cross-validation results with the PAPER's names.

Reads picks.csv as nested_cv.py writes it (internal ids: E04/E11, inputs 30/15-otf/15-paired) and
writes two CSVs in which raters, inputs and test groups carry the names used in the paper:
  picks.csv          one row per candidate and test group (30 rows): the setting picked by the
                     search, its validation scores and the refit's training and test scores
                     (appendix "Settings picked by each search")
  test_summary.csv   one row per candidate: mean and SD over the 5 test groups of the five test
                     metrics (table of test scores on unseen roads) and the mean training,
                     validation and test QWK (table "QWK on the training, validation and test roads")
All scores are grade-weighted. "adjacent_accuracy" is called within1 in the code, "amae" mae.

    python scripts/rating/nested_cv/export_results.py [RESULTS_DIR]
(default RESULTS_DIR: results/nested_cv, the paper's run, written to results/; any other folder,
e.g. scripts/rating/nested_cv/runs, gets the two CSVs written into it)
"""
import sys
from pathlib import Path

import pandas as pd

from scripts.rating.nested_cv.common import INPUTS, PAPER_BACKBONE, PAPER_INPUT, PAPER_RATER, RESULTS

METRICS = {"qwk": "qwk", "spearman_r": "spearman", "within1": "adjacent_accuracy", "mae": "amae", "rmse": "rmse"}


def paper_picks(p: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "experiment": p.exp.map(PAPER_RATER), "rater": p.exp.map(PAPER_BACKBONE),
        "input": p.leg.map(PAPER_INPUT), "test_group": "G" + p.test_group.astype(str),
        "trial": p.trial, "learning_rate": p.p_lr, "weight_decay": p.p_wd,
        "checks": p.check, "epochs": p.epoch,
        # the trial score: QWK of the worst of the 4 validation groups at the picked check
        "val_worst_qwk": p.val_score, "val_mean_qwk": p.val_mean_qwk, "val_sd_qwk": p.val_sd_qwk,
        "val_mean_adjacent_accuracy": p.val_mean_within1, "val_min_adjacent_accuracy": p.val_min_within1,
        "val_n_counting_checks": p.val_n_counting_checks})
    for g in range(1, 6):                     # per validation group (empty for the test group)
        out[f"val_qwk_G{g}"] = p[f"val_qwk_v{g}"]
        out[f"val_adjacent_accuracy_G{g}"] = p[f"val_within1_v{g}"]
    for split in ("test", "train"):
        for k, name in METRICS.items():
            out[f"{split}_{name}"] = p[f"{split}_{k}"]
    out["internal_exp"], out["internal_input"] = p.exp, p.leg
    return out


def test_summary(q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (exp, rater, inp), g in q.groupby(["experiment", "rater", "input"], sort=False):
        row = {"experiment": exp, "rater": rater, "input": inp, "test_groups": len(g)}
        for name in METRICS.values():
            row[f"test_{name}_mean"] = g[f"test_{name}"].mean()
            row[f"test_{name}_sd"] = g[f"test_{name}"].std(ddof=1)
        row.update({"train_qwk_mean": g.train_qwk.mean(), "val_mean_qwk_mean": g.val_mean_qwk.mean(),
                    "val_worst_qwk_mean": g.val_worst_qwk.mean()})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "nested_cv"
    p = pd.read_csv(src / "summary" / "picks.csv", float_precision="round_trip")
    # rows in the paper's order: rater, then input (native, otf, paired), then test group
    p = p.assign(_o=p.leg.map(INPUTS.index)).sort_values(["exp", "_o", "test_group"]).drop(columns="_o")
    q = paper_picks(p)
    s = test_summary(q)
    out = RESULTS if src.resolve() == (RESULTS / "nested_cv").resolve() else src
    out.mkdir(parents=True, exist_ok=True)
    q.to_csv(out / "picks.csv", index=False)
    s.to_csv(out / "test_summary.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["rater", "input"] + [f"test_{n}_{s_}" for n in ("qwk", "adjacent_accuracy", "amae") for s_ in ("mean", "sd")]
    print(s[cols].round(3).to_string(index=False))
    print(f"\nwritten {out / 'picks.csv'} and {out / 'test_summary.csv'}")


if __name__ == "__main__":
    main()
