"""The two reference points of the paper: the human baseline ("fresh expert") and the naive baseline.

Human baseline (paper: section "Metrics and reference points"; this script rebuilds the table of
the two reference points, by road group):
  * segments graded by all three specialists (labels.xlsx, sheet ALL, columns 5-7);
  * each specialist is compared with a consensus of the OTHER TWO only. Where those two agree, the
    reference is their grade; where they disagree, each of their two grades is the reference with
    probability 1/2 (a random tie-break). Linear quantities (identical, within one, absolute and
    squared error) are averaged over the two candidates, which is their exact expectation; QWK and
    Spearman are not linear, so they are averaged over N_DRAWS random tie-breaks;
  * grade weighting by the PUBLISHED label (the weight a model's segment would get);
  * every figure is the mean over the three specialists; RMSE is the root of the weighted mean
    squared error of each specialist, then averaged.
  * ICC(2,1) and ICC(2,3), Shrout & Fleiss two-way random, absolute agreement, on the gradings.
Naive baseline (no-skill benchmark): always predicting grade 3, scored with the same grade weights.

    python scripts/rating/nested_cv/human_reference.py [--exclude-out-of-scale]
Output: results/human_reference.csv
"""
import argparse

import numpy as np
import pandas as pd

from scripts.rating.nested_cv.balanced_metrics import rating_metrics
from scripts.rating.nested_cv.common import GROUPS, RESULTS, ROOT

LABELS = ROOT / "scripts" / "rating" / "data_inspection" / "labels.xlsx"
N_DRAWS = 500
SEED = 0


def groups_of(roads: pd.Series) -> pd.Series:
    g = {r: k for k, rs in GROUPS.items() for r in rs}
    return roads.map(g)


def icc2(X: np.ndarray) -> tuple[float, float]:
    """ICC(2,1) and ICC(2,k) of an n x k matrix (Shrout & Fleiss 1979)."""
    n, k = X.shape
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return ((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n),
            (msr - mse) / (msr + (msc - mse) / n))


def label_weights(label: np.ndarray) -> np.ndarray:
    n = pd.Series(label).map(pd.Series(label).value_counts()).values
    w = 1.0 / n
    return w / w.sum()


def fresh_expert(G: np.ndarray, label: np.ndarray, rng) -> dict:
    """G: n x 3 gradings. Returns the reliability columns and the five comparison metrics."""
    w = label_weights(label)
    out = {k: [] for k in ("identical", "within1", "amae", "rmse", "qwk", "spearman")}
    for s in range(3):
        o = [j for j in range(3) if j != s]
        x, a, b = G[:, s], G[:, o[0]], G[:, o[1]]
        out["identical"].append((w * ((x == a).astype(float) + (x == b)) / 2).sum())
        out["within1"].append((w * ((abs(x - a) <= 1).astype(float) + (abs(x - b) <= 1)) / 2).sum())
        out["amae"].append((w * (abs(x - a) + abs(x - b)) / 2).sum())
        out["rmse"].append(np.sqrt((w * ((x - a) ** 2 + (x - b) ** 2) / 2).sum()))
        q, r = [], []
        for _ in range(N_DRAWS):                  # QWK, Spearman: average over tie-breaks
            ref = np.where(rng.random(len(x)) < 0.5, a, b)
            m = rating_metrics(ref, x, weighted=True, w=w)
            q.append(m["qwk"]); r.append(m["spearman_r"])
        out["qwk"].append(np.mean(q)); out["spearman"].append(np.mean(r))
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["icc21"], res["icc23"] = icc2(G.astype(float))
    return res


def always3(label: np.ndarray) -> dict:
    m = rating_metrics(label, np.full(len(label), 3), weighted=True, w=label_weights(label))
    return {"qwk": 0.0, "spearman": 0.0, "within1": m["within1"], "amae": m["mae"], "rmse": m["rmse"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-out-of-scale", action="store_true",
                    help="drop the two segments with a grade outside 1-5 instead of clipping them")
    a = ap.parse_args()
    d = pd.read_excel(LABELS, "ALL")
    S = list(d.columns[4:7])
    d = d[d[S].notna().all(axis=1)].copy()
    d["group"] = groups_of(d.Identifier.str.split("__").str[0])
    bad = ~d[S].isin([1, 2, 3, 4, 5]).all(axis=1)
    if a.exclude_out_of_scale:
        print(f"excluding {bad.sum()} segment(s) with a grade outside 1-5: {', '.join(d.loc[bad, 'Identifier'])}")
        d = d[~bad]
    else:
        # two gradings lie outside the 1-5 scale (a 0 and a 7); clipping them into it reproduces the
        # published tab:reference exactly (checked 2026-10-05), so it is the default
        print(f"clipping {bad.sum()} segment(s) with a grade outside 1-5 into the scale: "
              f"{', '.join(d.loc[bad, 'Identifier'])}")
        d[S] = d[S].clip(1, 5)
    rng = np.random.default_rng(SEED)
    rows = []
    for name, sub in [*((f"G{int(g)}", s) for g, s in d.groupby("group")), ("All", d)]:
        G, lab = sub[S].to_numpy(int), sub.Etiqueta.to_numpy(int)
        rows.append({"group": name, "segments": len(sub), **fresh_expert(G, lab, rng),
                     **{f"always3_{k}": v for k, v in always3(lab).items()}})
    t = pd.DataFrame(rows).set_index("group")
    pd.set_option("display.width", 220)
    print(t.round(3).to_string())
    out = RESULTS / ("human_reference" + ("_excluded" if a.exclude_out_of_scale else "") + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
