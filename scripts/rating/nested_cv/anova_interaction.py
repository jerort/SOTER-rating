# Backbone x input interaction for Table 12: F test of the interaction in the two-way analysis of
# variance with the test group as a block, the six combinations (candidates) with mean, SD and
# Tukey letters, and the pairwise tests of the leading candidate (Tukey and unadjusted).
# Usage: python scripts/rating/nested_cv/anova_interaction.py [picks.csv]
#        (default: results/nested_cv/summary/picks.csv, the paper's run)
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]          # repository root
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results/nested_cv/summary/picks.csv")
d["cand"] = d.exp.map({"E04": "DenseNet-201", "E11": "ViT-B/16"}) + " / " + \
    d.leg.map({"30": "native", "15-otf": "otf", "15-paired": "paired"})
d["bb"], d["inp"] = d.exp, d.leg
METRICS = {"test_qwk": "QWK", "test_within1": "Within one", "test_mae": "AMAE"}
LEAD = "ViT-B/16 / otf"


def rss(cols, y):
    X = pd.get_dummies(d[cols].astype(str), drop_first=True).astype(float)
    X.insert(0, "const", 1.0)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    r = y - X.to_numpy() @ beta
    return (r ** 2).sum(), len(y) - np.linalg.matrix_rank(X.to_numpy())


def letters(means, sig):
    # compact letter display: insert-and-absorb on the levels sorted best first
    groups = [set(means.index)]
    for a, b in sig:
        new = []
        for g in groups:
            if a in g and b in g:
                new += [g - {a}, g - {b}]
            else:
                new.append(g)
        groups = [g for g in new if not any(g < h for h in new)]
        groups = [g for i, g in enumerate(groups) if g not in groups[:i]]
    order = list(means.index)
    groups.sort(key=lambda g: min(order.index(x) for x in g))
    return {c: "".join(chr(97 + i) for i, g in enumerate(groups) if c in g) for c in order}


for col, name in METRICS.items():
    y = d[col].to_numpy()
    rss_add, df_add = rss(["bb", "inp", "test_group"], y)
    rss_full, df_full = rss(["cand", "test_group"], y)       # cell means = full interaction model
    F = (rss_add - rss_full) / (df_add - df_full) / (rss_full / df_full)
    print(f"\n== {name}: interaction F({df_add - df_full},{df_full}) = {F:.2f}, "
          f"p = {stats.f.sf(F, df_add - df_full, df_full):.3f}")
    mse, n = rss_full / df_full, 5
    g = d.groupby("cand")[col]
    m = g.mean().sort_values(ascending=(name == "AMAE"))
    k = len(m)
    sig = []
    for a, b in itertools.combinations(m.index, 2):
        q = abs(m[a] - m[b]) / np.sqrt(mse / n)
        if stats.studentized_range.sf(q, k, df_full) < 0.05:
            sig.append((a, b))
    let = letters(m, sig)
    for c in m.index:
        print(f"  {c:22s} {m[c]:.3f} ± {g.std()[c]:.3f} {let[c]}")
    for c in m.index:
        if c == LEAD:
            continue
        diff = m[LEAD] - m[c]
        q = abs(diff) / np.sqrt(mse / n)
        t = abs(diff) / np.sqrt(2 * mse / n)
        print(f"  {LEAD} vs {c:22s} diff {diff:+.3f}  Tukey p = {stats.studentized_range.sf(q, k, df_full):.3f}"
              f"  unadjusted p = {2 * stats.t.sf(t, df_full):.3f}")
