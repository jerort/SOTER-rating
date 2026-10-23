# Reproduce Table 12 of the paper (additive two-way ANOVA with the test group as block) and add
# Tukey HSD comparisons of the levels of each factor, for a compact letter display.
# Usage: python scripts/rating/nested_cv/anova_letters.py [picks.csv]
#        (default: results/nested_cv/summary/picks.csv, the paper's run; internal ids E04/E11 and
#        30/15-otf/15-paired are mapped to the paper's names below)
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]          # repository root
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results/nested_cv/summary/picks.csv")
d["backbone"] = d.exp.map({"E04": "DenseNet-201", "E11": "ViT-B/16"})
d["input"] = d.leg.map({"30": "native", "15-otf": "otf", "15-paired": "paired"})
METRICS = {"test_qwk": "QWK", "test_within1": "Within one", "test_mae": "AMAE"}

for col, name in METRICS.items():
    y = d[col].to_numpy()
    # additive model y = mu + backbone + input + group, fitted by least squares
    X = pd.get_dummies(d[["backbone", "input", "test_group"]].astype(str), drop_first=True).astype(float)
    X.insert(0, "const", 1.0)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    resid = y - X.to_numpy() @ beta
    df_e = len(y) - X.shape[1]
    mse = (resid ** 2).sum() / df_e
    print(f"\n== {name}: residual df {df_e}, MSE {mse:.5f}")
    for f in ("backbone", "input", "test_group"):
        # F for each factor in the balanced design: SS of level means
        m = d.groupby(f)[col].mean()
        n = d.groupby(f)[col].size().iloc[0]
        ss = n * ((m - y.mean()) ** 2).sum()
        k = len(m)
        F = ss / (k - 1) / mse
        print(f"  {f:10s} F({k - 1},{df_e}) = {F:.2f}  p = {stats.f.sf(F, k - 1, df_e):.3f}")
    for f in ("backbone", "input"):
        m = d.groupby(f)[col].agg(["mean", "std"])
        n = d.groupby(f)[col].size().iloc[0]
        k = len(m)
        print(f"  {f}: " + ", ".join(f"{i} {r['mean']:.3f}±{r['std']:.3f}" for i, r in m.iterrows()))
        for a, b in itertools.combinations(m.index, 2):
            q = abs(m.loc[a, "mean"] - m.loc[b, "mean"]) / np.sqrt(mse / n)
            p = stats.studentized_range.sf(q, k, df_e)
            print(f"    Tukey {a} vs {b}: diff {m.loc[a, 'mean'] - m.loc[b, 'mean']:+.3f}  p = {p:.3f}")
