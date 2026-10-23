"""Exhaustive search for the 5 road groups of the nested road-group design (design B).

Every partition of the 13 roads into 5 non-empty groups is scored on two criteria:
  k_min       the smallest per-group k, where k = count of the scarcest grade in the group.
              A group's balanced val/test set has 5*k tiles, so k_min sets the worst case.
  train_range the spread of training-set sizes over the 20 (test, val) cells.
              Training = all tiles minus the test and validation groups, so the spread is
              (sum of the two largest groups) - (sum of the two smallest groups).
Both criteria conflict, so the script writes the whole Pareto front (no partition is better on
both) and, for each k_min level, the partitions with the smallest train_range.

S(13,5) = 7,508,501 partitions; runs in about 40 s (the partition matrices take a few GB of RAM).
Paper: appendix "Road-group partition search" (trade-off table, adopted partition at k_min = 13).
Output: results/partition_search/{pareto.csv, best_per_kmin.csv, per_road_counts.csv}

    python scripts/rating/nested_cv/partition_search.py
"""
from pathlib import PureWindowsPath

import numpy as np
import pandas as pd

from scripts.rating.nested_cv.common import POOL_CSV, RESULTS

OUT = RESULTS / "partition_search"
N_GROUPS = 5
TOP_PER_K = 20        # partitions kept per k_min level in best_per_kmin.csv


def load_counts() -> pd.DataFrame:
    """Tiles per road (rows) and grade 1..5 (columns), from the 5,354 labelled segments."""
    df = pd.read_csv(POOL_CSV)
    assert not df.filepath.duplicated().any()
    df["road"] = df.filepath.map(lambda p: PureWindowsPath(p).parts[1])
    tab = pd.crosstab(df.road, df.rating).reindex(columns=range(1, 6), fill_value=0)
    assert tab.values.sum() == 5354 and tab[5].sum() == 79, "label file changed"
    return tab


def partitions(n: int, k: int):
    """All set partitions of range(n) into exactly k blocks, as restricted-growth lists."""
    a = [0] * n

    def rec(i: int, used: int):
        if n - i < k - used:                      # not enough roads left to open the blocks
            return
        if i == n:
            if used == k:
                yield a
            return
        for b in range(min(used + 1, k)):
            a[i] = b
            yield from rec(i + 1, max(used, b + 1))

    yield from rec(0, 0)


def main() -> None:
    tab = load_counts()
    roads = list(tab.index)
    X = tab.values                                    # roads x 5 grades
    n_all = X.sum()

    # enumerate once into an int8 matrix: partitions x roads
    labels = np.array([p.copy() for p in partitions(len(roads), N_GROUPS)], dtype=np.int8)
    print(f"{len(labels):,} partitions of {len(roads)} roads into {N_GROUPS} groups")

    # per-group grade counts: partitions x groups x grades
    onehot = (labels[:, :, None] == np.arange(N_GROUPS)).astype(np.int32)   # P x R x G
    grp = np.einsum("prg,rc->pgc", onehot, X)
    k = grp.min(axis=2)                                                      # P x G
    tot = grp.sum(axis=2)
    srt = np.sort(tot, axis=1)
    train_min = n_all - srt[:, -1] - srt[:, -2]
    train_max = n_all - srt[:, 0] - srt[:, 1]

    res = pd.DataFrame({
        "k_min": k.min(1),
        "train_range": train_max - train_min,
        "train_min": train_min, "train_max": train_max,
        "group_tiles_min": srt[:, 0], "group_tiles_max": srt[:, -1],
        "k_sum": k.sum(1),                            # tie-break: more balanced tiles overall
    })

    def describe(i: int) -> dict:
        """Groups ordered by grade-5 count, descending, so G1 is the grade-5-richest."""
        order = np.argsort(-grp[i, :, 4], kind="stable")
        d = {}
        for j, g in enumerate(order, 1):
            d[f"G{j}"] = " + ".join(r for r, l in zip(roads, labels[i]) if l == g)
            d[f"G{j}_grades"] = "/".join(map(str, grp[i, g]))
            d[f"G{j}_k"] = int(k[i, g])
        return d

    # Pareto front: maximise k_min, minimise train_range
    best = res.groupby("k_min")["train_range"].min().sort_index(ascending=False)
    front, cur = [], np.inf
    for km, tr in best.items():
        if tr < cur:
            front.append(km)
            cur = tr
    rows = []
    for km in front:
        idx = res.index[(res.k_min == km) & (res.train_range == best[km])]
        for i in idx:
            rows.append({**res.loc[i].to_dict(), **describe(i)})
    pareto = pd.DataFrame(rows).sort_values(["k_min", "k_sum"], ascending=[False, False])

    # top partitions per k_min level (to see how flat the optimum is)
    rows = []
    for km in sorted(res.k_min.unique(), reverse=True)[:6]:
        sub = res[res.k_min == km].sort_values(["train_range", "k_sum"],
                                               ascending=[True, False]).head(TOP_PER_K)
        rows += [{**res.loc[i].to_dict(), **describe(i)} for i in sub.index]
    top = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    tab.assign(total=tab.sum(1)).to_csv(OUT / "per_road_counts.csv")
    pareto.to_csv(OUT / "pareto.csv", index=False)
    top.to_csv(OUT / "best_per_kmin.csv", index=False)

    print("\nPartitions per k_min level and their best training-size range:")
    print(res.groupby("k_min").agg(n=("train_range", "size"),
                                   best_train_range=("train_range", "min"))
             .sort_index(ascending=False).head(8).to_string())
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    cols = ["k_min", "train_range", "train_min", "train_max", "k_sum"] + \
           [c for j in range(1, 6) for c in (f"G{j}", f"G{j}_grades")]
    print("\nPareto front (k_min vs train_range):")
    print(pareto[cols].to_string(index=False))


if __name__ == "__main__":
    main()
