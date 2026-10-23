"""Why 5 road groups? Repeat the exhaustive partition search for G = 3..10 groups.

For each G it reports the Pareto front of partition_search.py (k_min vs training-size range),
plus what G implies for the design: G*(G-1) cells per configuration and the share of tiles a
run trains on (G-2 of G groups).
Paper: appendix "Road-group partition search" (best k_min as a function of the number of groups).
Output: results/partition_search/group_count_sweep.csv

    python scripts/rating/nested_cv/group_count_sweep.py
"""
import numpy as np
import pandas as pd

from scripts.rating.nested_cv.partition_search import OUT, load_counts, partitions

CHUNK = 500_000


def front_for(X: np.ndarray, n_groups: int) -> pd.DataFrame:
    n_all = X.sum()
    best = {}                                            # k_min -> (range, train_min, train_max)
    n_part = 0
    gen = partitions(len(X), n_groups)
    while True:
        chunk = [p.copy() for _, p in zip(range(CHUNK), gen)]
        if not chunk:
            break
        n_part += len(chunk)
        lab = np.array(chunk, dtype=np.int8)
        onehot = (lab[:, :, None] == np.arange(n_groups)).astype(np.int32)   # P x roads x G
        grp = np.einsum("prg,rc->pgc", onehot, X)                            # P x G x grades
        kmin = grp.min(2).min(1)
        srt = np.sort(grp.sum(2), axis=1)
        tmin = n_all - srt[:, -1] - srt[:, -2]
        tmax = n_all - srt[:, 0] - srt[:, 1]
        rng = tmax - tmin
        for km in np.unique(kmin):
            m = kmin == km
            i = np.flatnonzero(m)[rng[m].argmin()]
            if km not in best or rng[i] < best[km][0]:
                best[km] = (int(rng[i]), int(tmin[i]), int(tmax[i]))
    rows, cur = [], np.inf
    for km in sorted(best, reverse=True):              # Pareto: keep only range improvements
        if best[km][0] < cur:
            cur = best[km][0]
            rows.append({"groups": n_groups, "partitions": n_part, "k_min": int(km),
                         "balanced_min": 5 * int(km), "train_range": best[km][0],
                         "train_min": best[km][1], "train_max": best[km][2]})
    return pd.DataFrame(rows)


def main() -> None:
    X = load_counts().values
    out = []
    for g in range(3, 11):
        f = front_for(X, g)
        f["cells_per_config"] = g * (g - 1)
        f["train_share_mean"] = round((g - 2) / g, 2)
        print(f.to_string(index=False), flush=True)
        out.append(f)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(out).to_csv(OUT / "group_count_sweep.csv", index=False)


if __name__ == "__main__":
    main()
