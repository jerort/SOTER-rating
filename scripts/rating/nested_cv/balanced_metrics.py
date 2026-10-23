"""The five rating metrics, natural or grade-weighted, from true and predicted grades.

natural   every tile counts once.
weighted  every tile counts 1 / (tiles of its grade), so every grade counts equally.
          MAE becomes AMAE (Gutierrez et al. 2016); RMSE averages the squared error within
          each grade, then across grades, then takes the root; QWK is computed on the
          weighted confusion matrix; Spearman on ranks and moments under the weights.
With uniform weights every formula reduces to the ordinary metric. "within1" is the paper's
adjacent accuracy and "mae" with weighted=True is its AMAE.

T and P are integer grades 1..5 (for the BONN raters, P = the grade given by the learned
thresholds).
"""
import numpy as np

METRICS = ["qwk", "spearman_r", "within1", "mae", "rmse"]
_W = (np.subtract.outer(np.arange(5), np.arange(5)) ** 2) / 16.0     # QWK disagreement weights


def bonn_cost(T: np.ndarray, z: np.ndarray, thresholds: np.ndarray, sigma: float = 0.3883,
              weighted: bool = True) -> float:
    """BONN's loss written as an expected cost in grades (Lazaro & Figueiras-Vidal 2023).

    Each segment's output z is smoothed with a Gaussian of width sigma (the Parzen window);
    the probability of deciding grade d is the smoothed mass between thresholds u_{d-1} and
    u_d, and the cost of deciding d when the truth is t is |d - t|. The result is a smoothed
    MAE (weighted=False: every segment once) or a smoothed AMAE (weighted=True: every grade
    one fifth, the quantity BONN minimises in "amae" mode). It equals the logged BONN loss
    plus the constant sum_t pi_t |5 - t| that the training code omits."""
    from scipy.stats import norm
    T = np.asarray(T, int)
    u = np.concatenate([[-np.inf], np.asarray(thresholds, float), [np.inf]])
    F = norm.cdf((u[None, :] - np.asarray(z, float)[:, None]) / sigma)        # [N, 6]
    P = np.diff(F, axis=1)                                                      # [N, 5]
    cost = (P * np.abs(np.arange(1, 6)[None, :] - T[:, None])).sum(1)          # [N]
    w = grade_weights(T) if weighted else np.full(len(T), 1.0 / len(T))
    return float((w * cost).sum())


def grade_weights(T: np.ndarray) -> np.ndarray:
    """1 / (count of the tile's grade), normalised to sum to 1."""
    n = np.bincount(T, minlength=6)[1:]
    w = 1.0 / n[T - 1]
    return w / w.sum()


def rating_metrics(T: np.ndarray, P: np.ndarray, weighted: bool, w: np.ndarray | None = None) -> dict:
    """w: explicit per-tile weights (normalised here), overriding `weighted`; used for the
    fresh-expert reference, whose weights come from the published label rather than from T."""
    T, P = np.asarray(T, int), np.asarray(P, int)
    if w is not None:
        w = np.asarray(w, float) / np.sum(w)
    else:
        w = grade_weights(T) if weighted else np.full(len(T), 1.0 / len(T))
    err = np.abs(T - P)

    O = np.zeros((5, 5))                                  # (weighted) confusion matrix
    np.add.at(O, (T - 1, P - 1), w)
    E = np.outer(O.sum(1), O.sum(0))
    den = (_W * E).sum()
    qwk = 1 - (_W * O).sum() / den if den > 0 else np.nan

    def centred_rank(x):                                  # mid-ranks under the weights
        r = np.empty(len(x))
        for v in np.unique(x):
            m = x == v
            r[m] = w[x < v].sum() + w[m].sum() / 2
        return r - (w * r).sum()
    a, b = centred_rank(T), centred_rank(P)
    sd = np.sqrt((w * a * a).sum() * (w * b * b).sum())
    rho = (w * a * b).sum() / sd if sd > 0 else np.nan

    return {"qwk": float(qwk), "spearman_r": float(rho),
            "within1": float((w * (err <= 1)).sum()),
            "mae": float((w * err).sum()),
            "rmse": float(np.sqrt((w * err ** 2).sum()))}
