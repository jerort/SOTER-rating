r"""Figures of the paper on the rating design and results (internally "tanda 7", hence fig_t7_*).

    python scripts/rating/nested_cv/paper_figures.py [OUT_DIR] [RESULTS_DIR]

Reads the nested cross-validation results: RESULTS_DIR/summary/picks.csv and the curves CSV of
one search in RESULTS_DIR (default RESULTS_DIR: results/nested_cv, the paper's run; pass
scripts/rating/nested_cv/runs for a run of your own). Writes vector PDFs (Arial embedded as
TrueType, no mathtext) to OUT_DIR, by default results/figures:
  fig_t7_grid.pdf        the 5 x 5 road-group grid, one outer fold highlighted
  fig_t7_search.pdf      one outer fold: Bayesian search over the inner splits, refit, test
  fig_t7_trial.pdf       how one trial is scored (real validation curves)
  fig_t7_references.pdf  how the two reference points are obtained
  fig_t7_picks.pdf       the hyperparameters picked by the 30 searches
  fig_t7_test.pdf        test scores per road group, next to the two reference points
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]          # repository root
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "figures"
T7 = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "nested_cv"
PICKS = T7 / "summary" / "picks.csv"

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})

INPUTS = ["30", "15-otf", "15-paired"]
COL = {"30": "#0072B2", "15-otf": "#E69F00", "15-paired": "#009E73"}       # Okabe-Ito
NAME = {"30": "native", "15-otf": "otf", "15-paired": "paired"}           # input names in the paper
BACKBONE = {"E04": "DenseNet-201", "E11": "ViT-B/16"}
MARK = {"E04": "o", "E11": "^"}
GROUPS = [1, 2, 3, 4, 5]
# the two reference points (manuscript, Appendix / tab:reference-metrics), grade-weighted
EXPERT = {"qwk": [0.730, 0.782, 0.627, 0.813, 0.784, 0.752],
          "within1": [0.896, 0.937, 0.819, 0.958, 0.953, 0.915],
          "mae": [0.695, 0.617, 0.855, 0.592, 0.604, 0.663]}
NO_SKILL = {"qwk": 0.0, "within1": 0.600, "mae": 1.200}
GREY, DARK = "#8C8C8C", "#333333"
BLUE_BG, ORANGE_BG, RED_BG = "#DCEAF6", "#FBE7C6", "#F6D5D1"


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def box(ax, x, y, w, h, text, fc, ec=DARK, size=7.5, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.7))
    lines = text.split("\n")
    if bold_first:
        ax.text(x + w / 2, y + h - 0.035, lines[0], ha="center", va="top", fontsize=size,
                fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.04, "\n".join(lines[1:]), ha="center", va="center",
                fontsize=size - 0.5, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                linespacing=1.25)


def arrow(ax, p, q, rad=0.0, text=None, color=DARK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, text, ha="center", va="center", fontsize=7)


# ------------------------------------------------------------------ 1. grid
def fig_grid():
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    ax.set_xlim(-1.3, 7.9)
    ax.set_ylim(-0.9, 5.9)
    ax.axis("off")
    for i, t in enumerate(GROUPS):              # rows: test group
        y = 4 - i
        ax.text(-0.25, y + 0.42, f"G{t}", ha="right", va="center", fontsize=9, fontweight="bold",
                color="#B03A2E")
        for j, v in enumerate(GROUPS):          # columns: validation group
            x = j * 1.1
            if t == v:
                ax.add_patch(Rectangle((x, y), 1.0, 0.85, fc="white", ec="#CCCCCC", lw=0.6,
                                       hatch="////"))
                continue
            hl = t == 1
            ax.add_patch(FancyBboxPatch((x, y), 1.0, 0.85, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=ORANGE_BG if hl else "white",
                                        ec="#B07800" if hl else "#BBBBBB", lw=1.2 if hl else 0.6))
            train = "·".join(str(g) for g in GROUPS if g not in (t, v))
            ax.text(x + 0.5, y + 0.42, f"train\n{train}", ha="center", va="center", fontsize=7,
                    color=DARK)
    for j, v in enumerate(GROUPS):
        ax.text(j * 1.1 + 0.5, 5.05, f"G{v}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#1F5F99")
    ax.text(2.7, 5.55, "validation group (inner loop) →", ha="center", fontsize=8.5,
            color="#1F5F99", fontweight="bold")
    ax.text(-1.05, 2.4, "test group (outer loop) →", rotation=90, ha="center", va="center",
            fontsize=8.5, color="#B03A2E", fontweight="bold")
    ax.plot([5.5, 5.6, 5.6, 5.5], [4.0, 4.0, 4.85, 4.85], color="#B07800", lw=1.0)
    ax.text(5.75, 4.42, "outer fold G1:\nthe search uses these 4 splits;\nthen one refit on G2–G5,\n"
            "tested once on G1", ha="left", va="center", fontsize=7.5, color="#7A5300")
    ax.text(0.0, -0.45, "Hatched: impossible (a group cannot be test and validation at once). "
            "Each of the 20 splits has its own seed.", fontsize=7, color="#555555")
    save(fig, "fig_t7_grid.pdf")


# ------------------------------------------------------------------ 2. search
def fig_search():
    """Snake layout: the search loop on top (left to right), then pick, refit and test
    underneath (right to left), so that no arrow crosses a box."""
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.axis("off")
    w, h, gap = 0.215, 0.32, 0.035
    xs = [0.01 + k * (w + gap) for k in range(4)]
    y1, y2 = 0.60, 0.06
    top = ["1  Propose\nthe Bayesian optimiser\n(TPE) suggests a\nlearning rate and\na weight decay",
           "2  Train\n4 models, one per\ninner split (3 groups\ntrain, 1 validates);\n"
           "40 checks in 10 epochs",
           "3  Score each check\nworst validation QWK\nof the 4; counts only if\nevery group has adjacent\n"
           "accuracy ≥ 0.600, QWK > 0",
           "4  Report\ntrial score = best\ncounting check; its\nposition E sets the\n"
           "training length"]
    for x, t in zip(xs, top):
        box(ax, x, y1, w, h, t, BLUE_BG, size=7.2)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w + 0.003, y1 + h / 2), (b - 0.003, y1 + h / 2))
    arrow(ax, (xs[3] + w / 2, y1 + h + 0.01), (xs[0] + w / 2, y1 + h + 0.01), rad=0.12)
    ax.text(0.5, 1.065, "repeat: 50 trials per search (the first 10 at random)", ha="center",
            va="top", fontsize=7.5, style="italic")
    bottom = [(xs[3], "5  Pick\nthe best trial:\nits learning rate,\nweight decay and E", BLUE_BG),
              (xs[2], "6  Refit\none model on the\n4 groups together\n(G2–G5) for E checks",
               ORANGE_BG),
              (xs[1], "7  Test\nscore it once on G1,\nthe group never used\nin steps 1–6", RED_BG)]
    for x, t, fc in bottom:
        box(ax, x, y2, w, h, t, fc, size=7.2)
    arrow(ax, (xs[3] + w / 2, y1 - 0.005), (xs[3] + w / 2, y2 + h + 0.005))
    for (a, _, _), (b, _, _) in zip(bottom[:-1], bottom[1:]):
        arrow(ax, (a - 0.003, y2 + h / 2), (b + w + 0.003, y2 + h / 2))
    ax.text(xs[0] + w / 2, y2 + h / 2, "for one outer fold\n(test group G1).\n\nRepeated for\n"
            "5 test groups ×\n2 backbones ×\n3 inputs\n= 30 searches", ha="center", va="center",
            fontsize=7, color="#555555")
    ax.text(xs[2] - gap / 2, y1 - 0.045, "inner loop: training and validation groups only",
            ha="center", va="center", fontsize=7, color="#1F5F99")
    save(fig, "fig_t7_search.pdf")


# ------------------------------------------------------------------ 3. trial
def fig_trial(exp="E11", leg="15-otf", t=3):
    picks = pd.read_csv(PICKS)
    pk = picks[(picks.exp == exp) & (picks.leg == leg) & (picks.test_group == t)].iloc[0]
    c = pd.read_csv(T7 / f"curves_{exp}_{leg.replace('-', '_')}_test{t}.csv")
    c = c[c.trial == pk.trial]
    groups = sorted(c.val_group.unique())
    Q = np.array([c[c.val_group == g].sort_values("check").val_qwk.to_numpy() for g in groups])
    W = np.array([c[c.val_group == g].sort_values("check").val_within1.to_numpy() for g in groups])
    ep = c[c.val_group == groups[0]].sort_values("check").epoch.to_numpy()
    worst = Q.min(0)
    ok = (W.min(0) >= 0.600 - 1e-9) & (Q.min(0) > 1e-6)
    e = int(pk.check) - 1
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 3.9), sharex=True,
                                 gridspec_kw={"height_ratios": [1.6, 1]})
    shades = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    for k, g in enumerate(groups):
        a1.plot(ep, Q[k], color=shades[k], lw=0.9, label=f"validation G{g}")
        a2.plot(ep, W[k], color=shades[k], lw=0.9)
    a1.plot(ep, worst, color="black", lw=2.0, label="worst group = check score")
    for ax in (a1, a2):
        for i in np.flatnonzero(~ok):
            ax.axvspan(ep[i] - 0.125, ep[i] + 0.125, color="#EEEEEE", lw=0, zorder=0)
        ax.axvline(ep[e], color="#B03A2E", lw=1.0, ls="--")
    a1.plot(ep[e], worst[e], marker="*", color="#B03A2E", ms=11, zorder=5)
    a1.annotate(f"best counting check: E = {ep[e]:g} epochs,\ntrial score = {worst[e]:.3f}",
                (ep[e], worst[e]), xytext=(7.3, -0.22), textcoords="data", fontsize=7.5,
                color="#B03A2E", ha="left",
                arrowprops=dict(arrowstyle="-", color="#B03A2E", lw=0.6),
                bbox=dict(fc="white", ec="none", pad=1))
    a2.axhline(0.600, color=DARK, lw=0.8, ls=":")
    a2.text(0.15, 0.415, "dotted: 0.600, the level of the naive baseline", fontsize=7,
            bbox=dict(fc="white", ec="none", pad=1))
    a1.set_ylabel("validation QWK")
    a2.set_ylabel("validation\nadjacent accuracy")
    a2.set_xlabel("epoch (4 checks per epoch)")
    hs, _ = a1.get_legend_handles_labels()
    hs.append(plt.Rectangle((0, 0), 1, 1, fc="#EEEEEE", ec="none",
                            label="check does not count"))
    a1.legend(handles=hs, fontsize=7, frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_t7_trial.pdf")


# ------------------------------------------------------------------ 4. references
def fig_references():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.axis("off")
    a1.set_title("Human baseline (upper reference)", fontsize=8.5, fontweight="bold", loc="left")
    for i, (name, x) in enumerate((("Specialist A", 0.03), ("Specialist B", 0.36),
                                   ("Specialist C", 0.69))):
        box(a1, x, 0.68, 0.27, 0.18, name, "#E8F3E8" if i == 0 else "#F2F2F2", bold_first=False)
    box(a1, 0.43, 0.36, 0.50, 0.20, "consensus of B and C\n(a tie counts both grades, half each)",
        "#F2F2F2", bold_first=False, size=7)
    box(a1, 0.01, 0.02, 0.62, 0.20, "error of A against it, scored like a model;\n"
        "repeated for B and C, then averaged", "#E8F3E8", bold_first=False, size=7)
    arrow(a1, (0.495, 0.68), (0.62, 0.56))
    arrow(a1, (0.825, 0.68), (0.74, 0.56))
    arrow(a1, (0.165, 0.68), (0.165, 0.22))
    arrow(a1, (0.68, 0.36), (0.50, 0.22))
    grades = np.arange(1, 6)
    a2.bar(grades, np.abs(grades - 3), color="#D9D9D9", ec=DARK, lw=0.6, width=0.6)
    for g in grades:
        a2.text(g, abs(g - 3) + 0.06, f"{abs(g - 3)}", ha="center", fontsize=7.5)
    a2.set_title("Naive baseline (lower reference)", fontsize=8.5, fontweight="bold", loc="left")
    a2.set_xlabel("true grade")
    a2.set_ylabel("error of 'always grade 3'")
    a2.set_ylim(0, 3.1)
    a2.set_xticks(grades)
    a2.text(0.5, 0.98, "each grade weighs 1/5: QWK 0, AMAE 1.200,\nadjacent accuracy 0.600 (3 of 5 grades)",
            transform=a2.transAxes, ha="center", va="top", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_t7_references.pdf")


# ------------------------------------------------------------------ 5. picks
def fig_picks():
    p = pd.read_csv(PICKS)
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.add_patch(Rectangle((1e-5, 0.01), 1e-2 - 1e-5, 1000 - 0.01, fc="#F7F7F7", ec="#BBBBBB",
                           lw=0.6, ls="--", zorder=0))
    for (exp, leg), x in p.groupby(["exp", "leg"]):
        ax.scatter(x.p_lr, x.p_wd, s=12 + 9 * x.epoch, marker=MARK[exp], c=COL[leg],
                   ec="black", lw=0.4, alpha=0.9, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6e-6, 1.6e-2)
    ax.set_ylim(5e-3, 2e3)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("weight decay (head)")
    ax.text(1.1e-5, 1.3e3, "search space", fontsize=7, color="#777777", va="top")
    hs = [plt.Line2D([], [], ls="", marker=MARK[e], mfc="white", mec="black", label=BACKBONE[e])
          for e in MARK]
    hs += [plt.Line2D([], [], ls="", marker="s", mfc=COL[l], mec="none", label=NAME[l]) for l in INPUTS]
    hs += [plt.Line2D([], [], ls="", marker="o", mfc="white", mec="black", ms=np.sqrt(12 + 9 * e),
                      label=f"{e} epoch{'s' if e > 1 else ''}") for e in (1, 5, 10)]
    ax.legend(handles=hs, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig_t7_picks.pdf")


# ------------------------------------------------------------------ 6. test
def fig_test():
    p = pd.read_csv(PICKS)
    cands = [(e, l) for e in ("E04", "E11") for l in INPUTS]
    metrics = [("test_qwk", "qwk", "QWK\n(higher is better)"),
               ("test_within1", "within1", "adjacent accuracy\n(higher is better)"),
               ("test_mae", "mae", "AMAE\n(lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 3.0))
    rng = np.random.default_rng(0)
    for ax, (col, key, title) in zip(axes, metrics):
        lo, hi = min(EXPERT[key][:5]), max(EXPERT[key][:5])
        ax.axhspan(lo, hi, color="#E8F3E8", lw=0, zorder=0)
        ax.axhline(EXPERT[key][5], color="#2E7D32", lw=1.0)
        ax.axhline(NO_SKILL[key], color=GREY, lw=1.0, ls="--")
        for i, (e, l) in enumerate(cands):
            x = p[(p.exp == e) & (p.leg == l)].sort_values("test_group")
            xi = i + (0.4 if e == "E11" else 0)
            ax.scatter(xi + rng.uniform(-0.12, 0.12, len(x)), x[col], s=14, marker=MARK[e],
                       c=COL[l], ec="black", lw=0.3, zorder=3)
            ax.hlines(x[col].mean(), xi - 0.25, xi + 0.25, color="black", lw=1.6, zorder=4)
        ax.set_xticks([1, 4.4])
        ax.set_xticklabels(["DenseNet-201", "ViT-B/16"])
        ax.set_title(title, fontsize=8)
        ax.set_xlim(-0.6, 6.0)
    axes[0].text(5.9, EXPERT["qwk"][5] + 0.01, "human baseline", ha="right", va="bottom",
                 fontsize=6.5, color="#2E7D32")
    axes[0].text(5.9, 0.01, "naive baseline", ha="right", va="bottom", fontsize=6.5, color=GREY)
    hs = [plt.Line2D([], [], ls="", marker="s", mfc=COL[l], mec="none", label=f"{NAME[l]} (satellite view)")
          for l in INPUTS]
    hs += [plt.Line2D([], [], color="black", lw=1.6, label="mean of 5 test groups"),
           plt.Line2D([], [], color="#2E7D32", lw=1.0, label="human baseline (roof-mounted camera view)"),
           plt.Line2D([], [], color=GREY, lw=1.0, ls="--", label="naive baseline")]
    fig.legend(handles=hs, fontsize=7, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig_t7_test.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_grid()
    fig_search()
    fig_trial()
    fig_references()
    fig_picks()
    fig_test()
