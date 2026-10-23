r"""Figures of the paper on the rating design and results (internally "tanda 7", hence fig_t7_*).

    python scripts/rating/nested_cv/paper_figures.py [OUT_DIR] [RESULTS_DIR]

Reads the nested cross-validation results: RESULTS_DIR/summary/picks.csv and the curves CSV of
one search in RESULTS_DIR (default RESULTS_DIR: results/nested_cv, the paper's run; pass
scripts/rating/nested_cv/runs for a run of your own). Writes vector PDFs (Computer Modern,
cmr10 with TeX math, as in the paper, embedded as TrueType, drawn at the text width) to
OUT_DIR, by default results/figures:
  fig_t7_grid.pdf                the 5 x 5 road-group grid, one outer fold highlighted
  fig_t7_search.pdf              one outer fold: Bayesian search over the inner splits, refit, test
  fig_t7_trial.pdf               how one trial is scored (real validation curves)
  fig_design_weighting_a/b.pdf   segments per grade on G2, and each grade's share once weighted
  fig_t7_references_a/b.pdf      how the human and the naive baselines are obtained
  fig_t7_picks.pdf               the hyperparameters picked by the 30 searches
  fig_t7_test_a/b.pdf            test QWK and adjacent accuracy per road group, next to both baselines
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parents[3]          # repository root
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "figures"
T7 = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "nested_cv"
PICKS = T7 / "summary" / "picks.csv"

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
                     "font.size": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
TW = 4.77          # text width of the paper in inches (345 pt): figures are drawn at print size
SC = 3.54          # single column of the journal, 90 mm

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
    """The 5 x 5 road-group grid: rows are outer folds (test group), columns inner splits
    (validation group), and a last column with the refit of each outer fold. Every cell shows its
    training groups and its seed: inner split (t, v) uses seed 0..19 in row order, the refit of
    test group t uses seed 100 + t (run_pair_sweep.seed_of, tanda7.REFIT_SEED)."""
    fig, ax = plt.subplots(figsize=(TW, 2.75))
    ax.set_xlim(-1.3, 7.3)
    ax.set_ylim(-0.2, 5.9)
    ax.axis("off")
    cells = [(t, v) for t in GROUPS for v in GROUPS if t != v]
    xr = 5 * 1.1 + 0.35                         # x of the refit column
    for i, t in enumerate(GROUPS):              # rows: test group
        y = 4 - i
        hl = t == 1
        ax.text(-0.25, y + 0.42, f"G{t}", ha="right", va="center", fontsize=8, fontweight="bold",
                color="#B03A2E")
        for j, v in enumerate(GROUPS):          # columns: validation group
            x = j * 1.1
            if t == v:
                ax.add_patch(Rectangle((x, y), 1.0, 0.85, fc="white", ec="#CCCCCC", lw=0.6,
                                       hatch="////"))
                continue
            ax.add_patch(FancyBboxPatch((x, y), 1.0, 0.85, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=ORANGE_BG if hl else "white",
                                        ec="#B07800" if hl else "#BBBBBB", lw=1.2 if hl else 0.6))
            train = r"\cdot".join(str(g) for g in GROUPS if g not in (t, v))
            ax.text(x + 0.5, y + 0.55, f"${train}$", ha="center", va="center", fontsize=7,
                    color=DARK)
            ax.text(x + 0.5, y + 0.22, f"seed {cells.index((t, v))}", ha="center", va="center",
                    fontsize=6, color="#777777")
        ax.add_patch(FancyBboxPatch((xr, y), 1.25, 0.85, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=ORANGE_BG if hl else "#F2F2F2",
                                    ec="#B07800" if hl else "#BBBBBB", lw=1.2 if hl else 0.6))
        train = r"\cdot".join(str(g) for g in GROUPS if g != t)
        ax.text(xr + 0.625, y + 0.55, f"${train}$", ha="center", va="center", fontsize=7,
                color=DARK)
        ax.text(xr + 0.625, y + 0.22, f"seed {100 + t}", ha="center", va="center", fontsize=6,
                color="#777777")
    for j, v in enumerate(GROUPS):
        ax.text(j * 1.1 + 0.5, 5.05, f"G{v}", ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#1F5F99")
    ax.text(xr + 0.625, 5.05, "refit", ha="center", va="bottom", fontsize=8, fontweight="bold",
            color=DARK)
    ax.text(2.7, 5.55, "validation group $G^{\mathrm{val}}$ (inner loop) $\\rightarrow$", ha="center", fontsize=8,
            color="#1F5F99", fontweight="bold")
    ax.text(-1.05, 2.4, "test group $G^{\mathrm{test}}$ (outer loop) $\\rightarrow$", rotation=90, ha="center", va="center",
            fontsize=8, color="#B03A2E", fontweight="bold")
    save(fig, "fig_t7_grid.pdf")


# ------------------------------------------------------------------ 2. search
def trial_scores(c):
    """Score of every trial as the search computed it: at each check the worst of the four
    validation QWKs, counting only the checks where every group beats the naive baseline; the
    trial score is its best counting check (NaN if no check counts)."""
    out = {}
    for tr, d in c.groupby("trial"):
        Q = d.pivot(index="check", columns="val_group", values="val_qwk")
        W = d.pivot(index="check", columns="val_group", values="val_within1")
        ok = (W.min(axis=1) >= 0.600 - 1e-9) & (Q.min(axis=1) > 1e-6)
        s = Q.min(axis=1)[ok]
        out[tr] = s.max() if len(s) else np.nan
    return pd.Series(out).sort_index()


def fig_search(exp="E11", leg="15-otf", t=3):
    """One outer fold as six small pictures, drawn from the search of the trial figure:
    propose, train and score repeat for 50 trials (top, left to right); pick, refit and test
    run once (bottom, right to left), so that no arrow crosses a card."""
    picks = pd.read_csv(PICKS)
    pk = picks[(picks.exp == exp) & (picks.leg == leg) & (picks.test_group == t)].iloc[0]
    c = pd.read_csv(T7 / f"curves_{exp}_{leg.replace('-', '_')}_test{t}.csv")
    best = int(pk.trial)
    inner = [g for g in GROUPS if g != t]
    params = c.groupby("trial")[["lr", "wd"]].first()
    scores = trial_scores(c)
    RED, BLUE, LIGHT = "#B03A2E", "#3182bd", "#D9D9D9"
    small = dict(labelsize=5.5, length=2, pad=1)

    fig, ax = plt.subplots(figsize=(TW, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.axis("off")
    w, h = 0.27, 0.40
    xs = [0.02, 0.365, 0.71]
    y1, y2 = 0.56, 0.02

    def card(x, y, title, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    fc=fc, ec=DARK, lw=0.6))
        ax.text(x + 0.012, y + h - 0.02, title, ha="left", va="top", fontsize=7, fontweight="bold")
        return ax.inset_axes([x + 0.08, y + 0.085, w - 0.105, h - 0.18], transform=ax.transData)

    def label(a, x, y):
        a.set_xlabel(x, fontsize=7, labelpad=1)
        sym = y.startswith("$")                    # a symbol reads better horizontal
        a.set_ylabel(y, fontsize=7.5 if sym else 7, labelpad=2 if sym else 1,
                     rotation=0 if sym else 90, ha="right" if sym else "center", va="center")
        a.tick_params(**small)

    # 1  propose: the settings tried before, and the one this trial tries
    a = card(xs[0], y1, "1  Propose a setting", BLUE_BG)
    prev = params.loc[params.index < best]
    a.scatter(prev.lr, prev.wd, s=9, c=GREY, ec="none")
    a.scatter([params.loc[best, "lr"]], [params.loc[best, "wd"]], marker="*", s=70, c=RED, zorder=3, clip_on=False)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(7e-6, 1.5e-2)
    a.set_ylim(7e-3, 1.5e3)
    label(a, "$\\eta$", "$\\lambda$")
    a.set_xticks([])
    a.set_yticks([])
    a.minorticks_off()

    # 2  train: one model per inner split, three groups train and one validates
    a = card(xs[1], y1, "2  Train 4 models", BLUE_BG)
    a.axis("off")
    a.set_xlim(-0.2, 4.0)
    a.set_ylim(-0.15, 4.45)
    for r, v in enumerate(inner):
        for j, g in enumerate(inner):
            a.add_patch(Rectangle((j, 3 - r), 0.85, 0.85, fc=BLUE if g == v else LIGHT, ec="white", lw=0.8))
    for j, g in enumerate(inner):
        a.text(j + 0.425, 3.85, f"G{g}", ha="center", va="bottom", fontsize=6.5)
    ax.legend(handles=[Patch(fc=LIGHT, label="train"), Patch(fc=BLUE, label="validate")], fontsize=6,
              frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(xs[1] + w / 2, y1 + 0.002),
              bbox_transform=ax.transData, handlelength=0.9, handletextpad=0.4, columnspacing=0.8)

    # 3  score: the four validation curves, the worst group at each check, the best counting check
    a = card(xs[2], y1, "3  Score the trial", BLUE_BG)
    d = c[c.trial == best]
    Q = d.pivot(index="check", columns="val_group", values="val_qwk")
    for k, g in enumerate(Q.columns):
        a.plot(Q.index, Q[g], color=["#9ecae1", "#6baed6", "#3182bd", "#08519c"][k], lw=0.7)
    a.plot(Q.index, Q.min(axis=1), color="black", lw=1.4)
    e = int(pk.check) - 1
    a.plot(Q.index[e], Q.min(axis=1).iloc[e], marker="*", color=RED, ms=8, zorder=5)
    a.set_ylim(top=float(Q.max().max()) + 0.17)
    a.legend(handles=[plt.Line2D([], [], color="black", lw=1.4, label="$s(c)$")], fontsize=6,
             frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.2)
    label(a, "$c$", "$\\kappa$")
    a.set_xticks([])
    a.set_yticks([])

    for p, q in zip(xs[:-1], xs[1:]):
        arrow(ax, (p + w + 0.004, y1 + h / 2), (q - 0.004, y1 + h / 2))
    arrow(ax, (xs[2] + w / 2, y1 + h + 0.008), (xs[0] + w / 2, y1 + h + 0.008), rad=0.12)
    ax.text(0.5, 1.09, "$\\times$ 50 trials", ha="center", va="top", fontsize=7)
    ax.text(0.5, (y1 + y2 + h) / 2, f"$G^{{\\mathrm{{test}}}}$ = G{t} is never used in steps 1 to 5", ha="center", va="center",
            fontsize=6.5, color="#1F5F99")
    arrow(ax, (xs[2] + w / 2, y1 - 0.004), (xs[2] + w / 2, y2 + h + 0.004))

    # 4  pick: the score of every trial, the best one
    a = card(xs[2], y2, "4  Pick the best trial", BLUE_BG)
    a.scatter(scores.index, scores.values, s=8, c=GREY, ec="none")
    a.plot(best, scores[best], marker="*", color=RED, ms=8, zorder=5)
    label(a, "trial", "$S$")
    a.set_xticks([])
    a.set_yticks([])

    # 5  refit: one model on the four groups together, for the picked training length
    a = card(xs[1], y2, "5  Refit on 4 groups", ORANGE_BG)
    a.axis("off")
    a.set_xlim(-0.2, 4.0)
    a.set_ylim(-1.4, 2.7)
    for j, g in enumerate(inner):
        a.add_patch(Rectangle((j, 1.2), 0.85, 0.85, fc=LIGHT, ec="white", lw=0.8))
        a.text(j + 0.425, 2.1, f"G{g}", ha="center", va="bottom", fontsize=6.5)
    a.annotate("", xy=(1.9, 0.45), xytext=(1.9, 1.22), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=DARK))
    a.add_patch(FancyBboxPatch((0.0, -1.25), 3.8, 1.6, boxstyle="round,pad=0.03", fc="white", ec=DARK, lw=0.6))
    a.text(1.9, -0.45, f"one model\n$e = {pk.epoch:g}$", ha="center", va="center", fontsize=7)

    # 6  test: scored once on the group held out
    a = card(xs[0], y2, f"6  Test once on G{t}", RED_BG)
    a.axis("off")
    a.set_xlim(0, 4.0)
    a.set_ylim(-1.4, 2.7)
    a.text(2.0, 2.1, f"G{t}", ha="center", va="bottom", fontsize=6.5)
    a.add_patch(Rectangle((1.575, 1.2), 0.85, 0.85, fc=RED, ec="white", lw=0.8))
    a.annotate("", xy=(2.0, 0.45), xytext=(2.0, 1.15), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=DARK))
    a.text(2.0, -0.2, f"$\\kappa^{{\\mathrm{{test}}}} = {pk.test_qwk:.3f}$", ha="center", va="center", fontsize=7.5)

    for p, q in zip(xs[::-1][:-1], xs[::-1][1:]):
        arrow(ax, (p - 0.004, y2 + h / 2), (q + w + 0.004, y2 + h / 2))
    save(fig, "fig_t7_search.pdf")


# ------------------------------------------------------------------ 3. trial
def fig_trial(exp="E11", leg="15-otf", t=3):
    """One trial and how it is scored, in the notation of the paper: kappa_val(c) and a_adj_val(c) are the
    QWK and the adjacent accuracy of validation group G_val at check c, s(c) = min kappa_val(c) the check score
    (thick, in the colour of the worst group), shaded checks are discarded (min a_adj_val(c) < 0.600 or
    s(c) <= 0), and the star is the best retained check: trial score S and training length e."""
    picks = pd.read_csv(PICKS)
    pk = picks[(picks.exp == exp) & (picks.leg == leg) & (picks.test_group == t)].iloc[0]
    c = pd.read_csv(T7 / f"curves_{exp}_{leg.replace('-', '_')}_test{t}.csv")
    c = c[c.trial == pk.trial]
    Q = c.pivot(index="check", columns="val_group", values="val_qwk")
    A = c.pivot(index="check", columns="val_group", values="val_within1")
    ep = Q.index.to_numpy()                                   # x: the check c (4 per epoch)
    s = Q.min(axis=1).to_numpy()
    worst = Q.to_numpy().argmin(axis=1)                       # index of the worst group per check
    keep = (A.min(axis=1).to_numpy() >= 0.600 - 1e-9) & (s > 1e-6)
    e = int(pk.check) - 1
    shades = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(TW, 3.1), sharex=True,
                                 gridspec_kw={"height_ratios": [1.6, 1]})
    for k, g in enumerate(Q.columns):
        a1.plot(ep, Q[g], color=shades[k], lw=0.8, label=f"G{g}")
        a2.plot(ep, A[g], color=shades[k], lw=0.8)
    for i in range(len(ep) - 1):                              # s(c), coloured by its worst group
        a1.plot(ep[i:i + 2], s[i:i + 2], color=shades[worst[i]], lw=2.6, solid_capstyle="round", zorder=4)
    for ax in (a1, a2):
        for i in np.flatnonzero(~keep):
            ax.axvspan(ep[i] - 0.5, ep[i] + 0.5, color="#EEEEEE", lw=0, zorder=0)
        ax.axvline(ep[e], color="#B03A2E", lw=1.0, ls="--")
    a1.plot(ep[e], s[e], marker="*", color="#B03A2E", ms=11, zorder=5)
    a1.annotate(f"$S = {s[e]:.3f}$ (G{Q.columns[worst[e]]})\n$c$ = {ep[e]:g}, $e$ = {ep[e] / 4:g}",
                (ep[e], s[e]), xytext=(29, -0.22), textcoords="data", fontsize=7,
                color="#B03A2E", ha="left",
                arrowprops=dict(arrowstyle="-", color="#B03A2E", lw=0.6),
                bbox=dict(fc="white", ec="none", pad=1))
    a1.axhline(0.0, color=DARK, lw=0.8, ls=":", zorder=1)          # naive baseline: kappa = 0
    a2.axhline(0.600, color=DARK, lw=0.8, ls=":")                 # and a_adj = 0.600
    a1.set_ylabel("$\\kappa$", rotation=0, ha="right", va="center", labelpad=6)
    a2.set_ylabel("$a_{\\mathrm{adj}}$", rotation=0, ha="right", va="center", labelpad=6)
    a2.set_xlabel("$c$")
    hs, _ = a1.get_legend_handles_labels()
    hs.append(plt.Line2D([], [], color=GREY, lw=2.6, label="$s(c)$"))
    hs.append(plt.Rectangle((0, 0), 1, 1, fc="#EEEEEE", ec="none",
                            label="$a_{\\mathrm{adj}} < 0.600$ in some group"))
    hs.append(plt.Line2D([], [], color=DARK, lw=0.8, ls=":", label="naive baseline"))
    a1.legend(handles=hs, fontsize=7, frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.0)
    save(fig, "fig_t7_trial.pdf")


# ------------------------------------------------------------------ grade weighting
def fig_weighting():
    """Grade weighting on group G2 (counts from the road-group table of the paper): (a) segments per
    grade, (b) each grade's share of the score once every segment weighs the inverse of its grade's
    count. One file per panel, single-column width."""
    counts = [20, 315, 570, 113, 18]                     # grades 1..5 in G2
    grades = np.arange(1, 6)
    for part, vals, fc, ec in (("a", counts, BLUE_BG, "#3182bd"), ("b", [20] * 5, ORANGE_BG, "#B07800")):
        fig, a = plt.subplots(figsize=(SC, 1.55))
        a.bar(grades, vals, width=0.7, color=fc, ec=ec, lw=0.6)
        for g, n, v in zip(grades, counts, vals):
            if part == "a":
                a.text(g, v + 12, f"{v}", ha="center", va="bottom", fontsize=7)
            else:
                a.text(g, v + 1, "20%", ha="center", va="bottom", fontsize=7)
                a.text(g, v + 8, f"$\\times 1/{n}$", ha="center", va="bottom", fontsize=6.5, color="#7A5300")
        a.set_ylim(0, 650 if part == "a" else 36)
        a.set_xticks(grades)
        a.set_xlabel("$y$")
        a.set_yticks([])
        a.spines["left"].set_visible(False)
        fig.tight_layout()
        save(fig, f"fig_design_weighting_{part}.pdf")


# ------------------------------------------------------------------ 4. references
def fig_references():
    """(a) how the human baseline is obtained, (b) the naive baseline. One file per panel."""
    fig, a1 = plt.subplots(figsize=(SC, 1.75))
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.axis("off")
    for i, (name, x) in enumerate((("Specialist A", 0.02), ("Specialist B", 0.37),
                                   ("Specialist C", 0.71))):
        box(a1, x, 0.74, 0.27, 0.2, name, "#E8F3E8" if i == 0 else "#F2F2F2", bold_first=False, size=7)
    box(a1, 0.40, 0.38, 0.58, 0.22, "consensus of B and C\n(a tie: both grades, half each)",
        "#F2F2F2", bold_first=False, size=7)
    box(a1, 0.0, 0.0, 0.66, 0.24, "A scored against it like a model,\n"
        "repeated for B and C, then averaged", "#E8F3E8", bold_first=False, size=7)
    arrow(a1, (0.505, 0.74), (0.62, 0.60))
    arrow(a1, (0.845, 0.74), (0.77, 0.60))
    arrow(a1, (0.155, 0.74), (0.155, 0.24))
    arrow(a1, (0.69, 0.38), (0.52, 0.24))
    save(fig, "fig_t7_references_a.pdf")

    fig, a2 = plt.subplots(figsize=(SC, 1.9))
    grades = np.arange(1, 6)
    hit = np.abs(grades - 3) <= 1
    a2.bar(grades, np.abs(grades - 3), color=np.where(hit, "#9ecae1", "#D9D9D9"), ec=DARK, lw=0.6,
           width=0.6)
    a2.axhline(1, color=DARK, lw=0.8, ls=":")
    for g in grades:
        a2.text(g, abs(g - 3) + 0.06, f"{abs(g - 3)}", ha="center", fontsize=7.5)
    a2.set_xlabel("$y$")
    a2.set_ylabel("$|3 - y|$")
    a2.set_ylim(0, 3.1)
    a2.set_xticks(grades)
    a2.text(0.03, 0.98, "$a_{\\mathrm{adj}} = 0.600$\n$|3 - y| \\leq 1$\nfor 3 of 5 grades", transform=a2.transAxes,
            ha="left", va="top", fontsize=7)
    a2.text(0.97, 0.98, "$\\kappa = 0$", transform=a2.transAxes, ha="right", va="top", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_t7_references_b.pdf")


# ------------------------------------------------------------------ 5. picks
def fig_picks():
    """Settings picked by the 30 searches, single-column width with the legend under the plot."""
    p = pd.read_csv(PICKS)
    fig, ax = plt.subplots(figsize=(SC, 3.3))
    ax.add_patch(Rectangle((1e-5, 0.01), 1e-2 - 1e-5, 1000 - 0.01, fc="#F7F7F7", ec="#BBBBBB",
                           lw=0.6, ls="--", zorder=0))
    for (exp, leg), x in p.groupby(["exp", "leg"]):
        ax.scatter(x.p_lr, x.p_wd, s=12 + 9 * x.epoch, marker=MARK[exp], c=COL[leg],
                   ec="black", lw=0.4, alpha=0.9, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6e-6, 1.6e-2)
    ax.set_ylim(5e-3, 2e3)
    ax.set_xlabel("$\\eta$")
    ax.set_ylabel("$\\lambda$", rotation=0, ha="right", va="center", labelpad=6)
    ax.text(1.4e-5, 6.0e2, "search space", fontsize=7, color="#777777", va="top")
    hs = [plt.Line2D([], [], ls="", marker=MARK[e], mfc="white", mec="black", label=BACKBONE[e])
          for e in MARK]
    hs += [plt.Line2D([], [], ls="", marker="s", mfc=COL[l], mec="none", label=NAME[l]) for l in INPUTS]
    hs += [plt.Line2D([], [], ls="", marker="o", mfc="white", mec="black", ms=np.sqrt(12 + 9 * e),
                      label=f"$e$ = {e}") for e in (1, 5, 10)]
    ax.legend(handles=hs, fontsize=7, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), columnspacing=1.2)
    save(fig, "fig_t7_picks.pdf")


# ------------------------------------------------------------------ 6. test
def fig_test():
    """(a) test QWK and (b) test adjacent accuracy per test group, one file each at single-column
    width; the legend goes with (b)."""
    p = pd.read_csv(PICKS)
    cands = [(e, l) for e in ("E04", "E11") for l in INPUTS]
    metrics = [("a", "test_qwk", "qwk", "$\\kappa^{\\mathrm{test}}$"),
               ("b", "test_within1", "within1", "$a_{\\mathrm{adj}}^{\\mathrm{test}}$")]
    rng = np.random.default_rng(0)
    for part, col, key, ylab in metrics:
        fig, ax = plt.subplots(figsize=(SC, 2.0 if part == "a" else 2.9))
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
        ax.set_xlim(-0.6, 6.0)
        ax.set_ylabel(ylab, rotation=0, ha="right", va="center", labelpad=6)
        if part == "a":
            ax.text(5.9, EXPERT[key][5] + 0.01, "human baseline", ha="right", va="bottom", fontsize=6.5,
                    color="#2E7D32")
            ax.text(5.9, NO_SKILL[key] + 0.005, "naive baseline", ha="right", va="bottom", fontsize=6.5, color=GREY)
        else:
            hs = [plt.Line2D([], [], ls="", marker="s", mfc=COL[l], mec="none",
                             label=f"{NAME[l]} (satellite view)") for l in INPUTS]
            hs += [plt.Line2D([], [], color="black", lw=1.6, label="mean of 5 test groups"),
                   plt.Line2D([], [], color="#2E7D32", lw=1.0, label="human baseline (vehicle-mounted camera view)"),
                   plt.Line2D([], [], color=GREY, lw=1.0, ls="--", label="naive baseline")]
            ax.legend(handles=hs, fontsize=6.5, frameon=False, ncol=1, loc="upper center",
                      bbox_to_anchor=(0.5, -0.14))
        fig.tight_layout()
        save(fig, f"fig_t7_test_{part}.pdf")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_grid()
    fig_search()
    fig_trial()
    fig_weighting()
    fig_references()
    fig_picks()
    fig_test()
