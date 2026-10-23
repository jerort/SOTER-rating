r"""Nested cross-validation of the paper's six candidates (2 raters x 3 inputs) over 5 road groups.
(Internally the "seventh tanda", tanda7.py.)

Raters (internal ids, see common.py): E04 = DenseNet-201 (paper E01), E11 = ViT-B/16 (paper E02),
both a frozen ImageNet backbone + one linear output + BONN thresholds, trained on cached
features (head_features.py). Inputs: 30 (native), 15-otf (otf), 15-paired (paired).

Outer loop: each road group G1..G5 is the test group once. Inner loop: the 4 remaining groups
give 4 inner splits (3 groups train, 1 validates). For each (rater, input, test group) one
Bayesian search (Optuna TPE, 50 trials, the first 10 random) tunes:
  * learning rate 1e-5..1e-2 and AdamW weight decay on the head 0.01..1000, both log scale;
  * the training length: every run trains 10 epochs at a constant learning rate (no early
    stopping) and is checked 4 times per epoch, 40 checks.
TRIAL SCORE: at every check, the QWK (grade-weighted) of the WORST of the 4 validation groups.
A check counts only if EVERY validation group has within-1 (adjacent accuracy) >= 0.600 and
QWK > 0, the level of the naive baseline. The trial's score is its best counting check, whose
position is the training length E.
REFIT: the best trial's learning rate and weight decay, trained once on the 4 non-test groups
for E checks (seed 100 + t), then scored once on the test group.

    python scripts/rating/nested_cv/nested_cv.py run [--jobs 6]     every search (in parallel) + refits
    python scripts/rating/nested_cv/nested_cv.py status
    python scripts/rating/nested_cv/nested_cv.py summary            after the refits: runs/summary/
    python scripts/rating/nested_cv/nested_cv.py smoke              2 short searches + refits, scratch
(search / refit: --exp E11 --leg 30 --folds 1,3 to restrict.)

Needs the feature caches of head_features.py. Outputs go to runs/ (next to this file); the
outputs of the paper's run are in results/nested_cv/ at the repository root.
SAFE TO INTERRUPT: start it again and it continues; a trial cut off mid-way is run again with
the same parameters (a trial takes ~20 s on one CPU thread).
"""
import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState

from scripts.rating.nested_cv import common as cm
from scripts.rating.nested_cv import head_features as hf
from scripts.rating.nested_cv.balanced_metrics import bonn_cost, rating_metrics

BASE = cm.HERE / "runs"
EXPS = cm.RATERS                  # ["E04", "E11"]
# TPE seed of each search: base + 100 * input + test group
SAMPLER_BASE = {"E11": 7000, "E04": 4000}
INPUTS = cm.INPUTS                # ["30", "15-otf", "15-paired"]
EPOCHS = 10
CHECKS = 4                        # checks per epoch -> 40 checks per run
N_TRIALS = 50
N_STARTUP = 10                    # random trials before TPE takes over
MIN_WITHIN1 = 0.600
TOL = 1e-9                        # a constant prediction scores within-1 0.6 +- 1e-16
QWK_FLOOR = 1e-6                  # "QWK > 0": a collapsed model (QWK exactly 0) does not count
REFIT_SEED = cm.REFIT_SEED        # refit of test group t uses seed REFIT_SEED + t
SPACE = {"lr": FloatDistribution(1e-5, 1e-2, log=True),
         "wd": FloatDistribution(0.01, 1000.0, log=True)}


def name_of(exp, leg, t):
    return f"{exp}_{cm.leg_dir(leg)}_test{t}"


def db_path(exp, leg, t, base=BASE) -> Path:
    return base / f"optuna_{name_of(exp, leg, t)}.db"


def load_study_if_any(exp, leg, t, base=BASE):
    if not db_path(exp, leg, t, base).exists():
        return None
    try:
        return optuna.load_study(study_name=name_of(exp, leg, t),
                                 storage=f"sqlite:///{db_path(exp, leg, t, base).as_posix()}")
    except KeyError:
        return None


def attrs(tr) -> dict:
    """A trial's summary (score_curves), stored as ONE user attribute (one SQLite commit per
    attribute is slow on a hard disk)."""
    return tr.user_attrs.get("s", {})


def n_complete(exp, leg, t, base=BASE) -> int:
    study = load_study_if_any(exp, leg, t, base)
    return 0 if study is None else len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))


# ------------------------------------------------------------------ the trial score
def score_curves(curves: dict) -> dict:
    """curves: {validation group: per-check curves}. At each check: score = the worst group's QWK;
    the check counts if every group has within-1 >= 0.600 and QWK > 0. E = the counting check
    with the highest score. Without any counting check the trial is infeasible and E is the
    check closest to counting (for the record only).
    constraint (Optuna: <= 0 means feasible) = how far the check at E is from counting."""
    groups = sorted(curves)
    Q = np.nan_to_num(np.array([curves[g]["val_qwk"].to_numpy(float) for g in groups]))
    W = np.array([curves[g]["val_within1"].to_numpy(float) for g in groups])
    mean, sd = Q.mean(0), Q.std(0, ddof=1)
    score = Q.min(0)                                # the worst of the 4 validation groups
    viol = np.maximum(MIN_WITHIN1 - TOL - W.min(0), QWK_FLOOR - Q.min(0))
    ok = viol <= 0
    e = int(np.flatnonzero(ok)[np.argmax(score[ok])]) if ok.any() else int(np.argmin(viol))
    out = {"feasible": bool(ok.any()), "check": e + 1, "epoch": (e + 1) / CHECKS,
           "score": float(score[e]), "mean_qwk": float(mean[e]), "sd_qwk": float(sd[e]),
           "min_qwk": float(Q[:, e].min()), "mean_within1": float(W[:, e].mean()),
           "min_within1": float(W[:, e].min()),
           "constraint": float(viol[e]), "n_counting_checks": int(ok.sum())}
    for i, g in enumerate(groups):
        out[f"qwk_v{g}"] = float(Q[i, e])
        out[f"within1_v{g}"] = float(W[i, e])
    return out


# ------------------------------------------------------------------ search
def search_one(exp, leg, t, base=BASE, n_trials=N_TRIALS, epochs=EPOCHS) -> None:
    """The Bayesian search of one (rater, input, test group): trials one after another, because
    each proposal depends on the earlier results. Every run's curves go to curves_<search>.csv
    (one append per finished trial)."""
    base.mkdir(parents=True, exist_ok=True)
    pool = cm.load_pool()
    inner = [c for c in cm.CELLS if c[0] == t]
    splits = {c[1]: cm.split_paths(*c, pool) for c in inner}
    sampler = optuna.samplers.TPESampler(
        seed=SAMPLER_BASE[exp] + 100 * INPUTS.index(leg) + t, multivariate=True, n_startup_trials=N_STARTUP,
        constraints_func=lambda ft: [ft.user_attrs["constraint"]])
    study = optuna.create_study(study_name=name_of(exp, leg, t), storage=f"sqlite:///{db_path(exp, leg, t, base).as_posix()}",
                                direction="maximize", sampler=sampler, load_if_exists=True)
    for tr in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(tr.number, state=TrialState.FAIL)
        study.enqueue_trial(tr.params)
        print(f"  {exp} {leg} test group {t}: trial {tr.number} was interrupted; re-queued with the same parameters", flush=True)
    curves_csv = base / f"curves_{name_of(exp, leg, t)}.csv"
    failed_in_a_row = 0
    while True:
        done = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if len(done) >= n_trials:
            break
        trial = study.ask(SPACE)
        params = dict(trial.params)
        t0 = time.time()
        try:
            curves = {v: hf.train_cached(exp, leg, s["train"], s["val"], cm.seed_of((t, v), 0), params,
                                         epochs, checks_per_epoch=CHECKS)
                      for v, s in splits.items()}
        except Exception as exc:
            study.tell(trial, state=TrialState.FAIL)
            failed_in_a_row += 1
            print(f"  !! {exp} {leg} test group {t}: trial {trial.number} FAILED ({exc!r})", flush=True)
            if failed_in_a_row >= 3:
                raise SystemExit(f"{exp} {leg} test group {t}: 3 trials failed in a row; search stopped")
            continue
        failed_in_a_row = 0
        s = score_curves(curves)
        long = pd.concat([c.assign(trial=trial.number, lr=params["lr"], wd=params["wd"], val_group=v)
                          for v, c in curves.items()], ignore_index=True)
        long.to_csv(curves_csv, mode="a", header=not curves_csv.exists(), index=False)
        trial.set_user_attr("constraint", s["constraint"])      # read by the sampler
        trial.set_user_attr("s", s)
        study.tell(trial, s["score"])
        print(f"{datetime.now():%H:%M:%S} {exp} {leg} G{t} trial {trial.number:2d} ({len(done) + 1}/{n_trials}) "
              f"lr {params['lr']:.2e} wd {params['wd']:.2e}: score {s['score']:+.3f} "
              f"(mean {s['mean_qwk']:+.3f}, SD {s['sd_qwk']:.3f}, worst {s['min_qwk']:+.3f}) at check {s['check']}"
              + ("" if s["feasible"] else "  INFEASIBLE") + f"  ({time.time() - t0:.0f} s)", flush=True)
    best = best_trial(study)
    if best is not None:
        print(f"{exp} {leg} test group {t} done: lr {best.params['lr']:.3g}, wd {best.params['wd']:.3g}, "
              f"check {attrs(best)['check']} (epoch {attrs(best)['epoch']:g}), score {best.value:+.3f}", flush=True)


def best_trial(study):
    feas = [tr for tr in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            if attrs(tr).get("feasible")]
    return max(feas, key=lambda tr: tr.value) if feas else None


# ------------------------------------------------------------------ refit
def predict_rows(exp, leg, csv_path, head, thresholds, split, folder: Path, sigma) -> dict:
    """Score one split with the trained head (cached features): per-tile dump (preds_<split>.csv:
    raw output z and predicted grade) + the natural and grade-weighted (bal_*) metrics."""
    df, X = hf.features_of(exp, leg, csv_path)
    with torch.no_grad():
        z = head(X).squeeze(1)
        pred = thresholds.predict(z).numpy()
        u = thresholds().numpy()
    z = z.numpy()
    T = df["rating"].to_numpy(float).astype(int)
    pd.DataFrame({"filepath": df["filepath"], "rating": df["rating"], "raw": z, "pred": pred}).to_csv(
        folder / f"preds_{split}.csv", index=False)
    nat, bal = rating_metrics(T, pred, weighted=False), rating_metrics(T, pred, weighted=True)
    return {"dataset": split, "n": len(T), **nat, "smoothl1": float("nan"),
            "bonn_cost": bonn_cost(T, z, u, sigma, weighted=False),
            **{f"bal_{k}": v for k, v in bal.items()}, "bal_smoothl1": float("nan"),
            "bal_bonn_cost": bonn_cost(T, z, u, sigma, weighted=True)}


def refit(exp, leg, t, base=BASE, n_trials=N_TRIALS) -> None:
    study = load_study_if_any(exp, leg, t, base)
    n_done = n_complete(exp, leg, t, base)
    best = best_trial(study) if study is not None else None
    if n_done < n_trials or best is None:
        print(f"  {exp} {leg} test group {t}: search unfinished ({n_done}/{n_trials}) or no feasible trial; refit skipped")
        return
    folder = base / "refit" / name_of(exp, leg, t)
    a = attrs(best)
    spec = {"trial": best.number, "params": best.params, "checks": a["check"],
            "epochs": a["epoch"], "val_score_at_E": best.value,
            "val_mean_qwk_at_E": a["mean_qwk"], "val_sd_qwk_at_E": a["sd_qwk"],
            "seed": REFIT_SEED + t}
    if (folder / "metrics_all.csv").exists() and (folder / "refit.json").exists() \
            and json.loads((folder / "refit.json").read_text()) == json.loads(json.dumps(spec)):
        print(f"  {exp} {leg} test group {t}: refit already done")
        return
    folder.mkdir(parents=True, exist_ok=True)
    paths = cm.refit_paths(t, cm.load_pool())
    curves, head_sd, thr_sd = hf.train_cached(exp, leg, paths["train"], None, spec["seed"], best.params,
                                              checks_per_epoch=CHECKS, n_checks=spec["checks"], return_head=True)
    curves.to_csv(folder / "checks.csv", index=False)
    head = torch.nn.Linear(head_sd["weight"].shape[1], 1)
    head.load_state_dict(head_sd)
    thresholds = hf.rt.BONNThresholds(num_classes=5)
    thresholds.load_state_dict(thr_sd)
    sigma = hf.head_cfg(exp, leg).bonn_sigma
    rows = [predict_rows(exp, leg, paths[s], head, thresholds, s, folder, sigma) for s in ("train", "test")]
    pd.DataFrame(rows).to_csv(folder / "metrics_all.csv", index=False)
    # the trained head and thresholds (a few KB); with the frozen ImageNet backbone they ARE the model
    torch.save({"head_state_dict": head_sd, "bonn_thresholds_state_dict": thr_sd, "exp": exp, "leg": leg,
                "backbone": hf.head_cfg(exp, leg).backbone, "batchnorm": "eval (ImageNet statistics)",
                "spec": spec}, folder / "head.pt")
    (folder / "refit.json").write_text(json.dumps(spec, indent=1))
    m = pd.DataFrame(rows).set_index("dataset")
    print(f"  refit {exp} {leg} test group {t}: test QWK {m.loc['test', 'bal_qwk']:+.3f}, within-1 "
          f"{m.loc['test', 'bal_within1']:.3f} (validation at E: score {best.value:+.3f}, "
          f"mean {a['mean_qwk']:+.3f})", flush=True)


# ------------------------------------------------------------------ parallel searches
def run_parallel(units, jobs, base, n_trials, epochs, retries=2) -> list:
    """Each (rater, input, test group) search as its own process (one CPU thread each), at most
    `jobs` at a time; each refits its test group when its search ends."""
    logs = base / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    todo = [u for u in units if n_complete(*u, base=base) < n_trials
            or not (base / "refit" / name_of(*u) / "metrics_all.csv").exists()]
    tries = {u: 0 for u in todo}
    running = {}
    # the child processes import scripts.rating.nested_cv.*, so they need the repository root
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(cm.ROOT), os.environ.get("PYTHONPATH")]))}
    while todo or running:
        while todo and len(running) < jobs:
            u = todo.pop(0)
            exp, leg, t = u
            cmd = [sys.executable, str(Path(__file__).resolve()), "search", "--refit", "--exp", exp,
                   "--leg", leg, "--folds", str(t), "--base", str(base), "--trials", str(n_trials),
                   "--epochs", str(epochs), "--log", str(logs / f"{name_of(exp, leg, t)}.log")]
            running[u] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
            tries[u] += 1
            print(f"{datetime.now():%H:%M} started  {exp} {leg} test group {t} (pid {running[u].pid})", flush=True)
        time.sleep(5)
        for u, p in list(running.items()):
            code = p.poll()
            if code is None:
                continue
            del running[u]
            ok = n_complete(*u, base=base) >= n_trials
            print(f"{datetime.now():%H:%M} {'finished' if ok else f'stopped (exit {code})'} {u[0]} {u[1]} test group {u[2]}", flush=True)
            if not ok and tries[u] <= retries:
                todo.append(u)
    return [u for u in units if n_complete(*u, base=base) < n_trials]


# ------------------------------------------------------------------ status, summary
def status(base=BASE) -> None:
    units = [(e, l, t) for e in EXPS for l in INPUTS for t in cm.GROUPS]
    n_done = sum(min(n_complete(*u, base=base), N_TRIALS) for u in units)
    print(f"{datetime.now():%Y-%m-%d %H:%M}  trials done {n_done} of {len(units) * N_TRIALS}")
    for exp, leg, t in units:
        study = load_study_if_any(exp, leg, t, base)
        if study is None:
            print(f"{exp} {leg:9s} G{t}: not started")
            continue
        b = best_trial(study)
        rf = (base / "refit" / name_of(exp, leg, t) / "metrics_all.csv").exists()
        print(f"{exp} {leg:9s} G{t}: {n_complete(exp, leg, t, base):2d}/{N_TRIALS}"
              + (f"  best lr {b.params['lr']:.2e} wd {b.params['wd']:.2e} check {attrs(b)['check']:2d}"
                 f"  score {b.value:+.3f} (mean {attrs(b)['mean_qwk']:+.3f}, SD {attrs(b)['sd_qwk']:.3f})" if b else "")
              + ("  refit done" if rf else ""))


def summary(base=BASE) -> None:
    """summary/picks.csv (one row per rater, input and test group: the picked trial, its
    validation figures and the refit's train/test metrics, grade-weighted) and
    summary/test_table.csv (test QWK per group, mean, SD, worst; mean within-1 and AMAE)."""
    rows = []
    for exp, leg, t in [(e, l, t) for e in EXPS for l in INPUTS for t in cm.GROUPS]:
        folder = base / "refit" / name_of(exp, leg, t)
        if not (folder / "metrics_all.csv").exists():
            continue
        spec = json.loads((folder / "refit.json").read_text())
        study = load_study_if_any(exp, leg, t, base)
        b = study.trials[spec["trial"]]
        m = pd.read_csv(folder / "metrics_all.csv").set_index("dataset")
        row = {"exp": exp, "leg": leg, "test_group": t, "trial": spec["trial"],
               "p_lr": spec["params"]["lr"], "p_wd": spec["params"]["wd"], "check": spec["checks"],
               "epoch": spec["epochs"], **{f"val_{k}": v for k, v in attrs(b).items()
                                            if k not in ("feasible", "check", "epoch", "constraint")}}
        for split in ("test", "train"):
            for k in ("qwk", "spearman_r", "within1", "mae", "rmse"):
                row[f"{split}_{k}"] = m.loc[split, f"bal_{k}"]
        rows.append(row)
    if not rows:
        print("no refit yet")
        return
    df = pd.DataFrame(rows)
    out = base / "summary"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "picks.csv", index=False)
    lines = []
    for exp, leg in [(e, l) for e in EXPS for l in INPUTS]:
        g = df[(df.exp == exp) & (df.leg == leg)].sort_values("test_group")
        if g.empty:
            continue
        q = g.test_qwk.to_numpy()
        lines.append({"config": exp, "input": leg, **{f"G{int(r.test_group)}": round(r.test_qwk, 2) for r in g.itertuples()},
                      "mean QWK": round(q.mean(), 3), "SD": round(q.std(ddof=1), 3) if len(q) > 1 else np.nan,
                      "worst": round(q.min(), 2), "within-1": round(g.test_within1.mean(), 3),
                      "groups < 0.600": int((g.test_within1 < MIN_WITHIN1 - TOL).sum()),
                      "AMAE": round(g.test_mae.mean(), 3)})
    table = pd.DataFrame(lines)
    table.to_csv(out / "test_table.csv", index=False)
    print(f"picks -> {out / 'picks.csv'}\n\nRefits scored on test (grade-weighted):\n{table.to_string(index=False)}")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "search", "refit", "status", "summary", "smoke"])
    ap.add_argument("--exp", choices=EXPS)
    ap.add_argument("--leg", choices=INPUTS)
    ap.add_argument("--folds", default="1,2,3,4,5")
    ap.add_argument("--jobs", type=int, default=6, help="run: searches at the same time (one CPU thread each)")
    ap.add_argument("--refit", action="store_true", help="search: refit each test group when its search ends")
    ap.add_argument("--log", default="")
    ap.add_argument("--base", default=str(BASE), help=argparse.SUPPRESS)
    ap.add_argument("--trials", type=int, default=N_TRIALS, help=argparse.SUPPRESS)
    ap.add_argument("--epochs", type=int, default=EPOCHS, help=argparse.SUPPRESS)
    a = ap.parse_args()
    base = Path(a.base)
    exps = [a.exp] if a.exp else EXPS
    legs = [a.leg] if a.leg else INPUTS
    folds = [int(x) for x in a.folds.split(",")]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    torch.set_num_threads(1)                  # several searches share the CPU
    if a.command == "status":
        status(base)
        return
    if a.command == "summary":
        summary(base)
        return
    with contextlib.ExitStack() as stack:
        if a.log:
            stack.enter_context(cm.tee_output(Path(a.log)))
        if a.command == "smoke":
            sb = BASE / "_smoke"
            import shutil
            shutil.rmtree(sb, ignore_errors=True)
            units = [("E04", "30", 1), ("E11", "15-otf", 2)]
            t0 = time.time()
            left = run_parallel(units, 2, sb, n_trials=3, epochs=2)
            assert not left, f"smoke: unfinished searches {left}"
            for u in units:
                assert (sb / "refit" / name_of(*u) / "metrics_all.csv").exists(), f"smoke: no refit for {u}"
                print((sb / "logs" / f"{name_of(*u)}.log").read_text(encoding="utf-8"))
            shutil.rmtree(sb, ignore_errors=True)
            print(f"smoke run OK in {time.time() - t0:.0f} s (2 parallel searches of 3 two-epoch trials + refits)")
            return
        units = [(e, l, t) for e in exps for l in legs for t in folds]
        if a.command == "search":
            for exp, leg, t in units:
                with cm.lock(name_of(exp, leg, t), base):
                    search_one(exp, leg, t, base, a.trials, a.epochs)
                    if a.refit:
                        refit(exp, leg, t, base, a.trials)
        elif a.command == "refit":
            for exp, leg, t in units:
                refit(exp, leg, t, base, a.trials)
        elif a.command == "run":
            stack.enter_context(cm.lock("run", base))
            print(f"{datetime.now():%Y-%m-%d %H:%M} nested CV run: {len(units)} searches of {a.trials} trials, "
                  f"{a.jobs} at a time", flush=True)
            left = run_parallel(units, a.jobs, base, a.trials, a.epochs)
            print(f"{datetime.now():%Y-%m-%d %H:%M} nested CV run: "
                  + (f"incomplete, unfinished searches {left}" if left else "finished"), flush=True)
            if not left:
                summary(base)


if __name__ == "__main__":
    sys.exit(main())
