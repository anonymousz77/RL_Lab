#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
plot_better_ppo.py  —  Plotting script. Additive; does not modify the base codebase.

Rich learning-curve suite for the improved PPO-only run (ppo_better_train.py).
Chart conventions match the base codebase's simulator/visualize_training.py so curves
overlay cleanly side-by-side: x-axis "Iteration", success "Success Rate" y in
[0,1], train=steelblue / eval=darkorange, grid alpha=0.3. 200 dpi.

Reads results/better_ppo/metrics.csv and writes into results/:
    better_ppo_success.png           success (train+eval) vs Iteration (+top steps axis)
    better_ppo_success_vs_steps.png  success (train+eval) vs total env steps
    better_ppo_return.png            average return vs Iteration
    better_ppo_eplen.png             episode length vs Iteration
    better_ppo_entropy.png           policy entropy vs Iteration
    better_ppo_combined.png          2x2 multi-panel

Run:  set PYTHONUTF8=1  &&  .venv\\Scripts\\python.exe plot_better_ppo.py
"""
import sys, os, csv, math, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
METRICS = os.path.join(RESULTS, "better_ppo", "metrics.csv")

# base codebase conventions (simulator/visualize_training.py)
C_TRAIN = "steelblue"
C_EVAL  = "darkorange"
C_RET   = "#2ca02c"
C_LEN   = "#9467bd"
C_ENT   = "#e67e22"
TITLE_PREFIX = "Bare PPO-only (8 envs + GAE) — SimpleDoorKey, seed 42"
DPI = 200


def load():
    if not os.path.exists(METRICS):
        raise FileNotFoundError(f"{METRICS} not found — run ppo_better_train.py first.")
    cols = {}
    with open(METRICS, newline="") as f:
        rd = csv.DictReader(f)
        for k in rd.fieldnames:
            cols[k] = []
        for row in rd:
            for k in rd.fieldnames:
                v = row[k]
                cols[k].append(float(v) if v not in ("", "nan", "None") else np.nan)
    return {k: np.array(v) for k, v in cols.items()}


def _rolling(y, w=10):
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return y
    w = min(w, len(y))
    return np.convolve(y, np.ones(w) / w, mode="same")


def _add_steps_axis(ax, d):
    """Secondary top x-axis showing total env steps aligned to iteration."""
    it = d["iter"]
    steps = d["total_steps"]
    if len(it) < 2:
        return
    # constant steps/iter -> linear map iteration <-> steps
    per = (steps[-1] - steps[0]) / (it[-1] - it[0]) if it[-1] != it[0] else steps[-1]
    base = steps[0] - per * it[0]
    secax = ax.secondary_xaxis("top", functions=(lambda x: base + per * x,
                                                 lambda s: (s - base) / per))
    secax.set_xlabel("Total environment steps")


def _plot_success(ax, d, add_steps_axis=False):
    it = d["iter"]
    ax.plot(it, d["success_rate"], color=C_TRAIN, alpha=0.25, lw=1.0)
    ax.plot(it, _rolling(d["success_rate"], 10), color=C_TRAIN, lw=2.2, label="Train")
    ev = d["eval_success_rate"]
    mask = ~np.isnan(ev)
    if mask.any():
        ax.plot(it[mask], ev[mask], color=C_EVAL, lw=2.0, marker="o", ms=3.0, label="Eval")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    if add_steps_axis:
        _add_steps_axis(ax, d)


def _plot_success_vs_steps(ax, d):
    steps = d["total_steps"]
    ax.plot(steps, d["success_rate"], color=C_TRAIN, alpha=0.25, lw=1.0)
    ax.plot(steps, _rolling(d["success_rate"], 10), color=C_TRAIN, lw=2.2, label="Train")
    ev = d["eval_success_rate"]
    mask = ~np.isnan(ev)
    if mask.any():
        ax.plot(steps[mask], ev[mask], color=C_EVAL, lw=2.0, marker="o", ms=3.0, label="Eval")
    ax.set_xlabel("Total environment steps")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)


def _plot_return(ax, d):
    it = d["iter"]
    ax.plot(it, d["average_return"], color=C_RET, alpha=0.25, lw=1.0)
    ax.plot(it, _rolling(d["average_return"], 10), color=C_RET, lw=2.2, label="10-iter mean")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)


def _plot_eplen(ax, d):
    it = d["iter"]
    ax.plot(it, d["episode_length"], color=C_LEN, alpha=0.25, lw=1.0)
    ax.plot(it, _rolling(d["episode_length"], 10), color=C_LEN, lw=2.2, label="10-iter mean")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Episode Length")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)


def _plot_entropy(ax, d):
    it = d["iter"]
    ax.plot(it, d["entropy"], color=C_ENT, lw=2.0, label="Policy entropy")
    ax.axhline(math.log(7), color="grey", ls=":", lw=1.2, label="Max entropy (ln 7)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Policy Entropy (nats)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)


def _save_single(plot_fn, title, name, d, **kw):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_fn(ax, d, **kw)
    ax.set_title(f"{TITLE_PREFIX}\n{title}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RESULTS, name)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main():
    global METRICS, RESULTS, TITLE_PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default="better_ppo",
                    help="results/<subdir>/ for input metrics.csv AND PNG output")
    ap.add_argument("--title", default=TITLE_PREFIX, help="plot title prefix")
    args = ap.parse_args()
    # additive overrides; SimpleDoorKey defaults preserved
    RESULTS      = os.path.join(HERE, "results", args.subdir)
    METRICS      = os.path.join(RESULTS, "metrics.csv")
    TITLE_PREFIX = args.title

    d = load()
    n = len(d["iter"])
    print(f"[plot] loaded {n} iterations (final step {int(d['total_steps'][-1]):,})")

    # success vs Iteration (with top steps axis); leave headroom for the top axis
    fig, ax = plt.subplots(figsize=(9, 5.7))
    _plot_success(ax, d, add_steps_axis=True)
    ax.set_title(f"{TITLE_PREFIX}\nSuccess Rate (train + eval)", fontsize=12, fontweight="bold", pad=28)
    fig.tight_layout()
    out = os.path.join(RESULTS, "better_ppo_success.png")
    fig.savefig(out, dpi=DPI); plt.close(fig); print(f"[plot] wrote {out}")

    _save_single(_plot_success_vs_steps, "Success Rate vs environment steps (sample efficiency)",
                 "better_ppo_success_vs_steps.png", d)
    _save_single(_plot_return,  "Reward Curve (average return)", "better_ppo_return.png", d)
    _save_single(_plot_eplen,   "Episode Length",               "better_ppo_eplen.png", d)
    _save_single(_plot_entropy, "Policy Entropy",               "better_ppo_entropy.png", d)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    _plot_success(axes[0, 0], d); axes[0, 0].set_title("Success Rate (train + eval)", fontweight="bold")
    _plot_return(axes[0, 1], d);  axes[0, 1].set_title("Reward Curve", fontweight="bold")
    _plot_eplen(axes[1, 0], d);   axes[1, 0].set_title("Episode Length", fontweight="bold")
    _plot_entropy(axes[1, 1], d); axes[1, 1].set_title("Policy Entropy", fontweight="bold")
    fig.suptitle(TITLE_PREFIX, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(RESULTS, "better_ppo_combined.png")
    fig.savefig(out, dpi=DPI); plt.close(fig); print(f"[plot] wrote {out}")
    print("[plot] DONE")


if __name__ == "__main__":
    main()
