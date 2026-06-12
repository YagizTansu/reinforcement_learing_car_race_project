"""
plots.py — Aggregate results across seeds and produce report figures.

Requires all 16 runs to be complete (or at least the seeds you want to plot).
Partial results are handled gracefully: missing seeds are skipped with a warning.

Outputs (all saved to experiments/figures/)
-------------------------------------------
1. learning_curves.png      — rolling-mean return vs env steps, mean ± std per arch
2. final_performance.png    — bar chart: eval return + completion rate per arch
3. convergence_episodes.png — episodes to reach 90% of final return, mean ± std
4. convergence_wallclock.png — same but x = wall-clock seconds
5. sigma_decay.png          — policy log_std vs env steps per arch
6. results_table.md         — summary table, one row per arch

Convergence definition (stated in axis labels and docstrings)
-------------------------------------------------------------
  "Convergence" = first episode where the 50-episode rolling mean
  return >= 90% of the run's own final 50-episode rolling mean.
  If never reached, the run is excluded from that arch's average.

Usage
-----
  python -m src.plots                  # use eval from evaluate.py (20 eps)
  python -m src.plots --n-eval 5       # quick re-eval with fewer episodes
  python -m src.plots --no-eval        # skip re-eval, use cached eval_results.json
"""

import argparse
import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Experiment grid (must match run_experiments.py)
NET_ARCHS = [
    [16],
    [64, 64],
    [256, 256],
    [512, 512, 512],
]
SEEDS = [0, 1, 2, 3]
TRACK_MODE = "fixed"
FIGURES_DIR = os.path.join("experiments", "figures")
RUNS_DIR    = os.path.join("experiments", "runs")

ROLLING_WINDOW = 50       # episodes for rolling mean
CONVERGENCE_THRESHOLD = 0.90   # 90 % of final rolling mean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def arch_to_str(arch):
    return "_".join(str(x) for x in arch)


def arch_label(arch):
    """Human-readable label for plot legends."""
    return "[" + ",".join(str(x) for x in arch) + "]"


def run_name_for(arch, seed):
    return f"arch{arch_to_str(arch)}_seed{seed}_{TRACK_MODE}"


def run_dir(arch, seed):
    return os.path.join(RUNS_DIR, run_name_for(arch, seed))


def load_progress(arch, seed):
    """Load progress.csv → dict of arrays, or None if missing."""
    path = os.path.join(run_dir(arch, seed), "progress.csv")
    if not os.path.isfile(path):
        return None
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if len(data) == 0:
        return None
    return {
        "episode":       data[:, 0],
        "return":        data[:, 1],
        "length":        data[:, 2],
        "mean_log_std":  data[:, 3],
        "wall_clock_s":  data[:, 4],
    }


def load_config(arch, seed):
    """Load config.json, or return empty dict."""
    path = os.path.join(run_dir(arch, seed), "config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def rolling_mean(arr, window):
    """Centred rolling mean; edges are computed with a shrinking window."""
    kernel = np.ones(window) / window
    # Use 'valid' convolution and trim; prepend nans for alignment
    if len(arr) < window:
        return np.full_like(arr, float("nan"))
    valid = np.convolve(arr, kernel, mode="valid")
    pad = len(arr) - len(valid)
    return np.concatenate([np.full(pad, float("nan")), valid])


def steps_at_episode(progress):
    """Reconstruct cumulative env steps at each episode end.

    Each episode consumed `length` steps across all envs; we integrate.
    The progress.csv does not store cumulative steps directly, so we
    approximate from the sum of episode lengths.  Because 8 envs run in
    parallel, each CSV row corresponds to ONE env's episode; total steps
    ≈ cumulative sum of lengths (close enough for plotting; actual step
    count differs by up to n_envs−1 episodes).
    """
    return np.cumsum(progress["length"])


def convergence_episode(returns, window=ROLLING_WINDOW, threshold=CONVERGENCE_THRESHOLD):
    """First episode where rolling mean >= threshold * final rolling mean.

    Returns
    -------
    int or None
        Episode index (0-based), or None if never reached.
    """
    rm = rolling_mean(returns, window)
    final_rm = np.nanmean(rm[-window:])
    target = threshold * final_rm
    indices = np.where(rm >= target)[0]
    if len(indices) == 0 or np.isnan(target):
        return None
    return int(indices[0])


def load_or_run_eval(arch, seed, n_eval):
    """Return eval_results dict, running evaluate() if needed."""
    path = os.path.join(run_dir(arch, seed), "eval_results.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    # Try to run evaluate
    model_path = os.path.join(run_dir(arch, seed), "final_model.zip")
    if not os.path.isfile(model_path):
        return None
    from src.evaluate import evaluate
    evaluate(run_name_for(arch, seed), n_episodes=n_eval)
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# 1. Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(arch_data: dict) -> None:
    """Rolling-mean return vs cumulative env steps, mean ± std per arch."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(NET_ARCHS)))

    for arch, color in zip(NET_ARCHS, colors):
        label = arch_label(arch)
        seed_curves = []   # list of (steps_arr, rm_arr) per seed

        for seed in SEEDS:
            p = arch_data.get((tuple(arch), seed))
            if p is None:
                continue
            steps = steps_at_episode(p)
            rm    = rolling_mean(p["return"], ROLLING_WINDOW)
            seed_curves.append((steps, rm))

        if not seed_curves:
            warnings.warn(f"No data for arch {label}")
            continue

        # Interpolate all seeds onto a common step grid
        all_steps = np.concatenate([sc[0] for sc in seed_curves])
        step_grid = np.linspace(0, np.percentile(all_steps, 95), 300)

        interp_curves = []
        for steps, rm in seed_curves:
            # Remove NaNs before interpolation
            valid = ~np.isnan(rm)
            if valid.sum() < 2:
                continue
            interp = np.interp(step_grid, steps[valid], rm[valid])
            interp_curves.append(interp)

        if not interp_curves:
            continue

        mat = np.stack(interp_curves, axis=0)   # (n_seeds, n_points)
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)

        ax.plot(step_grid / 1e6, mean, label=label, color=color, linewidth=2)
        ax.fill_between(step_grid / 1e6, mean - std, mean + std,
                        alpha=0.2, color=color)

    ax.set_xlabel("Environment steps (millions)", fontsize=12)
    ax.set_ylabel(f"Episode return (rolling mean, w={ROLLING_WINDOW})", fontsize=12)
    ax.set_title("Learning curves — mean ± std across seeds", fontsize=13)
    ax.legend(title="Net arch", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "learning_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 2. Final performance bar chart
# ---------------------------------------------------------------------------

def plot_final_performance(eval_data: dict) -> None:
    """Bar chart: eval return and completion rate per arch, error bars across seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Final evaluation performance (mean ± std across seeds)", fontsize=13)

    x = np.arange(len(NET_ARCHS))
    width = 0.5
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(NET_ARCHS)))
    labels = [arch_label(a) for a in NET_ARCHS]

    # --- Return ---
    ax = axes[0]
    means, stds = [], []
    for arch in NET_ARCHS:
        vals = [eval_data[(tuple(arch), s)]["mean_return"]
                for s in SEEDS
                if eval_data.get((tuple(arch), s)) is not None]
        means.append(np.mean(vals) if vals else 0)
        stds.append(np.std(vals)   if len(vals) > 1 else 0)

    bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean eval return", fontsize=11)
    ax.set_title("Return")
    ax.grid(True, axis="y", alpha=0.3)

    # --- Completion rate ---
    ax = axes[1]
    means_cr, stds_cr = [], []
    for arch in NET_ARCHS:
        vals = [eval_data[(tuple(arch), s)]["completion_rate"] * 100
                for s in SEEDS
                if eval_data.get((tuple(arch), s)) is not None]
        means_cr.append(np.mean(vals) if vals else 0)
        stds_cr.append(np.std(vals)   if len(vals) > 1 else 0)

    ax.bar(x, means_cr, width, yerr=stds_cr, capsize=5,
           color=colors, alpha=0.8, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Completion rate (%)", fontsize=11)
    ax.set_title("Completion rate")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "final_performance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 3 & 4. Convergence plots
# ---------------------------------------------------------------------------

def _convergence_values(arch_data):
    """Return (conv_episodes, conv_wallclock) dicts keyed by (arch_tuple, seed)."""
    conv_ep = {}
    conv_wc = {}
    for arch in NET_ARCHS:
        for seed in SEEDS:
            p = arch_data.get((tuple(arch), seed))
            if p is None:
                continue
            ep_idx = convergence_episode(p["return"])
            if ep_idx is None:
                continue
            conv_ep[(tuple(arch), seed)] = ep_idx
            # Wall-clock at that episode
            if ep_idx < len(p["wall_clock_s"]):
                conv_wc[(tuple(arch), seed)] = float(p["wall_clock_s"][ep_idx])
    return conv_ep, conv_wc


def plot_convergence(arch_data: dict) -> None:
    """Two bar charts: convergence by episodes and by wall-clock seconds."""
    conv_ep, conv_wc = _convergence_values(arch_data)

    threshold_pct = int(CONVERGENCE_THRESHOLD * 100)
    xlabel_ep = (
        f"Episodes until rolling-mean return ≥ {threshold_pct}% of\n"
        f"final rolling-mean return (w={ROLLING_WINDOW})"
    )
    xlabel_wc = (
        f"Wall-clock seconds until rolling-mean return ≥ {threshold_pct}% of\n"
        f"final rolling-mean return (w={ROLLING_WINDOW})"
    )

    for metric_dict, ylabel, filename, title_suffix in [
        (conv_ep, xlabel_ep, "convergence_episodes.png", "episodes"),
        (conv_wc, xlabel_wc, "convergence_wallclock.png", "wall-clock seconds"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(NET_ARCHS))
        width = 0.5
        colors = plt.cm.tab10(np.linspace(0, 0.8, len(NET_ARCHS)))

        means, stds = [], []
        for arch in NET_ARCHS:
            vals = [metric_dict[(tuple(arch), s)]
                    for s in SEEDS
                    if (tuple(arch), s) in metric_dict]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals)   if len(vals) > 1 else 0)

        ax.bar(x, means, width, yerr=stds, capsize=5,
               color=colors, alpha=0.8, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([arch_label(a) for a in NET_ARCHS], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Convergence speed (mean ± std across seeds) — {title_suffix}",
                     fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        out = os.path.join(FIGURES_DIR, filename)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 5. Policy std decay
# ---------------------------------------------------------------------------

def plot_sigma_decay(arch_data: dict) -> None:
    """Mean policy log_std vs env steps, mean ± std across seeds per arch."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(NET_ARCHS)))

    for arch, color in zip(NET_ARCHS, colors):
        label = arch_label(arch)
        seed_curves = []

        for seed in SEEDS:
            p = arch_data.get((tuple(arch), seed))
            if p is None:
                continue
            steps = steps_at_episode(p)
            seed_curves.append((steps, p["mean_log_std"]))

        if not seed_curves:
            continue

        all_steps = np.concatenate([sc[0] for sc in seed_curves])
        step_grid = np.linspace(0, np.percentile(all_steps, 95), 300)

        interp_curves = []
        for steps, log_std in seed_curves:
            interp = np.interp(step_grid, steps, log_std)
            interp_curves.append(interp)

        mat  = np.stack(interp_curves, axis=0)
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)

        ax.plot(step_grid / 1e6, mean, label=label, color=color, linewidth=2)
        ax.fill_between(step_grid / 1e6, mean - std, mean + std,
                        alpha=0.2, color=color)

    ax.set_xlabel("Environment steps (millions)", fontsize=12)
    ax.set_ylabel("Mean policy log_std", fontsize=12)
    ax.set_title("Policy standard deviation decay — mean ± std across seeds",
                 fontsize=13)
    ax.legend(title="Net arch", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "sigma_decay.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 6. Results table
# ---------------------------------------------------------------------------

def write_results_table(arch_data: dict, eval_data: dict) -> None:
    """Write results_table.md — one row per net architecture."""
    from src.car import DT as _DT

    conv_ep, conv_wc = _convergence_values(arch_data)

    header = (
        "| Net arch | Final return | Completion | Lap time (steps) "
        "| Lap time (sim s) | Episodes to converge | Wall-clock to converge (s) "
        "| Wall-clock / 1k steps (s) |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []

    for arch in NET_ARCHS:
        label = arch_label(arch)

        # --- eval stats ---
        eval_vals = [eval_data[(tuple(arch), s)]
                     for s in SEEDS
                     if eval_data.get((tuple(arch), s)) is not None]
        if eval_vals:
            ret_mean = np.mean([v["mean_return"]        for v in eval_vals])
            ret_std  = np.std( [v["mean_return"]        for v in eval_vals])
            cr_mean  = np.mean([v["completion_rate"]    for v in eval_vals]) * 100
            lap_vals = [v["mean_lap_steps"] for v in eval_vals if v.get("mean_lap_steps")]
            lap_step_mean = np.mean(lap_vals) if lap_vals else float("nan")
            lap_sim_mean  = lap_step_mean * _DT if not np.isnan(lap_step_mean) else float("nan")
        else:
            ret_mean = ret_std = cr_mean = lap_step_mean = lap_sim_mean = float("nan")

        # --- convergence ---
        ep_vals = [conv_ep[(tuple(arch), s)] for s in SEEDS
                   if (tuple(arch), s) in conv_ep]
        wc_vals = [conv_wc[(tuple(arch), s)] for s in SEEDS
                   if (tuple(arch), s) in conv_wc]
        ep_str = (f"{np.mean(ep_vals):.0f} ± {np.std(ep_vals):.0f}"
                  if ep_vals else "N/A")
        wc_str = (f"{np.mean(wc_vals):.0f} ± {np.std(wc_vals):.0f}"
                  if wc_vals else "N/A")

        # --- throughput: seconds per 1k steps ---
        wc_per_1k_vals = []
        for seed in SEEDS:
            p = arch_data.get((tuple(arch), seed))
            cfg = load_config(arch, seed)
            if p is not None and cfg.get("total_steps") and cfg.get("elapsed_s"):
                wc_per_1k_vals.append(cfg["elapsed_s"] / cfg["total_steps"] * 1000)
        wc_1k_str = (f"{np.mean(wc_per_1k_vals):.3f} ± {np.std(wc_per_1k_vals):.3f}"
                     if wc_per_1k_vals else "N/A")

        row = (
            f"| {label} "
            f"| {ret_mean:.1f} ± {ret_std:.1f} "
            f"| {cr_mean:.1f}% "
            f"| {lap_step_mean:.0f} " if not np.isnan(lap_step_mean) else f"| N/A "
        )
        # Build row manually to handle NaNs cleanly
        lap_step_str = f"{lap_step_mean:.0f}" if not np.isnan(lap_step_mean) else "N/A"
        lap_sim_str  = f"{lap_sim_mean:.2f}"  if not np.isnan(lap_sim_mean)  else "N/A"
        row = (
            f"| {label} "
            f"| {ret_mean:.1f} ± {ret_std:.1f} "
            f"| {cr_mean:.1f}% "
            f"| {lap_step_str} "
            f"| {lap_sim_str} s "
            f"| {ep_str} "
            f"| {wc_str} "
            f"| {wc_1k_str} |"
        )
        rows.append(row)

    threshold_pct = int(CONVERGENCE_THRESHOLD * 100)
    content = (
        f"# Experiment results\n\n"
        f"**Convergence** = first episode where "
        f"{ROLLING_WINDOW}-episode rolling-mean return ≥ {threshold_pct}% "
        f"of the run's own final {ROLLING_WINDOW}-episode rolling mean.\n\n"
        + header
        + "\n".join(rows)
        + "\n"
    )

    out = os.path.join(FIGURES_DIR, "results_table.md")
    with open(out, "w") as f:
        f.write(content)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate experiment report figures")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="Episodes for evaluation (if re-running eval)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation; only use cached eval_results.json")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Load all progress data ---
    print("Loading progress CSV files...")
    arch_data = {}
    for arch in NET_ARCHS:
        for seed in SEEDS:
            p = load_progress(arch, seed)
            if p is None:
                warnings.warn(
                    f"Missing progress.csv for arch={arch_label(arch)} seed={seed}"
                )
            arch_data[(tuple(arch), seed)] = p

    n_loaded = sum(1 for v in arch_data.values() if v is not None)
    print(f"Loaded {n_loaded} / {len(arch_data)} progress files.")

    if n_loaded == 0:
        print("No data found. Run the experiment campaign first:")
        print("  python run_experiments.py")
        return

    # --- Load / run evaluations ---
    print("Loading evaluation results...")
    eval_data = {}
    for arch in NET_ARCHS:
        for seed in SEEDS:
            if args.no_eval:
                path = os.path.join(run_dir(arch, seed), "eval_results.json")
                if os.path.isfile(path):
                    with open(path) as f:
                        eval_data[(tuple(arch), seed)] = json.load(f)
                else:
                    eval_data[(tuple(arch), seed)] = None
            else:
                eval_data[(tuple(arch), seed)] = load_or_run_eval(arch, seed, args.n_eval)

    n_eval = sum(1 for v in eval_data.values() if v is not None)
    print(f"Evaluation data: {n_eval} / {len(eval_data)} runs.")

    # --- Generate all plots ---
    print("\nGenerating figures...")
    plot_learning_curves(arch_data)
    plot_sigma_decay(arch_data)

    if n_eval > 0:
        plot_final_performance(eval_data)
    else:
        print("Skipping final_performance.png (no eval data).")

    plot_convergence(arch_data)
    write_results_table(arch_data, eval_data)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
