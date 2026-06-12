"""
evaluate.py — Deterministic evaluation of a saved PPO model.

Usage
-----
    python -m src.evaluate --run-name arch64_64_seed0_fixed
    python -m src.evaluate --run-name arch64_64_seed0_fixed --n-episodes 20

Outputs (printed to stdout + saved in the run directory)
---------------------------------------------------------
eval_results.json  — mean/std return, completion rate, mean lap time
best_trajectory.png — trajectory plot of the best episode
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.track import monaco_inspired_track
from src.car import DT
from src.env import CarRacingEnv
from src.render import plot_track


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(run_name: str, n_episodes: int) -> None:
    """Load model from *run_name*, run *n_episodes* deterministic episodes.

    Always evaluates on the fixed Monaco-inspired track so results are
    comparable across runs regardless of how the model was trained.
    """
    run_dir = os.path.join("experiments", "runs", run_name)
    model_path = os.path.join(run_dir, "final_model.zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run 'python -m src.train --run-name {run_name}' first."
        )

    model = PPO.load(model_path)

    track = monaco_inspired_track()
    env = CarRacingEnv(track=track)

    returns = []
    lengths = []
    lap_times_steps = []   # steps for completed laps only
    completed = []
    best_return = -np.inf
    best_trajectory = None   # (xs, ys) of the best episode

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_return = 0.0
        xs, ys = [], []
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            xs.append(info["x"])
            ys.append(info["y"])

        returns.append(ep_return)
        lengths.append(info["lap_time_steps"])
        lap_done = terminated and info["unwrapped_s"] >= track.total_length
        completed.append(lap_done)
        if lap_done:
            lap_times_steps.append(info["lap_time_steps"])

        if ep_return > best_return:
            best_return = ep_return
            best_trajectory = (xs[:], ys[:], lap_done)

        print(f"  ep {ep+1:3d}/{n_episodes}  return={ep_return:8.1f}  "
              f"steps={info['lap_time_steps']:5d}  "
              f"{'LAP' if lap_done else 'DNF'}")

    # --- Aggregate stats ---
    mean_return = float(np.mean(returns))
    std_return  = float(np.std(returns))
    completion_rate = float(np.mean(completed))
    mean_lap_steps = float(np.mean(lap_times_steps)) if lap_times_steps else float("nan")
    std_lap_steps  = float(np.std(lap_times_steps))  if lap_times_steps else float("nan")
    mean_lap_secs  = mean_lap_steps * DT if not np.isnan(mean_lap_steps) else float("nan")

    print()
    print("=" * 52)
    print(f"  Episodes          : {n_episodes}")
    print(f"  Return  mean±std  : {mean_return:.1f} ± {std_return:.1f}")
    print(f"  Completion rate   : {completion_rate*100:.1f}%")
    if lap_times_steps:
        print(f"  Lap time (steps)  : {mean_lap_steps:.1f} ± {std_lap_steps:.1f}")
        print(f"  Lap time (sim s)  : {mean_lap_secs:.2f} s")
    else:
        print("  Lap time          : no completed laps")
    print("=" * 52)

    # --- Save results JSON ---
    results = {
        "run_name": run_name,
        "n_episodes": n_episodes,
        "mean_return": round(mean_return, 3),
        "std_return": round(std_return, 3),
        "completion_rate": round(completion_rate, 4),
        "mean_lap_steps": round(mean_lap_steps, 2) if not np.isnan(mean_lap_steps) else None,
        "std_lap_steps": round(std_lap_steps, 2) if not np.isnan(std_lap_steps) else None,
        "mean_lap_sim_s": round(mean_lap_secs, 3) if not np.isnan(mean_lap_secs) else None,
    }
    results_path = os.path.join(run_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # --- Best trajectory plot ---
    if best_trajectory is not None:
        xs, ys, lap_done = best_trajectory
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_track(track, ax=ax,
                   title=f"Best episode — {run_name}\n"
                         f"return={best_return:.1f}  "
                         f"{'LAP COMPLETED' if lap_done else 'DNF'}")

        n = len(xs)
        colors = plt.cm.plasma(np.linspace(0.0, 1.0, n))
        ax.scatter(xs, ys, c=colors, s=3, zorder=4, linewidths=0)
        ax.plot(xs[0], ys[0], "g^", markersize=9, zorder=5, label="Start")
        ax.plot(xs[-1], ys[-1], "rs", markersize=9, zorder=5, label="End")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()

        fig_path = os.path.join(run_dir, "best_trajectory.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Trajectory saved: {fig_path}")


# ---------------------------------------------------------------------------
# Quick reward-curve plot from progress.csv
# ---------------------------------------------------------------------------

def plot_reward_curve(run_name: str) -> None:
    """Plot return vs episode from progress.csv and save as reward_curve.png."""
    csv_path = os.path.join("experiments", "runs", run_name, "progress.csv")
    if not os.path.exists(csv_path):
        print(f"No progress.csv found at {csv_path}")
        return

    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]   # single row edge case

    episodes   = data[:, 0]
    returns    = data[:, 1]
    wall_clock = data[:, 4]

    # Smooth with a running mean (window = 10 % of episodes, min 5)
    window = max(5, int(len(returns) * 0.05))
    kernel = np.ones(window) / window
    smoothed = np.convolve(returns, kernel, mode="valid")
    ep_smooth = episodes[window - 1:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training curve — {run_name}", fontsize=12)

    ax = axes[0]
    ax.plot(episodes, returns, alpha=0.3, color="steelblue", linewidth=0.8,
            label="raw")
    ax.plot(ep_smooth, smoothed, color="steelblue", linewidth=2,
            label=f"smoothed (w={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(wall_clock / 60, returns, alpha=0.3, color="darkorange",
             linewidth=0.8)
    ax2.plot(wall_clock[window - 1:] / 60, smoothed, color="darkorange",
             linewidth=2)
    ax2.set_xlabel("Wall-clock time (min)")
    ax2.set_ylabel("Return")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join("experiments", "runs", run_name, "reward_curve.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Reward curve saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO model")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Subdirectory name under experiments/runs/")
    parser.add_argument("--n-episodes", type=int, default=20,
                        help="Number of deterministic evaluation episodes")
    parser.add_argument("--plot-curve", action="store_true",
                        help="Also plot the training reward curve from progress.csv")
    args = parser.parse_args()

    evaluate(args.run_name, args.n_episodes)

    if args.plot_curve:
        plot_reward_curve(args.run_name)


if __name__ == "__main__":
    main()
