"""
train.py — Training entry point for CarRacingEnv using SB3 PPO.

Usage
-----
    python -m src.train --net-arch 64,64 --seed 0 --total-steps 1000000
    python -m src.train --net-arch 256,256 --seed 1 --track random --run-name my_run

CLI arguments
-------------
--net-arch      Comma-separated hidden layer sizes, e.g. "64,64" or "16" or "512,512,512".
--seed          Integer random seed (seeds Python, NumPy, PyTorch, and SB3).
--total-steps   Total environment steps (default: 1_000_000).
--track         "fixed" (monaco) or "random" (new track each episode).
--run-name      Subdirectory name under experiments/runs/. Defaults to
                "arch{net_arch}_seed{seed}_{track}".
--n-envs        Number of parallel envs via SubprocVecEnv (default: 8).

Outputs (all in experiments/runs/{run-name}/)
--------------------------------------------
progress.csv    — per-episode: return, length, mean_log_std, wall_clock_s
final_model.zip — SB3 model checkpoint
config.json     — all CLI args + git commit hash
"""

import argparse
import csv
import json
import os
import random
import subprocess
import time
from typing import List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.track import track as default_track
from src.env import CarRacingEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_net_arch(s: str) -> List[int]:
    """Parse "64,64" → [64, 64]."""
    return [int(x) for x in s.split(",") if x.strip()]


def get_git_hash() -> str:
    """Return the current HEAD commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(track_mode: str, seed: int, rank: int):
    """Return a callable that creates a single Monitor-wrapped env.

    Each subprocess gets a unique seed = seed + rank so they are
    statistically independent.
    """
    def _init():
        env = CarRacingEnv(track=default_track())
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ---------------------------------------------------------------------------
# Custom callback: log per-episode stats to CSV
# ---------------------------------------------------------------------------

class EpisodeLoggerCallback(BaseCallback):
    """Write one CSV row per completed episode.

    Columns
    -------
    episode         : episode index (across all envs)
    return          : undiscounted sum of rewards
    length          : number of env steps
    mean_log_std    : mean of the policy's log_std parameter (exploration)
    wall_clock_s    : cumulative wall-clock seconds since training started
    """

    def __init__(self, csv_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._csv_path = csv_path
        self._episode_count = 0
        self._start_time: Optional[float] = None
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        self._start_time = time.perf_counter()
        os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
        self._file = open(self._csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["episode", "return", "length", "mean_log_std", "wall_clock_s"]
        )
        self._file.flush()

    def _on_step(self) -> bool:
        # SB3 Monitor stores episode info in the info dict under "episode"
        # when an episode finishes.
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue
            ep_return = ep["r"]
            ep_length = ep["l"]

            # Mean log_std of the policy (measure of exploration)
            log_std = self.model.policy.log_std.detach().cpu().numpy()
            mean_log_std = float(log_std.mean())

            wall_clock = time.perf_counter() - self._start_time

            self._writer.writerow([
                self._episode_count,
                round(float(ep_return), 4),
                int(ep_length),
                round(mean_log_std, 6),
                round(wall_clock, 3),
            ])
            self._file.flush()
            print(
                f"  ep {self._episode_count:5d} | "
                f"return {ep_return:8.1f} | "
                f"steps {ep_length:5d} | "
                f"log_std {mean_log_std:6.3f} | "
                f"{wall_clock:7.1f}s"
            )
            self._episode_count += 1
        return True   # True = keep training

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    net_arch: List[int],
    seed: int,
    total_steps: int,
    track_mode: str,
    run_name: str,
    n_envs: int,
) -> None:
    """Train a PPO agent and save artefacts to experiments/runs/{run_name}/."""

    run_dir = os.path.join("experiments", "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    seed_everything(seed)

    # --- Vectorised environment ---
    vec_env = SubprocVecEnv(
        [make_env(track_mode, seed, rank) for rank in range(n_envs)],
        start_method="fork",
    )

    # --- PPO model ---
    # policy_kwargs: same net_arch for both actor (pi) and critic (vf).
    # All other hyperparameters are SB3 defaults.
    policy_kwargs = dict(net_arch=net_arch)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        seed=seed,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # --- Callback ---
    csv_path = os.path.join(run_dir, "progress.csv")
    callback = EpisodeLoggerCallback(csv_path=csv_path)

    # --- Train ---
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)
    elapsed = time.perf_counter() - t0

    steps_per_sec = total_steps / elapsed
    print(f"\nTraining finished in {elapsed:.1f}s  ({steps_per_sec:.0f} steps/sec)")

    # --- Save model ---
    model_path = os.path.join(run_dir, "final_model")
    model.save(model_path)
    print(f"Model saved: {model_path}.zip")

    # --- Save config ---
    config = {
        "net_arch": net_arch,
        "seed": seed,
        "total_steps": total_steps,
        "track_mode": track_mode,
        "run_name": run_name,
        "n_envs": n_envs,
        "git_hash": get_git_hash(),
        "elapsed_s": round(elapsed, 2),
        "steps_per_sec": round(steps_per_sec, 1),
    }
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")

    vec_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on CarRacingEnv")
    parser.add_argument("--net-arch", type=str, default="64,64",
                        help='Hidden layer sizes, e.g. "64,64" or "256,256,256"')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--track", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Subdirectory name under experiments/runs/")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel SubprocVecEnv workers")
    args = parser.parse_args()

    net_arch = parse_net_arch(args.net_arch)

    run_name = args.run_name or (
        f"arch{'_'.join(str(x) for x in net_arch)}_seed{args.seed}_{args.track}"
    )

    print(f"Run name  : {run_name}")
    print(f"Net arch  : {net_arch}")
    print(f"Seed      : {args.seed}")
    print(f"Steps     : {args.total_steps:,}")
    print(f"Track     : {args.track}")
    print(f"N envs    : {args.n_envs}")
    print()

    train(
        net_arch=net_arch,
        seed=args.seed,
        total_steps=args.total_steps,
        track_mode=args.track,
        run_name=run_name,
        n_envs=args.n_envs,
    )


if __name__ == "__main__":
    main()
