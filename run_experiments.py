"""
run_experiments.py — Sequential experiment campaign launcher.

Grid
----
  net_archs = [[16], [64,64], [256,256]]
  seeds     = [0, 1, 2, 3]
  track     = random (procedural circuits, seeds 0–49 per episode)
  steps     = 1_000_000 per run
  n_envs    = 8 (parallel workers inside each run)

Naming convention
-----------------
  arch16_seed0_random
  arch64_64_seed0_random
  arch256_256_seed1_random

Resumability
------------
  A run is skipped if experiments/runs/{run-name}/final_model.zip already
  exists.  This lets you interrupt and resume safely.

Usage
-----
  python run_experiments.py                    # full campaign
  python run_experiments.py --dry-run          # print what would run, skip all
  python run_experiments.py --total-steps 300000  # quick smoke of the grid
"""

import argparse
import json
import os
import sys
import time

# Arch list and seed list from CLAUDE.md (fixed — do not change)
NET_ARCHS = [
    [16],
    [64, 64],
    [256, 256],
]
SEEDS = [0, 1, 2, 3]
DEFAULT_TRACK_MODE = "random"
DEFAULT_TOTAL_STEPS = 1_000_000
N_ENVS = 8
RUNS_DIR = os.path.join("experiments", "runs")


def arch_to_str(arch):
    """[64, 64] → '64_64'"""
    return "_".join(str(x) for x in arch)


def run_name_for(arch, seed, track=DEFAULT_TRACK_MODE):
    return f"arch{arch_to_str(arch)}_seed{seed}_{track}"


def model_exists(run_name):
    path = os.path.join(RUNS_DIR, run_name, "final_model.zip")
    return os.path.isfile(path)


def fmt_duration(seconds: float) -> str:
    """Format seconds as e.g. '14.5 min' or '3.2 h'."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def estimate_run_seconds(total_steps: int) -> float:
    """Guess one run duration from past config.json files, else 15 min."""
    times = []
    if os.path.isdir(RUNS_DIR):
        for name in os.listdir(RUNS_DIR):
            cfg_path = os.path.join(RUNS_DIR, name, "config.json")
            if not os.path.isfile(cfg_path):
                continue
            with open(cfg_path) as f:
                cfg = json.load(f)
            if cfg.get("total_steps") == total_steps and cfg.get("elapsed_s"):
                times.append(float(cfg["elapsed_s"]))
    if times:
        return sum(times) / len(times)
    return 15 * 60


def print_campaign_status(done: int, total: int, elapsed: float, eta: float) -> None:
    """Print overall campaign progress line."""
    pct = 100.0 * done / total if total else 100.0
    print()
    print("-" * 60)
    print(
        f"CAMPAIGN  {pct:5.1f}%  ({done}/{total} runs)  "
        f"elapsed {fmt_duration(elapsed)}  "
        f"ETA {fmt_duration(eta)}"
    )
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run the full experiment grid")
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS,
                        help="Steps per run (default 1_000_000)")
    parser.add_argument("--n-envs", type=int, default=N_ENVS,
                        help="Parallel envs per run (default 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan but do not train")
    parser.add_argument("--track", choices=["fixed", "random"],
                        default=DEFAULT_TRACK_MODE,
                        help="fixed = same circuit every episode; "
                             "random = procedural circuit each episode (default)")
    args = parser.parse_args()

    track_mode = args.track

    # Build the full grid: 3 archs × 4 seeds = 12 runs
    runs = [
        (arch, seed)
        for arch in NET_ARCHS
        for seed in SEEDS
    ]
    total_runs = len(runs)

    print("=" * 60)
    print(f"Experiment grid: {total_runs} runs")
    print(f"  net archs : {[arch_to_str(a) for a in NET_ARCHS]}")
    print(f"  seeds     : {SEEDS}")
    print(f"  steps/run : {args.total_steps:,}")
    print(f"  n_envs    : {args.n_envs}")
    print(f"  track     : {track_mode}")
    print("=" * 60)

    pending = []
    skipped = []
    for arch, seed in runs:
        name = run_name_for(arch, seed, track=track_mode)
        if model_exists(name):
            skipped.append(name)
        else:
            pending.append((arch, seed, name))

    done_at_start = len(skipped)
    if skipped:
        print(f"\nSkipping {len(skipped)} already-completed run(s):")
        for n in skipped:
            print(f"  [done] {n}")

    print(f"\nRunning {len(pending)} run(s):")
    for _, _, name in pending:
        print(f"  [todo] {name}")

    est_per_run = estimate_run_seconds(args.total_steps)
    est_total = est_per_run * len(pending)
    print(
        f"\nEstimated time for remaining runs: ~{fmt_duration(est_total)} "
        f"(~{fmt_duration(est_per_run)} per run, from past runs if any)"
    )
    if done_at_start:
        print(
            f"Campaign already {100.0 * done_at_start / total_runs:.1f}% complete "
            f"({done_at_start}/{total_runs} runs)"
        )

    if args.dry_run:
        print("\n--dry-run set, exiting without training.")
        return

    if not pending:
        print("\nAll runs already complete.")
        return

    # Import here so the module is only loaded when actually training
    from src.train import train as run_training

    campaign_start = time.perf_counter()
    run_durations = []

    for run_idx, (arch, seed, name) in enumerate(pending):
        global_done = done_at_start + run_idx
        global_pct = 100.0 * global_done / total_runs
        if run_durations:
            avg_run = sum(run_durations) / len(run_durations)
            eta = avg_run * (len(pending) - run_idx)
        else:
            eta = est_per_run * (len(pending) - run_idx)
        elapsed_total = time.perf_counter() - campaign_start

        print()
        print("=" * 60)
        print(f"RUN {run_idx + 1}/{len(pending)}  |  CAMPAIGN {global_pct:.1f}%  ({global_done}/{total_runs})")
        print(f"  name={name}  arch={arch}  seed={seed}")
        print(f"  campaign elapsed={fmt_duration(elapsed_total)}  ETA remaining={fmt_duration(eta)}")
        print("=" * 60)

        run_start = time.perf_counter()
        try:
            run_training(
                net_arch=arch,
                seed=seed,
                total_steps=args.total_steps,
                track_mode=track_mode,
                run_name=name,
                n_envs=args.n_envs,
            )
        except Exception as exc:
            print(f"\n[ERROR] Run {name} failed: {exc}", file=sys.stderr)
            print("Continuing with next run...", file=sys.stderr)
            continue

        run_elapsed = time.perf_counter() - run_start
        run_durations.append(run_elapsed)
        global_done = done_at_start + run_idx + 1
        elapsed_total = time.perf_counter() - campaign_start
        remaining = len(pending) - run_idx - 1
        avg_run = sum(run_durations) / len(run_durations)
        eta = avg_run * remaining
        print_campaign_status(global_done, total_runs, elapsed_total, eta)

    campaign_elapsed = time.perf_counter() - campaign_start
    print()
    print("=" * 60)
    print(
        f"Campaign finished.  {total_runs}/{total_runs} runs (100%).  "
        f"Total time: {fmt_duration(campaign_elapsed)}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
