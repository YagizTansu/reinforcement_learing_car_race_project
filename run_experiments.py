"""
run_experiments.py — Sequential experiment campaign launcher.

Grid
----
  net_archs = [[16], [64,64], [256,256]]
  seeds     = [0, 1, 2, 3]
  track     = fixed (Monaco-inspired)
  steps     = 1_000_000 per run
  n_envs    = 8 (parallel workers inside each run)

Naming convention
-----------------
  arch16_seed0_fixed
  arch64_64_seed0_fixed
  arch256_256_seed1_fixed

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
TRACK_MODE = "fixed"
DEFAULT_TOTAL_STEPS = 1_000_000
N_ENVS = 8


def arch_to_str(arch):
    """[64, 64] → '64_64'"""
    return "_".join(str(x) for x in arch)


def run_name_for(arch, seed, track=TRACK_MODE):
    return f"arch{arch_to_str(arch)}_seed{seed}_{track}"


def model_exists(run_name):
    path = os.path.join("experiments", "runs", run_name, "final_model.zip")
    return os.path.isfile(path)


def main():
    parser = argparse.ArgumentParser(description="Run the full experiment grid")
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS,
                        help="Steps per run (default 1_000_000)")
    parser.add_argument("--n-envs", type=int, default=N_ENVS,
                        help="Parallel envs per run (default 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan but do not train")
    args = parser.parse_args()

    # Build the full grid: 3 archs × 4 seeds = 12 runs
    runs = [
        (arch, seed)
        for arch in NET_ARCHS
        for seed in SEEDS
    ]

    print("=" * 60)
    print(f"Experiment grid: {len(runs)} runs")
    print(f"  net archs : {[arch_to_str(a) for a in NET_ARCHS]}")
    print(f"  seeds     : {SEEDS}")
    print(f"  steps/run : {args.total_steps:,}")
    print(f"  n_envs    : {args.n_envs}")
    print(f"  track     : {TRACK_MODE}")
    print("=" * 60)

    pending = []
    skipped = []
    for arch, seed in runs:
        name = run_name_for(arch, seed)
        if model_exists(name):
            skipped.append(name)
        else:
            pending.append((arch, seed, name))

    if skipped:
        print(f"\nSkipping {len(skipped)} already-completed run(s):")
        for n in skipped:
            print(f"  [done] {n}")

    print(f"\nRunning {len(pending)} run(s):")
    for _, _, name in pending:
        print(f"  [todo] {name}")

    if args.dry_run:
        print("\n--dry-run set, exiting without training.")
        return

    if not pending:
        print("\nAll runs already complete.")
        return

    # Import here so the module is only loaded when actually training
    from src.train import train as run_training

    campaign_start = time.perf_counter()

    for run_idx, (arch, seed, name) in enumerate(pending):
        elapsed_total = time.perf_counter() - campaign_start
        print()
        print("=" * 60)
        print(f"[{run_idx + 1}/{len(pending)}]  {name}")
        print(f"  arch={arch}  seed={seed}  "
              f"campaign elapsed={elapsed_total/60:.1f} min")
        print("=" * 60)

        try:
            run_training(
                net_arch=arch,
                seed=seed,
                total_steps=args.total_steps,
                track_mode=TRACK_MODE,
                run_name=name,
                n_envs=args.n_envs,
            )
        except Exception as exc:
            print(f"\n[ERROR] Run {name} failed: {exc}", file=sys.stderr)
            print("Continuing with next run...", file=sys.stderr)
            continue

    campaign_elapsed = time.perf_counter() - campaign_start
    print()
    print("=" * 60)
    print(f"Campaign finished.  Total time: {campaign_elapsed/3600:.2f} h")
    print("=" * 60)


if __name__ == "__main__":
    main()
