# Car Race — 2D Racing with PPO

A reinforcement learning project that trains a car to complete a 2D racetrack lap in **minimum time** using **Proximal Policy Optimization (PPO)**.

**Research question:** How does policy network size (`[16]`, `[64,64]`, `[256,256]`) affect (1) final performance, (2) episodes to convergence, and (3) wall-clock training time?

## Overview

- **Environment:** Custom Gymnasium env — kinematic bicycle model, Frenet coordinates
- **State:** Lateral offset, heading error, speed, lookahead curvature (14 dims)
- **Action:** Throttle/brake + steering (continuous, 2 dims)
- **Algorithm:** PPO with a Gaussian MLP policy (Stable-Baselines3)
- **Training:** A new procedural track every episode (`--track random`, seeds 0–49)
- **Testing:** Reference circuit + **held-out** tracks never seen in training (seeds 1000–1009)

```
src/
  track.py, car.py, frenet.py, env.py   # simulation
  train.py, evaluate.py, plots.py       # training & analysis
  render.py
run_experiments.py                      # 12-run campaign
tests/
experiments/
  runs/                                 # models & logs (gitignored)
  figures/                              # plots
report/                                 # PDF report
```

## Setup

Python 3.11+

```bash
pip install numpy matplotlib scipy gymnasium stable-baselines3 pytest
```

## How to run

All commands from the project root.

### 1. Tests

```bash
pytest tests/ -v
```

### 2. Training (12 runs)

3 architectures × 4 seeds, 1M steps per run, random track each episode:

```bash
python run_experiments.py
```

Runs are resumable — folders that already contain `final_model.zip` are skipped.

Quick smoke test:

```bash
python run_experiments.py --total-steps 100000
```

Fixed-track training (optional):

```bash
python run_experiments.py --track fixed
python -m src.plots --track fixed
```

### 3. Plots

Runs any missing evaluations and writes all figures:

```bash
python -m src.plots
```

Output: `experiments/figures/` (learning curves, final performance, convergence, generalization, etc.)

Plot from cached results only (no re-evaluation):

```bash
python -m src.plots --no-eval
```

### Full pipeline

```bash
pytest tests/ -v
python run_experiments.py
python -m src.plots
```

## Single run (manual)

```bash
python -m src.train --net-arch 64,64 --seed 0 --track random --total-steps 1000000
```

Default run name: `arch64_64_seed0_random`

### Evaluate one model on one track

From the project root. Uses deterministic actions (Gaussian mean).

**Reference circuit** (hand-designed benchmark, same for all runs):

```bash
python -m src.evaluate --run-name arch16_seed3_random --n-episodes 20
```

**Procedural track** by seed (e.g. held-out seed 1000):

```bash
python -m src.evaluate --run-name arch16_seed3_random --n-episodes 20 --track-seed 1000
```

**Training-pool track** (seeds 0–49, seen during random training):

```bash
python -m src.evaluate --run-name arch16_seed3_random --n-episodes 20 --track-seed 42
```

Results are written under `experiments/runs/<run-name>/`:

| Track | JSON | Trajectory PNG |
|-------|------|----------------|
| Reference (no `--track-seed`) | `eval_results.json` | `best_trajectory.png` |
| Procedural (`--track-seed N`) | `eval_track_seedN.json` | `best_trajectory_seedN.png` |

Optional: plot training reward curve from `progress.csv`:

```bash
python -m src.evaluate --run-name arch16_seed3_random --n-episodes 20 --plot-curve
```
