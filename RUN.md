# How to Run the Project From Scratch

All commands are run from the **project root** (`reinforcement_learing_car_race_project/`).

## 1. Dependencies

Python 3.11+ with:

- numpy, matplotlib, scipy
- gymnasium
- stable-baselines3
- torch (installed with SB3)
- pytest (for tests)

Example:

```bash
pip install numpy matplotlib scipy gymnasium stable-baselines3 pytest
```

## 2. (Optional) Clean old results

To start completely fresh:

```bash
rm -rf experiments/runs/*
rm -f experiments/figures/*.png experiments/figures/results_table.md
```

Old `[512,512,512]` runs (if any) can be removed with:

```bash
rm -rf experiments/runs/arch512_512_512_seed*
```

## 3. Run tests

```bash
pytest tests/ -v
```

## 4. Train all experiments

Grid: **3 architectures × 4 seeds = 12 runs**

| Architecture | Seeds |
|--------------|-------|
| `[16]` | 0, 1, 2, 3 |
| `[64, 64]` | 0, 1, 2, 3 |
| `[256, 256]` | 0, 1, 2, 3 |

Each run: 1,000,000 steps, fixed track, 8 parallel envs.

```bash
# See the plan without training
python run_experiments.py --dry-run

# Full campaign (several hours depending on CPU)
python run_experiments.py
```

Quick smoke test (shorter):

```bash
python run_experiments.py --total-steps 100000
```

**Resumable:** a run is skipped if `experiments/runs/{run-name}/final_model.zip` already exists.

### Train a single run manually

```bash
python -m src.train --net-arch 64,64 --seed 0 --total-steps 1000000
```

Run name defaults to `arch64_64_seed0_fixed`.

## 5. Evaluate and plot

After training, build all report figures:

```bash
python -m src.plots
```

Outputs (fixed filenames):

- `experiments/figures/learning_curves.png`
- `experiments/figures/final_performance.png`
- `experiments/figures/convergence_episodes.png`
- `experiments/figures/convergence_wallclock.png`
- `experiments/figures/sigma_decay.png`
- `experiments/figures/results_table.md`

Skip re-evaluation if `eval_results.json` already exists:

```bash
python -m src.plots --no-eval
```

Evaluate one run only:

```bash
python -m src.evaluate --run-name arch64_64_seed0_fixed --n-episodes 20
```

## 6. Visual checks (optional)

```bash
# Keyboard / scripted driving test
python -m src.manual_drive

# Render a trained policy
python -m src.evaluate --run-name arch64_64_seed0_fixed
```

## 7. Update the report

1. Copy the table from `experiments/figures/results_table.md` into `report/report.md` (Summary table section), or keep placeholders until plots finish.
2. Build HTML preview:

```bash
./report/build_html.sh
```

3. Build PDF (needs `pdflatex`):

```bash
./report/build_pdf.sh
```

## Typical full pipeline (copy-paste)

```bash
cd /path/to/reinforcement_learing_car_race_project
pytest tests/ -v
python run_experiments.py --dry-run
python run_experiments.py
python -m src.plots
./report/build_html.sh
```
