# How to Run the Project

All commands from the **project root**.

## 1. Install

```bash
pip install numpy matplotlib scipy gymnasium stable-baselines3 pytest
```

## 2. Test

```bash
pytest tests/ -v
```

## 3. Train (12 runs)

Default: **random** circuits each episode.

```bash
python run_experiments.py
```

Fixed circuit (same track every episode):

```bash
python run_experiments.py --track fixed
```

Run names: `arch64_64_seed0_random` or `arch64_64_seed0_fixed`.  
Resumable: skips runs that already have `final_model.zip`.

Quick smoke test:

```bash
python run_experiments.py --total-steps 100000
```

## 4. Plots (eval + all figures)

Match the track mode used during training:

```bash
python -m src.plots                  # random runs (default)
python -m src.plots --track fixed    # fixed runs
```

Figures: learning curves, final performance, convergence, sigma decay, results table, generalization.

Use cached eval only (faster, no new evaluate runs):

```bash
python -m src.plots --no-eval
```

## 5. Report (optional)

```bash
./report/build_html.sh
./report/build_pdf.sh
```

## Full pipeline

```bash
pytest tests/ -v
python run_experiments.py
python -m src.plots
```

## Manual debug (optional)

`evaluate.py` is used automatically by `plots`. For a single check:

```bash
python -m src.evaluate --run-name arch64_64_seed0_random
python -m src.evaluate --run-name arch64_64_seed0_random --track-seed 1000
```
