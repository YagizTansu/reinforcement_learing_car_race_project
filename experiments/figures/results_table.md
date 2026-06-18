# Experiment results

**Convergence** = first episode where 50-episode rolling-mean return ≥ 90% of the run's own final 50-episode rolling mean.

| Net arch | Final return | Completion | Lap time (steps) | Lap time (sim s) | Episodes to converge | Wall-clock to converge (s) | Wall-clock / 1k steps (s) |
|---|---|---|---|---|---|---|---|
| [16] | 1139.3 ± 0.5 | 100.0% | 720 | 36.00 s | 261 ± 34 | 149 ± 16 | 0.773 ± 0.010 |
| [64,64] | 1138.8 ± 0.0 | 100.0% | 719 | 35.94 s | 348 ± 45 | 228 ± 40 | 0.773 ± 0.022 |
| [256,256] | 1139.3 ± 0.5 | 100.0% | 720 | 36.01 s | 327 ± 43 | 208 ± 30 | 0.738 ± 0.023 |
**Winning architecture (reference circuit, mean over seeds):** [16]
