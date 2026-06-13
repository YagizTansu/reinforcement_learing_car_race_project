# Experiment results

**Convergence** = first episode where 50-episode rolling-mean return ≥ 90% of the run's own final 50-episode rolling mean.

| Net arch | Final return | Completion | Lap time (steps) | Lap time (sim s) | Episodes to converge | Wall-clock to converge (s) | Wall-clock / 1k steps (s) |
|---|---|---|---|---|---|---|---|
| [16] | 299.0 ± 0.0 | 0.0% | N/A | N/A s | 225 ± 0 | 71 ± 0 | 0.828 ± 0.000 |
| [64,64] | 1138.9 ± 0.0 | 100.0% | 714 | 35.70 s | 332 ± 0 | 162 ± 0 | 0.770 ± 0.000 |
| [256,256] | 1138.9 ± 0.0 | 100.0% | 711 | 35.55 s | 327 ± 0 | 248 ± 0 | 0.946 ± 0.000 |
| [512,512,512] | nan ± nan | nan% | N/A | N/A s | N/A | N/A | N/A |
