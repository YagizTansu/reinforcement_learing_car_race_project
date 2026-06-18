# Experiment results

**Convergence** = first episode where 50-episode rolling-mean return ≥ 90% of the run's own final 50-episode rolling mean.

| Net arch | Final return | Completion | Lap time (steps) | Lap time (sim s) | Episodes to converge | Wall-clock to converge (s) | Wall-clock / 1k steps (s) |
|---|---|---|---|---|---|---|---|
| [16] | 1139.2 ± 0.5 | 100.0% | 725 | 36.25 s | 333 ± 23 | 115 ± 8 | 0.636 ± 0.008 |
| [64,64] | 1139.4 ± 0.5 | 100.0% | 711 | 35.56 s | 284 ± 52 | 126 ± 25 | 0.686 ± 0.003 |
| [256,256] | 1139.4 ± 0.5 | 100.0% | 712 | 35.59 s | 326 ± 132 | 178 ± 63 | 0.786 ± 0.041 |
