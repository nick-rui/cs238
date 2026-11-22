### All eight held-out types

Mean accuracy (%) on the held-out type, averaged over types (policies: over seeds 338–339). Differences: warm minus control, pp, mean [seed 338, seed 339] (wins/2).

| Optimizer | Fold-retuned fixed | Cold held-out policy | **Warm held-out policy** | Warm − fixed | Warm − cold |
|---|---:|---:|---:|---:|---:|
| SGD | 64.76 | 65.15 | 66.75 | +1.99 [+2.64, +1.35] (2/2) | +1.60 [+4.84, -1.64] (1/2) |
| Adam | 68.19 | 75.26 | 77.52 | +9.33 [+9.32, +9.33] (2/2) | +2.26 [+3.60, +0.93] (2/2) |
| AdamW | 68.44 | 75.20 | 76.11 | +7.66 [+7.80, +7.53] (2/2) | +0.91 [+1.82, -0.01] (1/2) |
| RMSprop | 72.11 | 73.75 | 73.25 | +1.14 [+1.81, +0.47] (2/2) | -0.50 [+0.01, -1.01] (1/2) |

### Seven biased held-out types

Mean accuracy (%) on the held-out type, averaged over types (policies: over seeds 338–339). Differences: warm minus control, pp, mean [seed 338, seed 339] (wins/2).

| Optimizer | Fold-retuned fixed | Cold held-out policy | **Warm held-out policy** | Warm − fixed | Warm − cold |
|---|---:|---:|---:|---:|---:|
| SGD | 61.32 | 61.55 | 63.28 | +1.96 [+2.71, +1.21] (2/2) | +1.73 [+5.25, -1.78] (1/2) |
| Adam | 65.06 | 73.14 | 75.53 | +10.47 [+10.45, +10.49] (2/2) | +2.39 [+3.94, +0.83] (2/2) |
| AdamW | 65.35 | 73.05 | 73.93 | +8.58 [+8.72, +8.45] (2/2) | +0.88 [+1.93, -0.18] (1/2) |
| RMSprop | 69.47 | 71.39 | 70.68 | +1.21 [+1.89, +0.54] (2/2) | -0.71 [-0.17, -1.24] (0/2) |

### Per held-out type

Fixed / cold / warm accuracy (%), seed-averaged.

| Held-out type | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| mixed | 88.9 / 90.4 / 91.1 | 90.1 / 90.1 / 91.5 | 90.1 / 90.2 / 91.4 | 90.6 / 90.3 / 91.2 |
| single_max | 80.4 / 82.1 / 81.4 | 81.7 / 83.5 / 84.3 | 81.7 / 83.6 / 84.1 | 82.9 / 83.6 / 84.6 |
| single_late | 82.1 / 75.7 / 81.7 | 84.8 / 74.9 / 83.0 | 84.8 / 74.6 / 82.9 | 87.3 / 81.9 / 84.3 |
| pair_max | 73.4 / 80.0 / 82.6 | 77.2 / 82.3 / 82.9 | 77.7 / 82.4 / 82.4 | 80.9 / 85.0 / 83.7 |
| half_block | 49.3 / 63.9 / 71.2 | 50.4 / 75.2 / 79.0 | 51.9 / 75.9 / 70.6 | 55.0 / 59.4 / 56.1 |
| staircase | 75.3 / 45.2 / 41.6 | 84.0 / 78.5 / 80.5 | 84.0 / 77.8 / 79.8 | 86.6 / 85.1 / 87.1 |
| class_sorted | 9.9 / 18.6 / 18.5 | 12.2 / 34.8 / 41.2 | 12.2 / 35.8 / 40.2 | 20.4 / 25.5 / 22.2 |
| alternating_halves | 58.7 / 65.2 / 65.8 | 65.2 / 82.7 / 77.7 | 65.2 / 81.1 / 77.5 | 73.3 / 79.2 / 76.8 |
