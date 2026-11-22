| Optimizer | Condition | Fixed % | Policy % (338 / 339) | Policy − fixed pp (338 / 339) | Seed SD pp | Seed wins | 95% bootstrap CI pp | Episode win rate | Failed test (fixed / policy) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SGD | shuffled | 29.65 | 27.37 (26.88 / 27.86) | -2.28 (-2.77 / -1.79) | 0.69 | 0/2 | [-3.36, -1.23] | 0.14 | 0 / 0 |
| SGD | biased mean | 25.49 | 22.42 (22.03 / 22.82) | -3.07 (-3.47 / -2.67) | 0.56 | 0/2 | [-3.87, -2.21] | 0.17 | 0 / 0 |
| Adam | shuffled | 29.65 | 24.08 (24.33 / 23.82) | -5.57 (-5.32 / -5.83) | 0.36 | 0/2 | [-6.67, -4.45] | 0.00 | 0 / 0 |
| Adam | biased mean | 26.10 | 20.54 (20.83 / 20.26) | -5.56 (-5.28 / -5.84) | 0.40 | 0/2 | [-6.45, -4.71] | 0.06 | 0 / 0 |
| AdamW | shuffled | 29.69 | 24.30 (24.32 / 24.27) | -5.40 (-5.37 / -5.42) | 0.03 | 0/2 | [-6.75, -3.97] | 0.00 | 0 / 0 |
| AdamW | biased mean | 26.08 | 21.37 (21.14 / 21.60) | -4.71 (-4.94 / -4.49) | 0.32 | 0/2 | [-5.48, -3.96] | 0.07 | 0 / 0 |
| RMSprop | shuffled | 29.98 | 28.21 (28.54 / 27.87) | -1.77 (-1.44 / -2.11) | 0.47 | 0/2 | [-2.58, -0.91] | 0.15 | 0 / 0 |
| RMSprop | biased mean | 27.30 | 26.32 (26.62 / 26.02) | -0.98 (-0.67 / -1.28) | 0.43 | 0/2 | [-1.54, -0.42] | 0.29 | 0 / 0 |

| Optimizer | Tuned fixed configuration | Selected checkpoint (338 / 339) | Failed training episodes (338 / 339, of 4000 each) | Failed dev episodes | Train minutes (338 / 339) |
|---|---|---:|---:|---:|---:|
| SGD | lr=0.03, momentum=0.25, weight_decay=0.0001 | 100 / 300 | 7 / 6 | 0 / 0 | 102 / 101 |
| Adam | lr=0.001, weight_decay=0.0, beta1=0.5, beta2=0.9 | 400 / 500 | 4 / 3 | 0 / 0 | 152 / 154 |
| AdamW | lr=0.001, weight_decay=0.01, beta1=0.5, beta2=0.9 | 400 / 500 | 1 / 0 | 0 / 0 | 149 / 149 |
| RMSprop | lr=0.0003, weight_decay=0.0, momentum=0.5, alpha=0.9 | 400 / 500 | 26 / 34 | 0 / 0 | 151 / 151 |
