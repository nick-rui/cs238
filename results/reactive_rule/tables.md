### Regular suite: mean final accuracy (%)

Biased = equal-weight mean over non-shuffled conditions; seeded rows give policy-seed SD of the biased mean.

| Method | SGD biased | Adam biased | AdamW biased | RMSprop biased | SGD shuffled | Adam shuffled | AdamW shuffled | RMSprop shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed (saved) | 88.36 | 91.50 | 91.50 | 92.37 | 92.27 | 92.63 | 92.63 | 93.19 |
| fixed (same engine) | 87.95 (1 failed) | 91.50 | 91.50 | 92.37 | 92.26 | 92.64 | 92.64 | 93.20 |
| reactive rule | 87.95 (1 failed) | 91.50 | 91.50 | 92.37 | 92.26 | 92.64 | 92.64 | 93.20 |
| firing rule (secondary) | 88.59 | 91.50 | 91.50 | – | 91.73 | 92.24 | 92.24 | – |
| policy (saved, 5 seeds) | 89.61 ± 0.51 | 90.31 ± 0.66 | 90.34 ± 0.80 | 92.57 ± 0.07 | 92.48 | 92.72 | 92.67 | 93.56 |
| warm-start policy (5 seeds, fresh) | 90.86 ± 0.38 | 92.69 ± 0.26 | 92.66 ± 0.29 | 92.63 ± 0.08 | 93.07 | 93.61 | 93.57 | 93.52 |
| fixed (warm-start fresh replay) | 88.34 | 91.51 | 91.51 | 92.37 | 92.25 | 92.63 | 92.63 | 93.19 |

### Regular suite: paired differences, biased mean (pp) [95% CI]; seeds beating

| A − B | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| reactive rule − fixed (saved) | -0.42 [-1.26, +0.02] | -0.00 [-0.00, +0.00] | -0.00 [-0.00, +0.00] | +0.00 [-0.01, +0.01] |
| reactive rule − fixed (same engine) | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [-0.00, +0.01] |
| policy (saved, 5 seeds) − reactive rule | +1.66 [+0.84, +2.81]; 5/5 | -1.20 [-1.81, -0.62]; 0/5 | -1.16 [-1.85, -0.47]; 0/5 | +0.20 [+0.10, +0.30]; 5/5 |
| warm-start policy (5 seeds, fresh) − reactive rule | +2.91 [+2.06, +4.12]; 5/5 | +1.19 [+0.94, +1.44]; 5/5 | +1.16 [+0.90, +1.44]; 5/5 | +0.26 [+0.18, +0.34]; 5/5 |
| reactive rule − tuned schedule (dev-best kind) | -0.98 [-2.10, -0.15] | -1.71 [-1.86, -1.56] | -1.71 [-1.86, -1.56] | -0.91 [-1.02, -0.79] |
| firing rule (secondary) − fixed (same engine) | +0.64 [-0.08, +1.70] | -0.01 [-0.12, +0.08] | -0.01 [-0.12, +0.08] | – |
| policy (saved, 5 seeds) − firing rule (secondary) | +1.02 [+0.57, +1.50]; 5/5 | -1.19 [-1.81, -0.63]; 0/5 | -1.15 [-1.83, -0.46]; 0/5 | – |
| fixed (same engine) − fixed (saved) | -0.42 [-1.26, +0.02] | -0.00 [-0.00, +0.00] | -0.00 [-0.00, +0.00] | -0.00 [-0.00, +0.00] |

### Extreme suite: mean final accuracy (%)

Biased = equal-weight mean over non-shuffled conditions; seeded rows give policy-seed SD of the biased mean.

| Method | SGD biased | Adam biased | AdamW biased | RMSprop biased | SGD shuffled | Adam shuffled | AdamW shuffled | RMSprop shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed (saved) | 47.55 | 63.36 | 63.36 | 67.00 | 92.03 | 92.26 | 92.26 | 93.37 |
| fixed (same engine) | 45.18 (21 failed) | 63.35 | 63.35 | 67.01 | 92.00 | 92.26 | 92.26 | 93.37 |
| reactive rule | 45.18 (21 failed) | 63.35 | 63.35 | 67.01 | 92.00 | 92.26 | 92.26 | 93.37 |
| firing rule (secondary) | 48.44 (15 failed) | 64.09 | 64.09 | – | 91.40 | 92.41 | 92.41 | – |
| policy (saved, 5 seeds) | 51.66 ± 1.23 (1 failed) | 46.05 ± 10.02 (12 failed) | 49.04 ± 10.77 | 67.56 ± 0.92 | 92.55 | 92.81 | 92.76 | 93.64 |
| warm-start policy (5 seeds, fresh) | 50.09 ± 3.82 (41 failed) | 66.41 ± 2.76 | 66.98 ± 2.03 | 66.94 ± 2.31 | 93.09 | 93.72 | 93.68 | 93.62 |
| fixed (warm-start fresh replay) | 47.59 | 63.36 | 63.36 | 67.00 | 92.02 | 92.26 | 92.26 | 93.37 |

### Extreme suite: paired differences, biased mean (pp) [95% CI]; seeds beating

| A − B | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| reactive rule − fixed (saved) | -2.38 [-3.21, -1.60] | -0.02 [-0.05, +0.00] | -0.02 [-0.05, +0.00] | +0.01 [-0.02, +0.04] |
| reactive rule − fixed (same engine) | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.00 [-0.03, +0.02] |
| policy (saved, 5 seeds) − reactive rule | +6.48 [+5.10, +7.75]; 5/5 | -17.29 [-25.45, -9.79]; 0/5 | -14.31 [-23.37, -6.61]; 0/5 | +0.55 [-0.33, +1.23]; 4/5 |
| warm-start policy (5 seeds, fresh) − reactive rule | +4.91 [+1.48, +7.71]; 4/5 | +3.06 [+0.86, +5.38]; 5/5 | +3.63 [+1.96, +5.38]; 5/5 | -0.07 [-2.07, +1.61]; 3/5 |
| reactive rule − tuned schedule (dev-best kind) | -2.48 [-3.67, -1.32] | -2.47 [-3.21, -1.74] | -2.47 [-3.21, -1.74] | -1.40 [-2.18, -0.67] |
| firing rule (secondary) − fixed (same engine) | +3.26 [+2.52, +3.98] | +0.74 [+0.30, +1.18] | +0.74 [+0.30, +1.18] | – |
| policy (saved, 5 seeds) − firing rule (secondary) | +3.22 [+1.82, +4.53]; 5/5 | -18.03 [-26.15, -10.54]; 0/5 | -15.05 [-24.09, -7.34]; 0/5 | – |
| fixed (same engine) − fixed (saved) | -2.38 [-3.21, -1.60] | -0.02 [-0.05, +0.00] | -0.02 [-0.05, +0.00] | +0.01 [+0.00, +0.03] |
