### Regular suite: mean final accuracy (%)

Biased = equal-weight mean over non-shuffled conditions; seeded rows give policy-seed SD of the biased mean.

| Method | SGD biased | Adam biased | AdamW biased | RMSprop biased | SGD shuffled | Adam shuffled | AdamW shuffled | RMSprop shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed (saved) | 88.36 | 91.50 | 91.50 | 92.37 | 92.27 | 92.63 | 92.63 | 93.19 |
| fixed (same engine) | 87.95 (1 failed) | 91.50 | 91.50 | 92.37 | 92.26 | 92.64 | 92.64 | 93.20 |
| policy (saved, 5 seeds) | 89.61 ± 0.51 | 90.31 ± 0.66 | 90.34 ± 0.80 | 92.57 ± 0.07 | 92.48 | 92.72 | 92.67 | 93.56 |
| dev-mean schedule (5 seeds) | 88.71 ± 0.33 (1 failed) | 90.14 ± 0.62 | 90.08 ± 0.91 | 92.49 ± 0.08 | 92.51 | 92.83 | 92.72 | 93.47 |
| cosine schedule | 88.93 | 93.21 | 93.21 | 93.28 | 93.14 | 94.05 | 94.05 | 94.02 |
| step schedule | 87.50 (1 failed) | 92.08 | 91.78 | 92.52 | 92.70 | 93.24 | 92.75 | 93.45 |
| warmup_cosine schedule | 88.32 (1 failed) | 93.36 | 93.37 | 93.40 | 93.24 | 94.29 | 94.30 | 94.27 |
| tuned schedule (dev-best kind) | 88.93 | 93.21 | 93.21 | 93.28 | 93.14 | 94.05 | 94.05 | 94.02 |

### Regular suite: paired differences, biased mean (pp) [95% CI]; seeds beating

| A − B | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| policy (saved, 5 seeds) − dev-mean schedule (5 seeds) | +0.90 [+0.39, +1.55]; 5/5 | +0.17 [-0.18, +0.67]; 2/5 | +0.26 [-0.13, +0.71]; 3/5 | +0.08 [-0.01, +0.17]; 3/5 |
| dev-mean schedule (5 seeds) − fixed (saved) | +0.34 [-0.11, +0.71]; 4/5 | -1.36 [-1.93, -0.80]; 0/5 | -1.42 [-2.14, -0.62]; 1/5 | +0.12 [+0.01, +0.23]; 5/5 |
| policy (saved, 5 seeds) − tuned schedule (dev-best kind) | +0.68 [+0.13, +1.21]; 5/5 | -2.91 [-3.52, -2.34]; 0/5 | -2.87 [-3.57, -2.18]; 0/5 | -0.71 [-0.86, -0.57]; 0/5 |
| tuned schedule (dev-best kind) − fixed (saved) | +0.56 [+0.08, +1.08] | +1.71 [+1.56, +1.86] | +1.71 [+1.56, +1.86] | +0.91 [+0.79, +1.02] |
| tuned schedule (dev-best kind) − fixed (same engine) | +0.98 [+0.15, +2.10] | +1.71 [+1.56, +1.86] | +1.71 [+1.56, +1.86] | +0.91 [+0.79, +1.02] |
| policy (saved, 5 seeds) − cosine schedule | +0.68 [+0.13, +1.21]; 5/5 | -2.91 [-3.52, -2.34]; 0/5 | -2.87 [-3.57, -2.18]; 0/5 | -0.71 [-0.86, -0.57]; 0/5 |
| policy (saved, 5 seeds) − step schedule | +2.11 [+1.31, +3.08]; 5/5 | -1.78 [-2.39, -1.20]; 0/5 | -1.44 [-2.13, -0.75]; 0/5 | +0.06 [-0.09, +0.19]; 4/5 |
| policy (saved, 5 seeds) − warmup_cosine schedule | +1.29 [+0.49, +2.40]; 5/5 | -3.06 [-3.69, -2.47]; 0/5 | -3.03 [-3.74, -2.33]; 0/5 | -0.83 [-0.98, -0.70]; 0/5 |
| cosine schedule − fixed (saved) | +0.56 [+0.08, +1.08] | +1.71 [+1.56, +1.86] | +1.71 [+1.56, +1.86] | +0.91 [+0.79, +1.02] |
| step schedule − fixed (saved) | -0.86 [-1.69, -0.17] | +0.58 [+0.39, +0.76] | +0.28 [+0.09, +0.46] | +0.15 [+0.03, +0.25] |
| warmup_cosine schedule − fixed (saved) | -0.05 [-1.02, +0.69] | +1.86 [+1.67, +2.05] | +1.87 [+1.66, +2.07] | +1.03 [+0.91, +1.16] |
| fixed (same engine) − fixed (saved) | -0.42 [-1.26, +0.02] | -0.00 [-0.00, +0.00] | -0.00 [-0.00, +0.00] | -0.00 [-0.00, +0.00] |

### Extreme suite: mean final accuracy (%)

Biased = equal-weight mean over non-shuffled conditions; seeded rows give policy-seed SD of the biased mean.

| Method | SGD biased | Adam biased | AdamW biased | RMSprop biased | SGD shuffled | Adam shuffled | AdamW shuffled | RMSprop shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed (saved) | 47.55 | 63.36 | 63.36 | 67.00 | 92.03 | 92.26 | 92.26 | 93.37 |
| fixed (same engine) | 45.18 (21 failed) | 63.35 | 63.35 | 67.01 | 92.00 | 92.26 | 92.26 | 93.37 |
| policy (saved, 5 seeds) | 51.66 ± 1.23 (1 failed) | 46.05 ± 10.02 (12 failed) | 49.04 ± 10.77 | 67.56 ± 0.92 | 92.55 | 92.81 | 92.76 | 93.64 |
| dev-mean schedule (5 seeds) | 48.11 ± 2.85 (38 failed) | 49.90 ± 10.43 | 51.53 ± 10.93 | 66.40 ± 1.71 | 92.55 | 92.93 | 92.81 | 93.56 |
| cosine schedule | 47.66 (15 failed) | 65.82 | 65.82 | 68.41 | 93.19 | 94.07 | 94.07 | 94.14 |
| step schedule | 51.16 (17 failed) | 61.20 | 64.89 | 64.50 | 92.71 | 93.23 | 92.90 | 93.53 |
| warmup_cosine schedule | 45.83 (16 failed) | 64.36 | 64.61 | 66.31 | 93.30 | 94.33 | 94.36 | 94.31 |
| tuned schedule (dev-best kind) | 47.66 (15 failed) | 65.82 | 65.82 | 68.41 | 93.19 | 94.07 | 94.07 | 94.14 |

### Extreme suite: paired differences, biased mean (pp) [95% CI]; seeds beating

| A − B | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| policy (saved, 5 seeds) − dev-mean schedule (5 seeds) | +3.55 [+1.55, +6.41]; 5/5 | -3.85 [-7.04, +0.48]; 1/5 | -2.49 [-5.26, +1.53]; 1/5 | +1.16 [+0.41, +1.95]; 5/5 |
| dev-mean schedule (5 seeds) − fixed (saved) | +0.56 [-2.01, +2.81]; 3/5 | -13.46 [-21.63, -5.42]; 0/5 | -11.83 [-20.55, -3.84]; 0/5 | -0.60 [-2.13, +0.68]; 3/5 |
| policy (saved, 5 seeds) − tuned schedule (dev-best kind) | +4.00 [+2.49, +5.39]; 5/5 | -19.76 [-27.83, -12.25]; 0/5 | -16.78 [-25.80, -9.03]; 0/5 | -0.84 [-1.98, +0.17]; 0/5 |
| tuned schedule (dev-best kind) − fixed (saved) | +0.10 [-1.19, +1.54] | +2.45 [+1.73, +3.19] | +2.45 [+1.73, +3.19] | +1.41 [+0.68, +2.18] |
| tuned schedule (dev-best kind) − fixed (same engine) | +2.48 [+1.32, +3.67] | +2.47 [+1.74, +3.21] | +2.47 [+1.74, +3.21] | +1.39 [+0.67, +2.17] |
| policy (saved, 5 seeds) − cosine schedule | +4.00 [+2.49, +5.39]; 5/5 | -19.76 [-27.83, -12.25]; 0/5 | -16.78 [-25.80, -9.03]; 0/5 | -0.84 [-1.98, +0.17]; 0/5 |
| policy (saved, 5 seeds) − step schedule | +0.50 [-1.21, +2.11]; 3/5 | -15.15 [-23.24, -7.59]; 0/5 | -15.85 [-24.93, -8.14]; 0/5 | +3.06 [+2.04, +3.99]; 5/5 |
| policy (saved, 5 seeds) − warmup_cosine schedule | +5.83 [+4.30, +7.19]; 5/5 | -18.31 [-26.36, -10.80]; 0/5 | -15.57 [-24.69, -7.83]; 0/5 | +1.25 [+0.27, +2.15]; 4/5 |
| cosine schedule − fixed (saved) | +0.10 [-1.19, +1.54] | +2.45 [+1.73, +3.19] | +2.45 [+1.73, +3.19] | +1.41 [+0.68, +2.18] |
| step schedule − fixed (saved) | +3.61 [+1.91, +5.36] | -2.16 [-3.25, -1.23] | +1.53 [+0.67, +2.33] | -2.50 [-3.18, -1.85] |
| warmup_cosine schedule − fixed (saved) | -1.73 [-2.77, -0.56] | +1.00 [+0.18, +1.86] | +1.25 [+0.34, +2.07] | -0.68 [-1.40, +0.00] |
| fixed (same engine) − fixed (saved) | -2.38 [-3.21, -1.60] | -0.02 [-0.05, +0.00] | -0.02 [-0.05, +0.00] | +0.01 [+0.00, +0.03] |
