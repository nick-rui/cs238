| Comparison | Suite (scope) | Optimizer | Comparator official → clean | Method official → clean (mean ± SD) | Delta official → clean (pp) | Wins official / clean |
|---|---|---|---:|---:|---:|---:|
| frozen policy vs fixed | regular (biased) | SGD | 88.35 → 88.16 | 89.61 ± 0.51 → 89.40 ± 0.52 | +1.26 → +1.24 | 5/5 / 5/5 |
| frozen policy vs fixed | regular (biased) | Adam | 91.50 → 91.49 | 90.31 ± 0.66 → 90.22 ± 0.67 | -1.20 → -1.27 | 0/5 / 0/5 |
| frozen policy vs fixed | regular (biased) | AdamW | 91.50 → 91.49 | 90.35 ± 0.80 → 90.26 ± 0.82 | -1.16 → -1.23 | 0/5 / 0/5 |
| frozen policy vs fixed | regular (biased) | RMSprop | 92.37 → 92.26 | 92.57 ± 0.07 → 92.53 ± 0.09 | +0.20 → +0.27 | 5/5 / 5/5 |
| frozen policy vs fixed | regular (shuffled) | SGD | 92.26 → 92.20 | 92.48 ± 0.42 → 92.38 ± 0.49 | +0.23 → +0.18 | 3/5 / 3/5 |
| frozen policy vs fixed | regular (shuffled) | Adam | 92.64 → 92.59 | 92.72 ± 0.24 → 92.74 ± 0.24 | +0.09 → +0.15 | 4/5 / 4/5 |
| frozen policy vs fixed | regular (shuffled) | AdamW | 92.64 → 92.59 | 92.67 ± 0.38 → 92.71 ± 0.39 | +0.03 → +0.12 | 2/5 / 3/5 |
| frozen policy vs fixed | regular (shuffled) | RMSprop | 93.20 → 93.17 | 93.56 ± 0.11 → 93.60 ± 0.13 | +0.36 → +0.42 | 5/5 / 5/5 |
| frozen policy vs fixed | extreme (biased) | SGD | 47.46 → 47.36 | 51.63 ± 1.23 → 51.43 ± 1.23 | +4.18 → +4.07 | 5/5 / 5/5 |
| frozen policy vs fixed | extreme (biased) | Adam | 63.35 → 63.27 | 46.05 ± 10.01 → 45.95 ± 9.96 | -17.30 → -17.31 | 0/5 / 0/5 |
| frozen policy vs fixed | extreme (biased) | AdamW | 63.35 → 63.27 | 49.07 ± 10.78 → 49.02 ± 10.73 | -14.27 → -14.25 | 0/5 / 0/5 |
| frozen policy vs fixed | extreme (biased) | RMSprop | 67.01 → 66.87 | 67.56 ± 0.91 → 67.44 ± 0.89 | +0.55 → +0.57 | 4/5 / 4/5 |
| retrained vs extreme-retuned fixed | extreme (biased) | SGD | 61.32 → 61.03 | 67.48 ± 1.86 → 67.06 ± 1.86 | +6.17 → +6.03 | 5/5 / 5/5 |
| retrained vs extreme-retuned fixed | extreme (biased) | Adam | 65.91 → 65.63 | 69.22 ± 2.81 → 68.85 ± 2.73 | +3.31 → +3.22 | 5/5 / 5/5 |
| retrained vs extreme-retuned fixed | extreme (biased) | AdamW | 65.91 → 65.63 | 69.13 ± 3.47 → 68.76 ± 3.39 | +3.22 → +3.13 | 4/5 / 4/5 |
| retrained vs extreme-retuned fixed | extreme (biased) | RMSprop | 70.22 → 69.92 | 72.98 ± 1.05 → 72.60 ± 1.02 | +2.75 → +2.68 | 5/5 / 5/5 |
| retrained vs saved fixed | extreme (biased) | SGD | 47.46 → 47.36 | 67.48 ± 1.86 → 67.06 ± 1.86 | +20.03 → +19.70 | 5/5 / 5/5 |
| retrained vs saved fixed | extreme (biased) | Adam | 63.35 → 63.27 | 69.22 ± 2.81 → 68.85 ± 2.73 | +5.87 → +5.59 | 5/5 / 5/5 |
| retrained vs saved fixed | extreme (biased) | AdamW | 63.35 → 63.27 | 69.13 ± 3.47 → 68.76 ± 3.39 | +5.78 → +5.50 | 5/5 / 5/5 |
| retrained vs saved fixed | extreme (biased) | RMSprop | 67.01 → 66.87 | 72.98 ± 1.05 → 72.60 ± 1.02 | +5.96 → +5.73 | 5/5 / 5/5 |
| retrained vs frozen (seed-matched) | extreme (biased) | SGD | 51.63 → 51.43 | 67.48 ± 1.86 → 67.06 ± 1.86 | +15.85 → +15.62 | 5/5 / 5/5 |
| retrained vs frozen (seed-matched) | extreme (biased) | Adam | 46.05 → 45.95 | 69.22 ± 2.81 → 68.85 ± 2.73 | +23.17 → +22.90 | 5/5 / 5/5 |
| retrained vs frozen (seed-matched) | extreme (biased) | AdamW | 49.07 → 49.02 | 69.13 ± 3.47 → 68.76 ± 3.39 | +20.05 → +19.75 | 5/5 / 5/5 |
| retrained vs frozen (seed-matched) | extreme (biased) | RMSprop | 67.56 → 67.44 | 72.98 ± 1.05 → 72.60 ± 1.02 | +5.42 → +5.16 | 5/5 / 5/5 |
| warm vs fixed | regular (biased) | SGD | 88.35 → 88.16 | 90.86 ± 0.38 → 90.73 ± 0.38 | +2.52 → +2.57 | 5/5 / 5/5 |
| warm vs fixed | regular (biased) | Adam | 91.50 → 91.49 | 92.69 ± 0.26 → 92.67 ± 0.27 | +1.18 → +1.18 | 5/5 / 5/5 |
| warm vs fixed | regular (biased) | AdamW | 91.50 → 91.49 | 92.66 ± 0.29 → 92.64 ± 0.30 | +1.15 → +1.15 | 5/5 / 5/5 |
| warm vs fixed | regular (biased) | RMSprop | 92.37 → 92.26 | 92.63 ± 0.08 → 92.58 ± 0.08 | +0.26 → +0.32 | 5/5 / 5/5 |
| warm vs fixed | extreme (all) | SGD | 53.03 → 52.94 | 55.48 ± 3.41 → 55.29 ± 3.41 | +2.45 → +2.35 | 4/5 / 4/5 |
| warm vs fixed | extreme (all) | Adam | 66.96 → 66.89 | 69.81 ± 2.42 → 69.70 ± 2.41 | +2.85 → +2.81 | 5/5 / 5/5 |
| warm vs fixed | extreme (all) | AdamW | 66.96 → 66.89 | 70.33 ± 1.73 → 70.22 ± 1.73 | +3.37 → +3.34 | 5/5 / 5/5 |
| warm vs fixed | extreme (all) | RMSprop | 70.31 → 70.18 | 70.29 ± 2.03 → 70.18 ± 2.03 | -0.02 → -0.00 | 3/5 / 3/5 |
| warm vs full (seeds 338-339) | regular (biased) | SGD | 89.12 → 88.90 | 90.91 ± 0.65 → 90.74 ± 0.67 | +1.79 → +1.84 | 2/2 / 2/2 |
| warm vs full (seeds 338-339) | regular (biased) | Adam | 90.69 → 90.58 | 92.55 ± 0.15 → 92.53 ± 0.15 | +1.86 → +1.95 | 2/2 / 2/2 |
| warm vs full (seeds 338-339) | regular (biased) | AdamW | 90.51 → 90.39 | 92.55 ± 0.02 → 92.52 ± 0.01 | +2.03 → +2.13 | 2/2 / 2/2 |
| warm vs full (seeds 338-339) | regular (biased) | RMSprop | 92.57 → 92.52 | 92.62 ± 0.13 → 92.57 ± 0.13 | +0.06 → +0.05 | 2/2 / 2/2 |
| open loop vs fixed (seeds 338-339) | regular (biased) | SGD | 88.35 → 88.16 | 88.66 ± 0.19 → 88.45 ± 0.23 | +0.31 → +0.28 | 2/2 / 2/2 |
| open loop vs fixed (seeds 338-339) | regular (biased) | Adam | 91.50 → 91.49 | 90.22 ± 0.52 → 90.09 ± 0.46 | -1.28 → -1.40 | 0/2 / 0/2 |
| open loop vs fixed (seeds 338-339) | regular (biased) | AdamW | 91.50 → 91.49 | 90.08 ± 0.38 → 89.94 ± 0.33 | -1.42 → -1.55 | 0/2 / 0/2 |
| open loop vs fixed (seeds 338-339) | regular (biased) | RMSprop | 92.37 → 92.26 | 92.53 ± 0.07 → 92.48 ± 0.08 | +0.16 → +0.22 | 2/2 / 2/2 |
| frozen policy vs open loop (seeds 338-339) | regular (biased) | SGD | 88.66 → 88.45 | 89.12 ± 0.23 → 88.90 ± 0.26 | +0.46 → +0.46 | 2/2 / 2/2 |
| frozen policy vs open loop (seeds 338-339) | regular (biased) | Adam | 90.22 → 90.09 | 90.69 ± 0.38 → 90.58 ± 0.45 | +0.47 → +0.49 | 1/2 / 1/2 |
| frozen policy vs open loop (seeds 338-339) | regular (biased) | AdamW | 90.08 → 89.94 | 90.51 ± 0.48 → 90.39 ± 0.54 | +0.43 → +0.45 | 1/2 / 1/2 |
| frozen policy vs open loop (seeds 338-339) | regular (biased) | RMSprop | 92.53 → 92.48 | 92.57 ± 0.13 → 92.52 ± 0.16 | +0.04 → +0.04 | 1/2 / 1/2 |
