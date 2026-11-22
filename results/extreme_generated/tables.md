### All 200 generated orderings (N = 200)

Mean final test accuracy (%) ± policy-seed SD [min–max]; failures in parentheses (all orderings).

| Method | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| Fixed (regular-tuned) | 68.70 | 82.21 | 82.21 | 85.64 |
| Fixed (extreme-retuned) | 76.99 | 80.10 | 80.10 | 82.83 |
| Cosine (regular-tuned) | 76.53 (7 failed) | 89.01 | 89.01 | 89.60 |
| Frozen regular policy | 75.35 ± 0.90 [73.78–76.07] (16 failed) | 74.93 ± 7.39 [66.07–81.30] (36 failed) | 77.68 ± 6.14 [67.22–81.94] (2 failed) | 86.46 ± 0.82 [85.67–87.42] |
| Cold extreme policy | 81.04 ± 2.24 [78.13–83.07] | 82.42 ± 0.78 [81.73–83.63] | 82.39 ± 1.25 [80.93–83.90] | 85.34 ± 0.50 [84.83–86.07] |
| Extreme warm-start policy | 83.16 ± 0.39 [82.70–83.66] | 85.76 ± 0.59 [85.32–86.74] | 85.77 ± 0.54 [85.12–86.41] | 85.52 ± 2.25 [82.83–87.82] |

Paired difference A − B (pp), mean [95% two-level bootstrap CI]; share of orderings with A > B; policy seeds with positive mean difference (of 5) and policy-seed SD.

| A − B | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| Extreme warm-start policy − Fixed (regular-tuned) | +14.46 [+12.03, +17.04]; 88%; 5/5, SD 0.39 | +3.55 [+1.80, +5.43]; 38%; 5/5, SD 0.59 | +3.56 [+1.82, +5.47]; 38%; 5/5, SD 0.54 | -0.12 [-1.97, +1.80]; 21%; 3/5, SD 2.25 |
| Extreme warm-start policy − Fixed (extreme-retuned) | +6.17 [+4.69, +7.72]; 88%; 5/5, SD 0.39 | +5.65 [+4.52, +6.84]; 94%; 5/5, SD 0.59 | +5.66 [+4.56, +6.88]; 94%; 5/5, SD 0.54 | +2.69 [+0.94, +4.50]; 99%; 4/5, SD 2.25 |
| Extreme warm-start policy − Cosine (regular-tuned) | +6.63 [+4.64, +8.71]; 64%; 5/5, SD 0.39 | -3.25 [-4.28, -2.20]; 14%; 0/5, SD 0.59 | -3.25 [-4.29, -2.22]; 14%; 0/5, SD 0.54 | -4.08 [-6.13, -2.18]; 11%; 0/5, SD 2.25 |
| Extreme warm-start policy − Frozen regular policy | +7.81 [+5.77, +10.03]; 71%; 5/5, SD 0.69 | +10.83 [+4.58, +17.81]; 72%; 5/5, SD 7.88 | +8.08 [+3.77, +13.69]; 70%; 5/5, SD 6.04 | -0.94 [-3.28, +1.41]; 19%; 2/5, SD 2.99 |
| Extreme warm-start policy − Cold extreme policy | +2.12 [+0.63, +3.95]; 70%; 5/5, SD 1.94 | +3.33 [+2.37, +4.31]; 97%; 5/5, SD 1.01 | +3.38 [+2.46, +4.36]; 98%; 5/5, SD 0.99 | +0.18 [-1.88, +2.22]; 76%; 3/5, SD 2.57 |
| Cold extreme policy − Fixed (regular-tuned) | +12.34 [+9.58, +15.22]; 86%; 5/5, SD 2.24 | +0.22 [-1.43, +1.91]; 23%; 3/5, SD 0.78 | +0.18 [-1.70, +2.06]; 22%; 2/5, SD 1.25 | -0.30 [-1.45, +0.94]; 22%; 1/5, SD 0.50 |
| Cold extreme policy − Fixed (extreme-retuned) | +4.04 [+1.86, +6.19]; 82%; 5/5, SD 2.24 | +2.32 [+1.18, +3.44]; 81%; 5/5, SD 0.78 | +2.28 [+0.87, +3.66]; 76%; 5/5, SD 1.25 | +2.51 [+1.62, +3.51]; 86%; 5/5, SD 0.50 |
| Cold extreme policy − Cosine (regular-tuned) | +4.51 [+1.58, +7.19]; 60%; 5/5, SD 2.24 | -6.59 [-7.94, -5.29]; 7%; 0/5, SD 0.78 | -6.62 [-8.19, -5.15]; 6%; 0/5, SD 1.25 | -4.26 [-5.19, -3.36]; 10%; 0/5, SD 0.50 |
| Cold extreme policy − Frozen regular policy | +5.69 [+3.29, +8.13]; 62%; 5/5, SD 1.96 | +7.50 [+1.66, +14.11]; 58%; 5/5, SD 7.52 | +4.70 [+0.51, +9.79]; 52%; 4/5, SD 5.67 | -1.12 [-2.13, +0.05]; 19%; 0/5, SD 0.79 |
| Frozen regular policy − Fixed (regular-tuned) | +6.64 [+5.39, +7.97]; 96%; 5/5, SD 0.90 | -7.28 [-13.86, -1.67]; 26%; 0/5, SD 7.39 | -4.52 [-10.24, -0.60]; 28%; 0/5, SD 6.14 | +0.82 [+0.12, +1.66]; 72%; 5/5, SD 0.82 |
| Frozen regular policy − Cosine (regular-tuned) | -1.18 [-3.82, +1.43]; 51%; 0/5, SD 0.90 | -14.08 [-20.78, -8.17]; 2%; 0/5, SD 7.39 | -11.33 [-17.23, -6.91]; 4%; 0/5, SD 6.14 | -3.14 [-4.58, -1.83]; 18%; 0/5, SD 0.82 |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | +8.29 [+6.56, +10.16]; 70% | -2.10 [-3.00, -1.14]; 16% | -2.10 [-3.00, -1.14]; 16% | -2.81 [-3.28, -2.29]; 8% |

### Excluding near-duplicates of the seven originals (N = 198)

Mean final test accuracy (%) ± policy-seed SD [min–max]; failures in parentheses (all orderings).

| Method | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| Fixed (regular-tuned) | 68.89 | 82.48 | 82.48 | 85.83 |
| Fixed (extreme-retuned) | 77.18 | 80.26 | 80.26 | 82.96 |
| Cosine (regular-tuned) | 76.38 | 88.97 | 88.97 | 89.56 |
| Frozen regular policy | 75.51 ± 0.88 [74.00–76.33] | 75.11 ± 7.44 [66.22–81.46] | 77.88 ± 6.17 [67.38–82.17] | 86.63 ± 0.81 [85.87–87.58] |
| Cold extreme policy | 81.27 ± 2.17 [78.43–83.19] | 82.57 ± 0.79 [81.86–83.77] | 82.52 ± 1.26 [81.06–84.03] | 85.40 ± 0.47 [84.90–86.09] |
| Extreme warm-start policy | 83.25 ± 0.37 [82.82–83.73] | 85.84 ± 0.59 [85.41–86.81] | 85.86 ± 0.52 [85.20–86.48] | 85.62 ± 2.22 [82.95–87.89] |

Paired difference A − B (pp), mean [95% two-level bootstrap CI]; share of orderings with A > B; policy seeds with positive mean difference (of 5) and policy-seed SD.

| A − B | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| Extreme warm-start policy − Fixed (regular-tuned) | +14.36 [+11.95, +16.91]; 89%; 5/5, SD 0.37 | +3.37 [+1.65, +5.17]; 38%; 5/5, SD 0.59 | +3.38 [+1.69, +5.20]; 38%; 5/5, SD 0.52 | -0.21 [-2.03, +1.73]; 21%; 2/5, SD 2.22 |
| Extreme warm-start policy − Fixed (extreme-retuned) | +6.07 [+4.62, +7.59]; 88%; 5/5, SD 0.37 | +5.58 [+4.46, +6.76]; 93%; 5/5, SD 0.59 | +5.59 [+4.48, +6.79]; 93%; 5/5, SD 0.52 | +2.66 [+0.92, +4.48]; 99%; 4/5, SD 2.22 |
| Extreme warm-start policy − Cosine (regular-tuned) | +6.87 [+4.91, +8.89]; 65%; 5/5, SD 0.37 | -3.12 [-4.15, -2.05]; 14%; 0/5, SD 0.59 | -3.11 [-4.12, -2.07]; 15%; 0/5, SD 0.52 | -3.94 [-5.92, -2.06]; 11%; 0/5, SD 2.22 |
| Extreme warm-start policy − Frozen regular policy | +7.74 [+5.70, +9.90]; 71%; 5/5, SD 0.68 | +10.74 [+4.54, +17.64]; 72%; 5/5, SD 7.95 | +7.98 [+3.68, +13.77]; 70%; 5/5, SD 6.10 | -1.01 [-3.34, +1.35]; 19%; 2/5, SD 2.95 |
| Extreme warm-start policy − Cold extreme policy | +1.99 [+0.55, +3.75]; 70%; 5/5, SD 1.88 | +3.28 [+2.33, +4.22]; 97%; 5/5, SD 0.99 | +3.34 [+2.40, +4.35]; 97%; 5/5, SD 1.02 | +0.22 [-1.87, +2.23]; 76%; 3/5, SD 2.54 |
| Cold extreme policy − Fixed (regular-tuned) | +12.37 [+9.60, +15.24]; 86%; 5/5, SD 2.17 | +0.09 [-1.53, +1.74]; 23%; 3/5, SD 0.79 | +0.04 [-1.80, +1.91]; 22%; 2/5, SD 1.26 | -0.43 [-1.54, +0.78]; 22%; 1/5, SD 0.47 |
| Cold extreme policy − Fixed (extreme-retuned) | +4.09 [+1.94, +6.24]; 82%; 5/5, SD 2.17 | +2.30 [+1.15, +3.42]; 81%; 5/5, SD 0.79 | +2.25 [+0.83, +3.64]; 76%; 5/5, SD 1.26 | +2.45 [+1.56, +3.42]; 86%; 5/5, SD 0.47 |
| Cold extreme policy − Cosine (regular-tuned) | +4.88 [+2.08, +7.53]; 61%; 5/5, SD 2.17 | -6.40 [-7.68, -5.09]; 7%; 0/5, SD 0.79 | -6.45 [-7.97, -4.99]; 6%; 0/5, SD 1.26 | -4.16 [-5.09, -3.27]; 11%; 0/5, SD 0.47 |
| Cold extreme policy − Frozen regular policy | +5.75 [+3.43, +8.13]; 63%; 5/5, SD 1.84 | +7.46 [+1.58, +14.03]; 59%; 5/5, SD 7.58 | +4.64 [+0.47, +9.94]; 52%; 4/5, SD 5.72 | -1.23 [-2.20, -0.08]; 19%; 0/5, SD 0.76 |
| Frozen regular policy − Fixed (regular-tuned) | +6.62 [+5.37, +7.92]; 96%; 5/5, SD 0.88 | -7.37 [-13.85, -1.71]; 26%; 0/5, SD 7.44 | -4.60 [-10.44, -0.62]; 27%; 0/5, SD 6.17 | +0.80 [+0.11, +1.64]; 72%; 5/5, SD 0.81 |
| Frozen regular policy − Cosine (regular-tuned) | -0.87 [-3.49, +1.63]; 51%; 0/5, SD 0.88 | -13.86 [-20.49, -8.01]; 3%; 0/5, SD 7.44 | -11.09 [-17.08, -6.72]; 4%; 0/5, SD 6.17 | -2.93 [-4.35, -1.64]; 19%; 0/5, SD 0.81 |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | +8.29 [+6.53, +10.13]; 70% | -2.21 [-3.08, -1.31]; 16% | -2.21 [-3.08, -1.31]; 16% | -2.87 [-3.33, -2.38]; 8% |

### Excluding orderings within profile distance 0.15 of an original (N = 189)

Mean final test accuracy (%) ± policy-seed SD [min–max]; failures in parentheses (all orderings).

| Method | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| Fixed (regular-tuned) | 69.38 | 83.07 | 83.07 | 86.09 |
| Fixed (extreme-retuned) | 77.57 | 80.59 | 80.59 | 83.16 |
| Cosine (regular-tuned) | 75.84 | 88.80 | 88.80 | 89.42 |
| Frozen regular policy | 75.88 ± 0.86 [74.63–77.01] | 75.31 ± 7.57 [66.33–81.68] | 78.13 ± 6.25 [67.51–82.52] | 86.81 ± 0.73 [86.16–87.66] |
| Cold extreme policy | 81.64 ± 1.91 [79.09–83.27] | 82.79 ± 0.80 [82.00–83.97] | 82.70 ± 1.25 [81.32–84.19] | 85.45 ± 0.42 [84.95–86.03] |
| Extreme warm-start policy | 83.31 ± 0.33 [82.95–83.71] | 85.92 ± 0.60 [85.47–86.84] | 85.93 ± 0.49 [85.26–86.50] | 85.72 ± 2.13 [83.16–87.90] |

Paired difference A − B (pp), mean [95% two-level bootstrap CI]; share of orderings with A > B; policy seeds with positive mean difference (of 5) and policy-seed SD.

| A − B | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| Extreme warm-start policy − Fixed (regular-tuned) | +13.93 [+11.47, +16.49]; 88%; 5/5, SD 0.33 | +2.85 [+1.22, +4.67]; 37%; 5/5, SD 0.60 | +2.86 [+1.24, +4.70]; 37%; 5/5, SD 0.49 | -0.37 [-2.14, +1.52]; 19%; 2/5, SD 2.13 |
| Extreme warm-start policy − Fixed (extreme-retuned) | +5.74 [+4.29, +7.32]; 88%; 5/5, SD 0.33 | +5.32 [+4.24, +6.52]; 93%; 5/5, SD 0.60 | +5.33 [+4.25, +6.55]; 93%; 5/5, SD 0.49 | +2.56 [+0.87, +4.31]; 99%; 4/5, SD 2.13 |
| Extreme warm-start policy − Cosine (regular-tuned) | +7.47 [+5.44, +9.54]; 66%; 5/5, SD 0.33 | -2.88 [-3.91, -1.82]; 15%; 0/5, SD 0.60 | -2.87 [-3.90, -1.87]; 15%; 0/5, SD 0.49 | -3.70 [-5.69, -1.86]; 12%; 0/5, SD 2.13 |
| Extreme warm-start policy − Frozen regular policy | +7.43 [+5.36, +9.62]; 70%; 5/5, SD 0.66 | +10.61 [+4.31, +17.93]; 71%; 5/5, SD 8.11 | +7.80 [+3.41, +13.71]; 70%; 5/5, SD 6.24 | -1.09 [-3.30, +1.15]; 17%; 2/5, SD 2.80 |
| Extreme warm-start policy − Cold extreme policy | +1.67 [+0.41, +3.16]; 69%; 5/5, SD 1.64 | +3.13 [+2.22, +4.05]; 97%; 5/5, SD 0.96 | +3.22 [+2.26, +4.27]; 98%; 5/5, SD 1.08 | +0.27 [-1.72, +2.27]; 77%; 3/5, SD 2.42 |
| Cold extreme policy − Fixed (regular-tuned) | +12.26 [+9.56, +15.10]; 86%; 5/5, SD 1.91 | -0.28 [-1.90, +1.39]; 22%; 2/5, SD 0.80 | -0.36 [-2.18, +1.49]; 21%; 2/5, SD 1.25 | -0.64 [-1.76, +0.55]; 21%; 0/5, SD 0.42 |
| Cold extreme policy − Fixed (extreme-retuned) | +4.06 [+2.06, +6.06]; 82%; 5/5, SD 1.91 | +2.19 [+1.03, +3.31]; 80%; 5/5, SD 0.80 | +2.11 [+0.68, +3.47]; 76%; 5/5, SD 1.25 | +2.29 [+1.39, +3.24]; 86%; 5/5, SD 0.42 |
| Cold extreme policy − Cosine (regular-tuned) | +5.80 [+3.15, +8.29]; 63%; 5/5, SD 1.91 | -6.01 [-7.29, -4.72]; 7%; 0/5, SD 0.80 | -6.10 [-7.56, -4.62]; 6%; 0/5, SD 1.25 | -3.97 [-4.86, -3.09]; 11%; 0/5, SD 0.42 |
| Cold extreme policy − Frozen regular policy | +5.76 [+3.62, +8.03]; 62%; 5/5, SD 1.46 | +7.48 [+1.43, +14.41]; 59%; 5/5, SD 7.75 | +4.58 [+0.30, +9.97]; 52%; 3/5, SD 5.85 | -1.36 [-2.32, -0.26]; 17%; 0/5, SD 0.67 |
| Frozen regular policy − Fixed (regular-tuned) | +6.50 [+5.31, +7.82]; 96%; 5/5, SD 0.86 | -7.76 [-14.56, -2.00]; 24%; 0/5, SD 7.57 | -4.94 [-10.84, -0.94]; 25%; 0/5, SD 6.25 | +0.72 [+0.06, +1.50]; 71%; 5/5, SD 0.73 |
| Frozen regular policy − Cosine (regular-tuned) | +0.04 [-2.47, +2.49]; 52%; 3/5, SD 0.86 | -13.49 [-20.37, -7.47]; 3%; 0/5, SD 7.57 | -10.67 [-16.75, -6.34]; 4%; 0/5, SD 6.25 | -2.61 [-3.97, -1.40]; 20%; 0/5, SD 0.73 |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | +8.20 [+6.38, +10.07]; 69% | -2.47 [-3.32, -1.57]; 14% | -2.47 [-3.32, -1.57]; 14% | -2.93 [-3.38, -2.41]; 7% |

### Severity regression (all orderings)

OLS of the per-ordering difference on mean KL (pp per nat) and on biased batches (pp per 100 batches); 95% CI from the same ordering-level bootstrap.

| A − B | Family | slope vs mean KL | r | slope vs biased batches | r |
|---|---|---:|---:|---:|---:|
| Extreme warm-start policy − Fixed (regular-tuned) | SGD | +20.99 [+9.51, +31.81] | +0.34 | +9.11 [+1.90, +15.33] | +0.21 |
| Extreme warm-start policy − Fixed (regular-tuned) | Adam | +4.00 [-6.42, +13.87] | +0.09 | -0.91 [-6.64, +4.52] | -0.03 |
| Extreme warm-start policy − Fixed (regular-tuned) | AdamW | +4.30 [-6.54, +14.21] | +0.10 | -0.81 [-6.62, +4.72] | -0.03 |
| Extreme warm-start policy − Fixed (regular-tuned) | RMSprop | +0.27 [-3.44, +4.85] | +0.01 | -1.24 [-3.31, +0.65] | -0.09 |
| Extreme warm-start policy − Fixed (extreme-retuned) | SGD | +3.69 [-5.45, +11.85] | +0.10 | -1.50 [-6.53, +3.23] | -0.06 |
| Extreme warm-start policy − Fixed (extreme-retuned) | Adam | +7.54 [+0.19, +13.19] | +0.28 | +1.06 [-2.90, +4.93] | +0.06 |
| Extreme warm-start policy − Fixed (extreme-retuned) | AdamW | +7.84 [-0.01, +13.90] | +0.29 | +1.17 [-2.99, +5.29] | +0.06 |
| Extreme warm-start policy − Fixed (extreme-retuned) | RMSprop | +1.31 [-0.64, +3.65] | +0.13 | -0.32 [-1.83, +0.80] | -0.04 |
| Extreme warm-start policy − Cosine (regular-tuned) | SGD | +34.43 [+27.19, +42.93] | +0.68 | +22.69 [+17.83, +27.66] | +0.63 |
| Extreme warm-start policy − Cosine (regular-tuned) | Adam | +3.82 [-1.55, +8.80] | +0.16 | +3.19 [+0.73, +5.87] | +0.19 |
| Extreme warm-start policy − Cosine (regular-tuned) | AdamW | +4.12 [-1.65, +9.21] | +0.17 | +3.30 [+0.76, +6.03] | +0.19 |
| Extreme warm-start policy − Cosine (regular-tuned) | RMSprop | +0.98 [-3.35, +5.28] | +0.04 | +2.19 [-0.49, +5.27] | +0.13 |
| Extreme warm-start policy − Frozen regular policy | SGD | +19.07 [+7.47, +29.03] | +0.37 | +5.76 [-1.60, +11.98] | +0.16 |
| Extreme warm-start policy − Frozen regular policy | Adam | +31.64 [+16.29, +49.16] | +0.63 | +20.14 [+6.36, +34.43] | +0.56 |
| Extreme warm-start policy − Frozen regular policy | AdamW | +24.41 [+11.85, +37.65] | +0.57 | +13.93 [+3.66, +25.39] | +0.46 |
| Extreme warm-start policy − Frozen regular policy | RMSprop | +0.98 [-2.04, +4.68] | +0.09 | +0.50 [-1.23, +2.13] | +0.06 |
| Extreme warm-start policy − Cold extreme policy | SGD | +0.48 [-2.57, +2.91] | +0.03 | -0.91 [-3.73, +1.16] | -0.08 |
| Extreme warm-start policy − Cold extreme policy | Adam | +2.52 [-0.11, +5.92] | +0.18 | +0.23 [-1.30, +2.08] | +0.02 |
| Extreme warm-start policy − Cold extreme policy | AdamW | +3.03 [+0.37, +6.42] | +0.23 | +0.44 [-1.12, +2.38] | +0.05 |
| Extreme warm-start policy − Cold extreme policy | RMSprop | -2.05 [-6.12, +3.04] | -0.15 | -0.37 [-2.75, +2.46] | -0.04 |
| Cold extreme policy − Fixed (regular-tuned) | SGD | +20.52 [+9.78, +30.52] | +0.37 | +10.02 [+3.28, +15.78] | +0.25 |
| Cold extreme policy − Fixed (regular-tuned) | Adam | +1.49 [-9.30, +10.94] | +0.04 | -1.14 [-6.87, +4.20] | -0.04 |
| Cold extreme policy − Fixed (regular-tuned) | AdamW | +1.27 [-10.08, +11.06] | +0.03 | -1.24 [-7.18, +4.28] | -0.04 |
| Cold extreme policy − Fixed (regular-tuned) | RMSprop | +2.32 [-5.00, +8.63] | +0.08 | -0.87 [-4.83, +2.64] | -0.04 |
| Cold extreme policy − Fixed (extreme-retuned) | SGD | +3.22 [-5.77, +11.23] | +0.10 | -0.59 [-5.55, +4.07] | -0.03 |
| Cold extreme policy − Fixed (extreme-retuned) | Adam | +5.02 [-3.66, +11.35] | +0.21 | +0.83 [-3.41, +4.85] | +0.05 |
| Cold extreme policy − Fixed (extreme-retuned) | AdamW | +4.81 [-4.27, +11.59] | +0.20 | +0.73 [-3.74, +4.90] | +0.04 |
| Cold extreme policy − Fixed (extreme-retuned) | RMSprop | +3.37 [-3.10, +8.15] | +0.16 | +0.05 [-3.59, +3.16] | +0.00 |
| Cold extreme policy − Cosine (regular-tuned) | SGD | +33.95 [+26.79, +42.41] | +0.61 | +23.60 [+18.17, +29.50] | +0.60 |
| Cold extreme policy − Cosine (regular-tuned) | Adam | +1.30 [-4.93, +6.58] | +0.04 | +2.96 [+0.15, +5.94] | +0.14 |
| Cold extreme policy − Cosine (regular-tuned) | AdamW | +1.09 [-5.33, +6.58] | +0.04 | +2.86 [+0.06, +5.73] | +0.14 |
| Cold extreme policy − Cosine (regular-tuned) | RMSprop | +3.03 [-2.62, +7.74] | +0.15 | +2.57 [+0.06, +5.04] | +0.17 |
| Cold extreme policy − Frozen regular policy | SGD | +18.59 [+7.24, +28.17] | +0.41 | +6.67 [-0.82, +13.40] | +0.20 |
| Cold extreme policy − Frozen regular policy | Adam | +29.12 [+14.08, +45.32] | +0.66 | +19.91 [+6.28, +33.97] | +0.63 |
| Cold extreme policy − Frozen regular policy | AdamW | +21.37 [+9.08, +33.47] | +0.59 | +13.49 [+3.32, +24.82] | +0.52 |
| Cold extreme policy − Frozen regular policy | RMSprop | +3.03 [-4.13, +8.93] | +0.14 | +0.87 [-2.90, +4.28] | +0.06 |
| Frozen regular policy − Fixed (regular-tuned) | SGD | +1.92 [-2.83, +8.06] | +0.08 | +3.35 [-0.72, +7.80] | +0.19 |
| Frozen regular policy − Fixed (regular-tuned) | Adam | -27.64 [-45.53, -13.85] | -0.65 | -21.05 [-36.04, -7.96] | -0.69 |
| Frozen regular policy − Fixed (regular-tuned) | AdamW | -20.10 [-34.09, -9.63] | -0.62 | -14.73 [-26.99, -5.44] | -0.64 |
| Frozen regular policy − Fixed (regular-tuned) | RMSprop | -0.71 [-2.99, +2.18] | -0.07 | -1.74 [-3.07, -0.46] | -0.24 |
| Frozen regular policy − Cosine (regular-tuned) | SGD | +15.36 [+5.69, +27.76] | +0.24 | +16.94 [+10.36, +24.79] | +0.38 |
| Frozen regular policy − Cosine (regular-tuned) | Adam | -27.82 [-45.08, -13.16] | -0.49 | -16.95 [-31.64, -2.89] | -0.42 |
| Frozen regular policy − Cosine (regular-tuned) | AdamW | -20.28 [-34.15, -8.38] | -0.40 | -10.63 [-22.57, +0.04] | -0.29 |
| Frozen regular policy − Cosine (regular-tuned) | RMSprop | +0.00 [-6.23, +5.86] | +0.00 | +1.69 [-2.14, +5.86] | +0.08 |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | SGD | +17.30 [+10.71, +25.66] | +0.39 | +10.61 [+6.20, +15.02] | +0.34 |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | Adam | -3.54 [-8.93, +2.86] | -0.15 | -1.98 [-5.13, +0.79] | -0.12 |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | AdamW | -3.54 [-8.93, +2.86] | -0.15 | -1.98 [-5.12, +0.79] | -0.12 |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | RMSprop | -1.05 [-3.56, +2.01] | -0.09 | -0.92 [-2.12, +0.22] | -0.11 |

### Severity quintiles (all orderings)

Mean difference (pp) [95% CI] by quintile of mean KL.

| A − B | Family | Q1 (0.13–0.27) | Q2 (0.27–0.36) | Q3 (0.36–0.47) | Q4 (0.47–0.62) | Q5 (0.62–2.07) |
|---|---|---:|---:|---:|---:|---:|
| Extreme warm-start policy − Fixed (regular-tuned) | SGD | +8.95 [+4.59, +14.23] | +8.99 [+5.24, +13.38] | +12.40 [+8.01, +17.30] | +19.13 [+13.90, +24.71] | +22.83 [+16.21, +29.74] |
| Extreme warm-start policy − Fixed (regular-tuned) | Adam | +4.21 [+0.85, +8.30] | +3.32 [+0.42, +6.97] | +2.35 [+0.23, +4.90] | +3.31 [+0.12, +7.54] | +4.58 [-0.77, +10.61] |
| Extreme warm-start policy − Fixed (regular-tuned) | AdamW | +4.24 [+0.91, +8.32] | +3.34 [+0.43, +6.99] | +2.36 [+0.21, +4.92] | +3.27 [+0.10, +7.50] | +4.61 [-0.83, +10.75] |
| Extreme warm-start policy − Fixed (regular-tuned) | RMSprop | +0.17 [-1.75, +2.67] | -0.30 [-2.14, +1.99] | -0.59 [-2.49, +1.69] | +0.08 [-2.21, +2.91] | +0.05 [-2.65, +3.43] |
| Extreme warm-start policy − Fixed (extreme-retuned) | SGD | +6.07 [+3.67, +8.93] | +6.05 [+3.84, +8.76] | +5.58 [+3.45, +7.93] | +6.48 [+3.51, +9.84] | +6.66 [+1.78, +12.02] |
| Extreme warm-start policy − Fixed (extreme-retuned) | Adam | +4.85 [+3.07, +7.04] | +5.24 [+3.47, +7.44] | +5.05 [+3.33, +7.06] | +5.08 [+3.28, +7.13] | +8.04 [+4.56, +11.84] |
| Extreme warm-start policy − Fixed (extreme-retuned) | AdamW | +4.88 [+3.12, +7.15] | +5.26 [+3.49, +7.49] | +5.06 [+3.34, +7.10] | +5.04 [+3.23, +7.16] | +8.06 [+4.48, +12.03] |
| Extreme warm-start policy − Fixed (extreme-retuned) | RMSprop | +2.35 [+0.74, +4.39] | +2.63 [+0.88, +4.70] | +2.75 [+0.86, +5.01] | +2.66 [+0.91, +4.72] | +3.06 [+1.07, +5.51] |
| Extreme warm-start policy − Cosine (regular-tuned) | SGD | -2.79 [-5.34, -0.80] | -0.77 [-3.08, +1.37] | +3.54 [+0.31, +7.17] | +10.50 [+6.69, +14.68] | +22.61 [+17.98, +27.49] |
| Extreme warm-start policy − Cosine (regular-tuned) | Adam | -4.13 [-6.29, -2.45] | -4.19 [-6.13, -2.60] | -3.69 [-5.71, -1.84] | -3.07 [-4.49, -1.65] | -1.20 [-4.40, +2.09] |
| Extreme warm-start policy − Cosine (regular-tuned) | AdamW | -4.10 [-6.34, -2.42] | -4.17 [-6.14, -2.60] | -3.68 [-5.70, -1.86] | -3.11 [-4.53, -1.74] | -1.17 [-4.39, +2.16] |
| Extreme warm-start policy − Cosine (regular-tuned) | RMSprop | -4.76 [-7.74, -2.50] | -4.58 [-7.39, -2.37] | -4.33 [-7.55, -1.73] | -3.69 [-6.14, -1.57] | -3.06 [-6.38, +0.15] |
| Extreme warm-start policy − Frozen regular policy | SGD | +3.76 [+0.60, +7.98] | +4.33 [+1.59, +7.94] | +5.56 [+2.61, +9.10] | +10.68 [+6.11, +15.74] | +14.72 [+8.15, +21.64] |
| Extreme warm-start policy − Frozen regular policy | Adam | +3.08 [+0.36, +6.54] | +4.83 [+1.34, +9.15] | +8.47 [+2.27, +16.32] | +11.63 [+3.55, +21.07] | +26.09 [+11.91, +40.95] |
| Extreme warm-start policy − Frozen regular policy | AdamW | +2.87 [+0.34, +6.23] | +3.80 [+0.83, +7.61] | +6.13 [+1.63, +12.12] | +8.02 [+2.40, +15.41] | +19.56 [+8.80, +31.96] |
| Extreme warm-start policy − Frozen regular policy | RMSprop | -0.90 [-3.10, +1.71] | -1.07 [-3.42, +1.49] | -1.19 [-3.52, +1.34] | -1.00 [-3.20, +1.46] | -0.55 [-3.55, +2.99] |
| Extreme warm-start policy − Cold extreme policy | SGD | +2.64 [+0.05, +6.30] | +1.68 [+0.01, +4.18] | +1.77 [+0.22, +3.60] | +2.17 [+0.69, +4.23] | +2.37 [+0.79, +4.17] |
| Extreme warm-start policy − Cold extreme policy | Adam | +2.64 [+1.43, +4.26] | +2.75 [+1.62, +4.26] | +3.52 [+2.18, +5.20] | +2.82 [+1.75, +4.10] | +4.93 [+2.82, +7.23] |
| Extreme warm-start policy − Cold extreme policy | AdamW | +2.53 [+1.42, +3.80] | +2.71 [+1.60, +4.02] | +3.69 [+2.17, +5.51] | +2.80 [+1.80, +4.16] | +5.15 [+2.91, +7.78] |
| Extreme warm-start policy − Cold extreme policy | RMSprop | +0.10 [-2.02, +1.93] | +0.38 [-1.80, +2.29] | +0.03 [-2.59, +2.27] | +0.30 [-1.99, +2.35] | +0.08 [-3.25, +3.08] |
| Cold extreme policy − Fixed (regular-tuned) | SGD | +6.31 [+2.98, +10.69] | +7.31 [+4.00, +11.43] | +10.62 [+5.98, +15.69] | +16.96 [+11.69, +22.51] | +20.46 [+13.95, +27.27] |
| Cold extreme policy − Fixed (regular-tuned) | Adam | +1.56 [-0.94, +4.55] | +0.58 [-1.65, +3.34] | -1.17 [-2.97, +0.61] | +0.50 [-2.41, +4.07] | -0.35 [-5.97, +5.66] |
| Cold extreme policy − Fixed (regular-tuned) | AdamW | +1.71 [-1.03, +4.99] | +0.63 [-1.74, +3.60] | -1.33 [-3.46, +0.89] | +0.47 [-2.61, +4.16] | -0.54 [-6.35, +5.83] |
| Cold extreme policy − Fixed (regular-tuned) | RMSprop | +0.08 [-1.84, +2.54] | -0.68 [-2.12, +1.13] | -0.62 [-2.13, +1.45] | -0.22 [-2.36, +2.46] | -0.04 [-3.99, +4.18] |
| Cold extreme policy − Fixed (extreme-retuned) | SGD | +3.43 [+0.76, +6.24] | +4.37 [+2.04, +7.05] | +3.81 [+1.57, +6.36] | +4.31 [+1.24, +7.74] | +4.29 [-0.94, +9.86] |
| Cold extreme policy − Fixed (extreme-retuned) | Adam | +2.20 [+1.11, +3.43] | +2.49 [+1.29, +3.86] | +1.53 [+0.02, +2.81] | +2.26 [+0.58, +4.05] | +3.10 [-1.02, +7.22] |
| Cold extreme policy − Fixed (extreme-retuned) | AdamW | +2.35 [+1.04, +3.85] | +2.55 [+1.19, +4.09] | +1.37 [-0.50, +3.11] | +2.24 [+0.34, +4.23] | +2.91 [-1.52, +7.43] |
| Cold extreme policy − Fixed (extreme-retuned) | RMSprop | +2.25 [+0.99, +3.91] | +2.26 [+1.06, +3.77] | +2.72 [+1.25, +4.68] | +2.36 [+1.01, +4.03] | +2.98 [-0.34, +6.24] |
| Cold extreme policy − Cosine (regular-tuned) | SGD | -5.42 [-11.10, -1.22] | -2.46 [-6.74, +0.80] | +1.77 [-2.43, +6.02] | +8.33 [+3.74, +12.94] | +20.24 [+15.66, +24.88] |
| Cold extreme policy − Cosine (regular-tuned) | Adam | -6.78 [-10.15, -4.07] | -6.93 [-9.97, -4.49] | -7.21 [-10.30, -4.55] | -5.88 [-7.84, -4.17] | -6.13 [-9.42, -2.80] |
| Cold extreme policy − Cosine (regular-tuned) | AdamW | -6.63 [-9.74, -4.12] | -6.88 [-9.77, -4.55] | -7.37 [-10.57, -4.70] | -5.90 [-8.08, -4.14] | -6.32 [-9.93, -2.66] |
| Cold extreme policy − Cosine (regular-tuned) | RMSprop | -4.85 [-6.55, -3.58] | -4.96 [-6.54, -3.69] | -4.36 [-6.19, -2.72] | -3.98 [-5.22, -2.79] | -3.15 [-6.13, -0.07] |
| Cold extreme policy − Frozen regular policy | SGD | +1.12 [-2.84, +5.09] | +2.65 [-0.49, +6.44] | +3.79 [+0.72, +7.45] | +8.51 [+4.25, +13.19] | +12.34 [+6.19, +18.93] |
| Cold extreme policy − Frozen regular policy | Adam | +0.43 [-1.52, +3.06] | +2.09 [-0.77, +5.73] | +4.95 [-1.04, +12.69] | +8.82 [+1.41, +17.49] | +21.16 [+7.54, +35.70] |
| Cold extreme policy − Frozen regular policy | AdamW | +0.34 [-1.60, +3.06] | +1.09 [-1.39, +4.43] | +2.44 [-1.74, +7.79] | +5.23 [+0.04, +11.85] | +14.41 [+3.70, +26.09] |
| Cold extreme policy − Frozen regular policy | RMSprop | -1.00 [-2.40, +1.07] | -1.45 [-2.49, +0.05] | -1.22 [-2.45, +0.55] | -1.30 [-2.64, +0.49] | -0.64 [-4.12, +3.17] |
| Frozen regular policy − Fixed (regular-tuned) | SGD | +5.19 [+2.82, +8.47] | +4.65 [+2.90, +7.00] | +6.84 [+4.42, +9.64] | +8.44 [+5.50, +11.94] | +8.11 [+5.20, +11.40] |
| Frozen regular policy − Fixed (regular-tuned) | Adam | +1.13 [-0.26, +2.97] | -1.51 [-4.98, +1.10] | -6.12 [-13.72, -0.61] | -8.32 [-17.14, -0.99] | -21.51 [-35.78, -8.31] |
| Frozen regular policy − Fixed (regular-tuned) | AdamW | +1.37 [-0.21, +3.46] | -0.46 [-3.65, +2.15] | -3.77 [-9.80, +0.03] | -4.75 [-11.92, +0.42] | -14.95 [-27.30, -5.16] |
| Frozen regular policy − Fixed (regular-tuned) | RMSprop | +1.07 [+0.27, +2.28] | +0.77 [+0.03, +1.85] | +0.60 [-0.02, +1.50] | +1.08 [-0.03, +2.79] | +0.60 [-0.92, +2.39] |
| Frozen regular policy − Cosine (regular-tuned) | SGD | -6.54 [-12.65, -1.85] | -5.11 [-10.43, -0.85] | -2.02 [-7.52, +2.96] | -0.18 [-6.70, +5.85] | +7.89 [+1.39, +13.85] |
| Frozen regular policy − Cosine (regular-tuned) | Adam | -7.21 [-12.79, -2.99] | -9.02 [-14.54, -4.56] | -12.16 [-20.19, -5.74] | -14.70 [-24.07, -6.69] | -27.29 [-41.48, -13.70] |
| Frozen regular policy − Cosine (regular-tuned) | AdamW | -6.97 [-12.44, -2.92] | -7.97 [-13.35, -3.81] | -9.81 [-16.71, -4.48] | -11.13 [-18.99, -4.97] | -20.73 [-33.21, -10.15] |
| Frozen regular policy − Cosine (regular-tuned) | RMSprop | -3.86 [-7.38, -1.23] | -3.51 [-6.35, -1.31] | -3.14 [-6.13, -0.61] | -2.69 [-4.94, -0.75] | -2.51 [-6.14, +0.87] |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | SGD | +2.88 [+0.45, +5.76] | +2.93 [+0.85, +5.47] | +6.82 [+3.08, +10.96] | +12.65 [+8.73, +16.89] | +16.17 [+11.62, +21.09] |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | Adam | -0.64 [-2.32, +1.48] | -1.92 [-3.30, -0.16] | -2.70 [-3.62, -1.66] | -1.77 [-3.55, +0.83] | -3.46 [-6.24, -0.38] |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | AdamW | -0.64 [-2.32, +1.49] | -1.92 [-3.30, -0.17] | -2.70 [-3.62, -1.66] | -1.77 [-3.55, +0.83] | -3.46 [-6.24, -0.38] |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | RMSprop | -2.18 [-2.95, -1.16] | -2.93 [-3.34, -2.46] | -3.33 [-3.80, -2.85] | -2.58 [-3.73, -0.96] | -3.02 [-4.60, -1.23] |

### Replication: seven original orderings vs generated suite

Same code and environment; seven originals use seeds 1920000–1920019 (bootstrap over the 7 orderings, then episode seeds, then policy seeds). "Replicates" = both 95% CIs exclude 0 with the same sign.

| A − B | Family | Seven originals | orderings A > B | Generated (N = 200) | orderings A > B | Sign |
|---|---|---:|---:|---:|---:|---|
| Extreme warm-start policy − Fixed (regular-tuned) | SGD | +22.36 [+11.24, +34.85] | 6/7 | +14.46 [+12.03, +17.04] | 88% | replicates |
| Extreme warm-start policy − Fixed (regular-tuned) | Adam | +11.12 [+3.27, +18.74] | 6/7 | +3.55 [+1.80, +5.43] | 38% | replicates |
| Extreme warm-start policy − Fixed (regular-tuned) | AdamW | +11.56 [+3.52, +19.70] | 6/7 | +3.56 [+1.82, +5.47] | 38% | replicates |
| Extreme warm-start policy − Fixed (regular-tuned) | RMSprop | +4.63 [+1.36, +7.91] | 6/7 | -0.12 [-1.97, +1.80] | 21% | does not replicate (generated CI includes 0) |
| Extreme warm-start policy − Fixed (extreme-retuned) | SGD | +8.50 [-2.21, +18.52] | 5/7 | +6.17 [+4.69, +7.72] | 88% | same sign, now CI excludes 0 |
| Extreme warm-start policy − Fixed (extreme-retuned) | Adam | +8.56 [+1.90, +15.91] | 5/7 | +5.65 [+4.52, +6.84] | 94% | replicates |
| Extreme warm-start policy − Fixed (extreme-retuned) | AdamW | +8.99 [+2.25, +16.76] | 5/7 | +5.66 [+4.56, +6.88] | 94% | replicates |
| Extreme warm-start policy − Fixed (extreme-retuned) | RMSprop | +1.42 [-0.04, +3.36] | 6/7 | +2.69 [+0.94, +4.50] | 99% | same sign, now CI excludes 0 |
| Extreme warm-start policy − Cosine (regular-tuned) | SGD | +22.16 [+9.73, +35.08] | 6/7 | +6.63 [+4.64, +8.71] | 64% | replicates |
| Extreme warm-start policy − Cosine (regular-tuned) | Adam | +8.65 [-0.63, +18.89] | 4/7 | -3.25 [-4.28, -2.20] | 14% | **opposite sign, generated CI excludes 0** |
| Extreme warm-start policy − Cosine (regular-tuned) | AdamW | +9.09 [-0.45, +19.32] | 4/7 | -3.25 [-4.29, -2.22] | 14% | **opposite sign, generated CI excludes 0** |
| Extreme warm-start policy − Cosine (regular-tuned) | RMSprop | +3.24 [-5.41, +11.78] | 4/7 | -4.08 [-6.13, -2.18] | 11% | **opposite sign, generated CI excludes 0** |
| Extreme warm-start policy − Frozen regular policy | SGD | +18.18 [+9.02, +27.44] | 6/7 | +7.81 [+5.77, +10.03] | 71% | replicates |
| Extreme warm-start policy − Frozen regular policy | Adam | +28.42 [+16.17, +41.72] | 7/7 | +10.83 [+4.58, +17.81] | 72% | replicates |
| Extreme warm-start policy − Frozen regular policy | AdamW | +25.83 [+14.08, +37.97] | 7/7 | +8.08 [+3.77, +13.69] | 70% | replicates |
| Extreme warm-start policy − Frozen regular policy | RMSprop | +4.09 [+0.69, +7.48] | 5/7 | -0.94 [-3.28, +1.41] | 19% | does not replicate (generated CI includes 0) |
| Extreme warm-start policy − Cold extreme policy | SGD | +2.33 [+0.20, +4.93] | 6/7 | +2.12 [+0.63, +3.95] | 70% | replicates |
| Extreme warm-start policy − Cold extreme policy | Adam | +5.25 [+1.12, +10.54] | 6/7 | +3.33 [+2.37, +4.31] | 97% | replicates |
| Extreme warm-start policy − Cold extreme policy | AdamW | +5.78 [+1.44, +11.01] | 6/7 | +3.38 [+2.46, +4.36] | 98% | replicates |
| Extreme warm-start policy − Cold extreme policy | RMSprop | -1.33 [-5.44, +1.87] | 3/7 | +0.18 [-1.88, +2.22] | 76% | neither CI excludes 0 |
| Cold extreme policy − Fixed (regular-tuned) | SGD | +20.03 [+9.66, +32.20] | 7/7 | +12.34 [+9.58, +15.22] | 86% | replicates |
| Cold extreme policy − Fixed (regular-tuned) | Adam | +5.87 [-2.46, +13.94] | 5/7 | +0.22 [-1.43, +1.91] | 23% | neither CI excludes 0 |
| Cold extreme policy − Fixed (regular-tuned) | AdamW | +5.78 [-2.82, +14.06] | 5/7 | +0.18 [-1.70, +2.06] | 22% | neither CI excludes 0 |
| Cold extreme policy − Fixed (regular-tuned) | RMSprop | +5.96 [+0.80, +11.57] | 4/7 | -0.30 [-1.45, +0.94] | 22% | does not replicate (generated CI includes 0) |
| Cold extreme policy − Fixed (extreme-retuned) | SGD | +6.17 [-4.48, +15.79] | 5/7 | +4.04 [+1.86, +6.19] | 82% | same sign, now CI excludes 0 |
| Cold extreme policy − Fixed (extreme-retuned) | Adam | +3.31 [-5.08, +10.77] | 5/7 | +2.32 [+1.18, +3.44] | 81% | same sign, now CI excludes 0 |
| Cold extreme policy − Fixed (extreme-retuned) | AdamW | +3.22 [-5.08, +10.91] | 5/7 | +2.28 [+0.87, +3.66] | 76% | same sign, now CI excludes 0 |
| Cold extreme policy − Fixed (extreme-retuned) | RMSprop | +2.75 [-1.21, +7.31] | 4/7 | +2.51 [+1.62, +3.51] | 86% | same sign, now CI excludes 0 |
| Cold extreme policy − Cosine (regular-tuned) | SGD | +19.83 [+8.34, +32.12] | 6/7 | +4.51 [+1.58, +7.19] | 60% | replicates |
| Cold extreme policy − Cosine (regular-tuned) | Adam | +3.40 [-6.17, +12.61] | 3/7 | -6.59 [-7.94, -5.29] | 7% | **opposite sign, generated CI excludes 0** |
| Cold extreme policy − Cosine (regular-tuned) | AdamW | +3.31 [-5.43, +11.97] | 4/7 | -6.62 [-8.19, -5.15] | 6% | **opposite sign, generated CI excludes 0** |
| Cold extreme policy − Cosine (regular-tuned) | RMSprop | +4.57 [-2.30, +12.10] | 3/7 | -4.26 [-5.19, -3.36] | 10% | **opposite sign, generated CI excludes 0** |
| Cold extreme policy − Frozen regular policy | SGD | +15.85 [+7.44, +24.54] | 7/7 | +5.69 [+3.29, +8.13] | 62% | replicates |
| Cold extreme policy − Frozen regular policy | Adam | +23.17 [+12.45, +36.66] | 7/7 | +7.50 [+1.66, +14.11] | 58% | replicates |
| Cold extreme policy − Frozen regular policy | AdamW | +20.05 [+9.67, +31.55] | 7/7 | +4.70 [+0.51, +9.79] | 52% | replicates |
| Cold extreme policy − Frozen regular policy | RMSprop | +5.42 [+0.21, +10.84] | 4/7 | -1.12 [-2.13, +0.05] | 19% | does not replicate (generated CI includes 0) |
| Frozen regular policy − Fixed (regular-tuned) | SGD | +4.18 [+0.85, +9.07] | 6/7 | +6.64 [+5.39, +7.97] | 96% | replicates |
| Frozen regular policy − Fixed (regular-tuned) | Adam | -17.30 [-34.03, -3.71] | 0/7 | -7.28 [-13.86, -1.67] | 26% | replicates |
| Frozen regular policy − Fixed (regular-tuned) | AdamW | -14.27 [-28.27, -2.10] | 0/7 | -4.52 [-10.24, -0.60] | 28% | replicates |
| Frozen regular policy − Fixed (regular-tuned) | RMSprop | +0.55 [-0.96, +1.61] | 5/7 | +0.82 [+0.12, +1.66] | 72% | same sign, now CI excludes 0 |
| Frozen regular policy − Cosine (regular-tuned) | SGD | +3.98 [-1.89, +11.18] | 3/7 | -1.18 [-3.82, +1.43] | 51% | neither CI excludes 0 |
| Frozen regular policy − Cosine (regular-tuned) | Adam | -19.77 [-37.17, -4.04] | 2/7 | -14.08 [-20.78, -8.17] | 2% | replicates |
| Frozen regular policy − Cosine (regular-tuned) | AdamW | -16.74 [-33.87, -1.93] | 2/7 | -11.33 [-17.23, -6.91] | 4% | replicates |
| Frozen regular policy − Cosine (regular-tuned) | RMSprop | -0.85 [-10.30, +8.44] | 3/7 | -3.14 [-4.58, -1.83] | 18% | same sign, now CI excludes 0 |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | SGD | +13.86 [+2.66, +27.42] | 5/7 | +8.29 [+6.56, +10.16] | 70% | replicates |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | Adam | +2.56 [-0.26, +5.24] | 5/7 | -2.10 [-3.00, -1.14] | 16% | **opposite sign, generated CI excludes 0** |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | AdamW | +2.56 [-0.26, +5.24] | 5/7 | -2.10 [-3.00, -1.14] | 16% | **opposite sign, generated CI excludes 0** |
| Fixed (extreme-retuned) − Fixed (regular-tuned) | RMSprop | +3.21 [-0.03, +6.74] | 4/7 | -2.81 [-3.28, -2.29] | 8% | **opposite sign, generated CI excludes 0** |
