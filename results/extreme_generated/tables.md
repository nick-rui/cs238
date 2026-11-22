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
| Cosine (extreme-retuned) | 83.53 | 86.89 | 86.93 | 87.45 |
| Warmup-cosine (extreme-retuned) | 83.00 | 86.67 | 86.67 | 88.15 |
| Random schedule (cold) | 75.32 ± 0.41 [74.93–75.98] (14 failed) | 75.11 ± 0.51 [74.60–75.98] (68 failed) | 77.37 ± 0.46 [76.76–77.98] (36 failed) | 70.17 ± 0.40 [69.73–70.77] (377 failed) |
| Random schedule (warm) | 75.32 ± 0.41 [74.93–75.98] (14 failed) | 75.43 ± 0.22 [75.24–75.76] (16 failed) | 76.35 ± 0.30 [76.04–76.80] (17 failed) | 76.01 ± 0.29 [75.71–76.48] (30 failed) |
| Untrained policy (cold) | 77.59 ± 0.00 [77.59–77.59] | 77.41 ± 0.00 [77.41–77.41] | 77.55 ± 0.00 [77.55–77.55] | 83.83 ± 0.00 [83.83–83.83] |
| Untrained policy (warm) | 77.00 ± 0.00 [77.00–77.00] | 80.11 ± 0.00 [80.11–80.11] | 80.10 ± 0.00 [80.10–80.10] | 82.81 ± 0.00 [82.81–82.81] |
| Random fixed config | 53.28 ± 16.63 [25.41–68.40] (13 failed) | 70.69 ± 5.26 [64.37–77.33] | 76.08 ± 4.03 [73.03–82.39] | 57.23 ± 29.17 [7.23–77.47] (958 failed) |

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
| Extreme warm-start policy − Cosine (extreme-retuned) | -0.37 [-1.28, +0.49]; 56%; 1/5, SD 0.39 | -1.14 [-2.09, -0.19]; 57%; 0/5, SD 0.59 | -1.17 [-2.13, -0.28]; 56%; 0/5, SD 0.54 | -1.93 [-4.10, +0.04]; 60%; 2/5, SD 2.25 |
| Cold extreme policy − Cosine (extreme-retuned) | -2.49 [-4.87, -0.51]; 46%; 0/5, SD 2.24 | -4.47 [-5.85, -3.19]; 15%; 0/5, SD 0.78 | -4.54 [-6.11, -3.11]; 10%; 0/5, SD 1.25 | -2.11 [-3.08, -1.25]; 24%; 0/5, SD 0.50 |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | +0.16 [-0.76, +1.07]; 61%; 3/5, SD 0.39 | -0.91 [-1.84, +0.03]; 56%; 1/5, SD 0.59 | -0.90 [-1.85, -0.01]; 56%; 0/5, SD 0.54 | -2.63 [-4.71, -0.72]; 34%; 0/5, SD 2.25 |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | -1.96 [-4.34, +0.03]; 48%; 1/5, SD 2.24 | -4.24 [-5.59, -2.99]; 12%; 0/5, SD 0.78 | -4.28 [-5.83, -2.85]; 9%; 0/5, SD 1.25 | -2.81 [-3.72, -1.99]; 12%; 0/5, SD 0.50 |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | +6.54 [+4.68, +8.52]; 82% | +6.79 [+5.24, +8.49]; 90% | +6.83 [+5.27, +8.52]; 90% | +4.62 [+3.26, +6.08]; 78% |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | +7.00 [+5.35, +8.78]; 64% | -2.12 [-2.58, -1.58]; 14% | -2.08 [-2.54, -1.55]; 12% | -2.15 [-2.77, -1.50]; 15% |
| Cosine (extreme-retuned) − Random schedule (cold) | +8.22 [+6.54, +10.06]; 92%; 5/5, SD 0.41 | +11.78 [+10.11, +13.54]; 98%; 5/5, SD 0.51 | +9.56 [+7.98, +11.25]; 97%; 5/5, SD 0.46 | +17.28 [+15.43, +19.24]; 100%; 5/5, SD 0.40 |
| Cosine (extreme-retuned) − Random schedule (warm) | +8.22 [+6.54, +10.06]; 92%; 5/5, SD 0.41 | +11.47 [+9.88, +13.16]; 98%; 5/5, SD 0.22 | +10.58 [+9.02, +12.25]; 98%; 5/5, SD 0.30 | +11.44 [+9.54, +13.43]; 96%; 5/5, SD 0.29 |
| Cosine (extreme-retuned) − Untrained policy (cold) | +5.94 [+3.79, +8.25]; 61%; 5/5, SD 0.00 | +9.49 [+7.29, +11.94]; 70%; 5/5, SD 0.00 | +9.39 [+7.21, +11.82]; 68%; 5/5, SD 0.00 | +3.62 [+1.96, +5.42]; 34%; 5/5, SD 0.00 |
| Cosine (extreme-retuned) − Untrained policy (warm) | +6.53 [+4.67, +8.52]; 82%; 5/5, SD 0.00 | +6.79 [+5.24, +8.48]; 90%; 5/5, SD 0.00 | +6.83 [+5.27, +8.52]; 90%; 5/5, SD 0.00 | +4.64 [+3.28, +6.09]; 78%; 5/5, SD 0.00 |
| Cosine (extreme-retuned) − Random fixed config | +30.25 [+19.27, +44.84]; 100%; 5/5, SD 16.63 | +16.21 [+11.80, +20.57]; 100%; 5/5, SD 5.26 | +10.85 [+7.12, +14.09]; 100%; 5/5, SD 4.03 | +30.22 [+12.20, +56.21]; 100%; 5/5, SD 29.17 |
| Cosine (regular-tuned) − Random schedule (cold) | +1.21 [-1.14, +3.52]; 58%; 5/5, SD 0.41 | +13.90 [+12.23, +15.63]; 98%; 5/5, SD 0.51 | +11.64 [+10.07, +13.26]; 96%; 5/5, SD 0.46 | +19.43 [+17.67, +21.26]; 98%; 5/5, SD 0.40 |
| Extreme warm-start policy − Random schedule (warm) | +7.84 [+6.45, +9.37]; 94%; 5/5, SD 0.59 | +10.33 [+9.06, +11.66]; 99%; 5/5, SD 0.59 | +9.42 [+8.23, +10.69]; 100%; 5/5, SD 0.35 | +9.50 [+7.56, +11.55]; 100%; 5/5, SD 2.12 |
| Cold extreme policy − Random schedule (cold) | +5.72 [+3.40, +7.95]; 90%; 5/5, SD 2.47 | +7.31 [+5.85, +8.80]; 95%; 5/5, SD 1.10 | +5.02 [+3.39, +6.69]; 90%; 5/5, SD 1.53 | +15.17 [+13.53, +16.92]; 100%; 5/5, SD 0.81 |

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
| Cosine (extreme-retuned) | 83.48 | 86.86 | 86.90 | 87.42 |
| Warmup-cosine (extreme-retuned) | 82.94 | 86.63 | 86.63 | 88.12 |
| Random schedule (cold) | 75.47 ± 0.40 [75.14–76.13] | 75.26 ± 0.52 [74.72–76.12] | 77.51 ± 0.44 [76.89–78.07] | 70.28 ± 0.46 [69.77–70.91] |
| Random schedule (warm) | 75.47 ± 0.40 [75.14–76.13] | 75.56 ± 0.18 [75.41–75.81] | 76.48 ± 0.27 [76.17–76.86] | 76.16 ± 0.26 [75.87–76.57] |
| Untrained policy (cold) | 77.83 ± 0.00 [77.83–77.83] | 77.69 ± 0.00 [77.69–77.69] | 77.83 ± 0.00 [77.83–77.83] | 84.02 ± 0.00 [84.02–84.02] |
| Untrained policy (warm) | 77.19 ± 0.00 [77.19–77.19] | 80.27 ± 0.00 [80.27–80.27] | 80.26 ± 0.00 [80.26–80.26] | 82.94 ± 0.00 [82.94–82.94] |
| Random fixed config | 53.41 ± 16.73 [25.40–68.64] | 70.73 ± 5.30 [64.32–77.56] | 76.18 ± 4.13 [73.03–82.62] | 57.38 ± 29.23 [7.29–77.66] |

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
| Extreme warm-start policy − Cosine (extreme-retuned) | -0.22 [-1.05, +0.63]; 56%; 1/5, SD 0.37 | -1.02 [-1.93, -0.07]; 58%; 0/5, SD 0.59 | -1.04 [-1.97, -0.15]; 56%; 0/5, SD 0.52 | -1.81 [-3.91, +0.14]; 60%; 2/5, SD 2.22 |
| Cold extreme policy − Cosine (extreme-retuned) | -2.21 [-4.50, -0.32]; 46%; 0/5, SD 2.17 | -4.30 [-5.59, -3.06]; 15%; 0/5, SD 0.79 | -4.38 [-5.88, -2.97]; 11%; 0/5, SD 1.26 | -2.02 [-2.97, -1.16]; 24%; 0/5, SD 0.47 |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | +0.31 [-0.55, +1.21]; 61%; 3/5, SD 0.37 | -0.78 [-1.68, +0.15]; 56%; 1/5, SD 0.59 | -0.78 [-1.68, +0.11]; 56%; 0/5, SD 0.52 | -2.50 [-4.52, -0.65]; 35%; 0/5, SD 2.22 |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | -1.68 [-3.98, +0.23]; 48%; 1/5, SD 2.17 | -4.06 [-5.33, -2.85]; 12%; 0/5, SD 0.79 | -4.11 [-5.61, -2.70]; 9%; 0/5, SD 1.26 | -2.72 [-3.61, -1.90]; 12%; 0/5, SD 0.47 |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | +6.30 [+4.47, +8.17]; 82% | +6.60 [+5.04, +8.21]; 89% | +6.63 [+5.07, +8.25]; 89% | +4.47 [+3.12, +5.90]; 77% |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | +7.09 [+5.41, +8.86]; 64% | -2.11 [-2.57, -1.57]; 14% | -2.07 [-2.54, -1.54]; 13% | -2.14 [-2.76, -1.46]; 15% |
| Cosine (extreme-retuned) − Random schedule (cold) | +8.01 [+6.35, +9.78]; 92%; 5/5, SD 0.40 | +11.60 [+9.95, +13.35]; 98%; 5/5, SD 0.52 | +9.38 [+7.86, +11.01]; 97%; 5/5, SD 0.44 | +17.14 [+15.26, +19.06]; 99%; 5/5, SD 0.46 |
| Cosine (extreme-retuned) − Random schedule (warm) | +8.01 [+6.35, +9.78]; 92%; 5/5, SD 0.40 | +11.30 [+9.75, +12.92]; 97%; 5/5, SD 0.18 | +10.42 [+8.89, +12.01]; 97%; 5/5, SD 0.27 | +11.26 [+9.41, +13.22]; 96%; 5/5, SD 0.26 |
| Cosine (extreme-retuned) − Untrained policy (cold) | +5.65 [+3.56, +7.81]; 61%; 5/5, SD 0.00 | +9.17 [+7.06, +11.47]; 70%; 5/5, SD 0.00 | +9.07 [+6.97, +11.34]; 68%; 5/5, SD 0.00 | +3.40 [+1.74, +5.12]; 33%; 5/5, SD 0.00 |
| Cosine (extreme-retuned) − Untrained policy (warm) | +6.29 [+4.46, +8.16]; 82%; 5/5, SD 0.00 | +6.59 [+5.04, +8.21]; 89%; 5/5, SD 0.00 | +6.63 [+5.07, +8.25]; 89%; 5/5, SD 0.00 | +4.48 [+3.14, +5.91]; 77%; 5/5, SD 0.00 |
| Cosine (extreme-retuned) − Random fixed config | +30.07 [+19.02, +44.88]; 100%; 5/5, SD 16.73 | +16.13 [+11.73, +20.51]; 100%; 5/5, SD 5.30 | +10.71 [+6.78, +14.06]; 99%; 5/5, SD 4.13 | +30.05 [+11.94, +56.16]; 100%; 5/5, SD 29.23 |
| Cosine (regular-tuned) − Random schedule (cold) | +0.92 [-1.36, +3.21]; 58%; 5/5, SD 0.40 | +13.70 [+12.07, +15.45]; 97%; 5/5, SD 0.52 | +11.45 [+9.94, +13.06]; 96%; 5/5, SD 0.44 | +19.28 [+17.49, +21.08]; 98%; 5/5, SD 0.46 |
| Extreme warm-start policy − Random schedule (warm) | +7.78 [+6.40, +9.31]; 94%; 5/5, SD 0.59 | +10.29 [+9.02, +11.61]; 99%; 5/5, SD 0.59 | +9.38 [+8.18, +10.63]; 99%; 5/5, SD 0.35 | +9.45 [+7.54, +11.53]; 99%; 5/5, SD 2.09 |
| Cold extreme policy − Random schedule (cold) | +5.80 [+3.53, +8.07]; 90%; 5/5, SD 2.42 | +7.30 [+5.84, +8.75]; 95%; 5/5, SD 1.11 | +5.00 [+3.37, +6.69]; 90%; 5/5, SD 1.54 | +15.12 [+13.46, +16.88]; 99%; 5/5, SD 0.83 |

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
| Cosine (extreme-retuned) | 83.26 | 86.72 | 86.76 | 87.33 |
| Warmup-cosine (extreme-retuned) | 82.72 | 86.47 | 86.48 | 88.00 |
| Random schedule (cold) | 75.77 ± 0.39 [75.35–76.41] | 75.57 ± 0.46 [74.94–76.22] | 77.78 ± 0.42 [77.14–78.21] | 70.53 ± 0.51 [69.94–71.15] |
| Random schedule (warm) | 75.77 ± 0.39 [75.35–76.41] | 75.92 ± 0.14 [75.77–76.12] | 76.83 ± 0.19 [76.55–77.01] | 76.49 ± 0.28 [76.13–76.90] |
| Untrained policy (cold) | 78.39 ± 0.00 [78.39–78.39] | 78.25 ± 0.00 [78.25–78.25] | 78.37 ± 0.00 [78.37–78.37] | 84.34 ± 0.00 [84.34–84.34] |
| Untrained policy (warm) | 77.58 ± 0.00 [77.58–77.58] | 80.60 ± 0.00 [80.60–80.60] | 80.59 ± 0.00 [80.59–80.59] | 83.14 ± 0.00 [83.14–83.14] |
| Random fixed config | 53.74 ± 16.92 [25.43–69.28] | 70.86 ± 5.34 [64.25–78.13] | 76.39 ± 4.38 [72.86–83.09] | 57.62 ± 29.64 [6.87–78.17] |

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
| Extreme warm-start policy − Cosine (extreme-retuned) | +0.05 [-0.75, +0.87]; 59%; 3/5, SD 0.33 | -0.81 [-1.74, +0.13]; 59%; 1/5, SD 0.60 | -0.83 [-1.75, +0.05]; 58%; 0/5, SD 0.49 | -1.61 [-3.75, +0.28]; 62%; 2/5, SD 2.13 |
| Cold extreme policy − Cosine (extreme-retuned) | -1.62 [-3.59, +0.05]; 48%; 1/5, SD 1.91 | -3.94 [-5.22, -2.71]; 16%; 0/5, SD 0.80 | -4.05 [-5.53, -2.62]; 11%; 0/5, SD 1.25 | -1.89 [-2.84, -1.05]; 24%; 0/5, SD 0.42 |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | +0.60 [-0.24, +1.48]; 63%; 5/5, SD 0.33 | -0.56 [-1.46, +0.38]; 58%; 1/5, SD 0.60 | -0.55 [-1.44, +0.33]; 57%; 1/5, SD 0.49 | -2.28 [-4.29, -0.46]; 36%; 0/5, SD 2.13 |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | -1.08 [-3.05, +0.63]; 50%; 2/5, SD 1.91 | -3.69 [-4.94, -2.47]; 12%; 0/5, SD 0.80 | -3.77 [-5.24, -2.33]; 10%; 0/5, SD 1.25 | -2.55 [-3.43, -1.76]; 13%; 0/5, SD 0.42 |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | +5.68 [+3.91, +7.59]; 81% | +6.13 [+4.62, +7.75]; 89% | +6.17 [+4.65, +7.79]; 89% | +4.17 [+2.85, +5.64]; 77% |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | +7.42 [+5.64, +9.27]; 65% | -2.08 [-2.57, -1.52]; 15% | -2.04 [-2.53, -1.48]; 13% | -2.08 [-2.72, -1.40]; 16% |
| Cosine (extreme-retuned) − Random schedule (cold) | +7.49 [+5.87, +9.26]; 92%; 5/5, SD 0.39 | +11.15 [+9.55, +12.90]; 98%; 5/5, SD 0.46 | +8.97 [+7.48, +10.60]; 97%; 5/5, SD 0.42 | +16.81 [+14.97, +18.73]; 99%; 5/5, SD 0.51 |
| Cosine (extreme-retuned) − Random schedule (warm) | +7.49 [+5.87, +9.26]; 92%; 5/5, SD 0.39 | +10.80 [+9.28, +12.42]; 97%; 5/5, SD 0.14 | +9.93 [+8.47, +11.52]; 97%; 5/5, SD 0.19 | +10.84 [+9.01, +12.82]; 96%; 5/5, SD 0.28 |
| Cosine (extreme-retuned) − Untrained policy (cold) | +4.87 [+2.90, +7.06]; 59%; 5/5, SD 0.00 | +8.48 [+6.42, +10.74]; 69%; 5/5, SD 0.00 | +8.39 [+6.35, +10.63]; 67%; 5/5, SD 0.00 | +2.99 [+1.40, +4.76]; 31%; 5/5, SD 0.00 |
| Cosine (extreme-retuned) − Untrained policy (warm) | +5.68 [+3.90, +7.58]; 81%; 5/5, SD 0.00 | +6.13 [+4.61, +7.74]; 89%; 5/5, SD 0.00 | +6.17 [+4.65, +7.79]; 89%; 5/5, SD 0.00 | +4.19 [+2.86, +5.65]; 77%; 5/5, SD 0.00 |
| Cosine (extreme-retuned) − Random fixed config | +29.52 [+18.22, +44.18]; 100%; 5/5, SD 16.92 | +15.86 [+11.38, +20.19]; 100%; 5/5, SD 5.34 | +10.37 [+6.28, +13.86]; 99%; 5/5, SD 4.38 | +29.72 [+11.35, +56.07]; 100%; 5/5, SD 29.64 |
| Cosine (regular-tuned) − Random schedule (cold) | +0.07 [-2.17, +2.34]; 56%; 4/5, SD 0.39 | +13.23 [+11.64, +14.92]; 97%; 5/5, SD 0.46 | +11.01 [+9.53, +12.60]; 96%; 5/5, SD 0.42 | +18.89 [+17.14, +20.68]; 98%; 5/5, SD 0.51 |
| Extreme warm-start policy − Random schedule (warm) | +7.54 [+6.12, +9.12]; 94%; 5/5, SD 0.59 | +9.99 [+8.74, +11.30]; 99%; 5/5, SD 0.53 | +9.10 [+7.91, +10.38]; 99%; 5/5, SD 0.37 | +9.23 [+7.32, +11.22]; 99%; 5/5, SD 2.01 |
| Cold extreme policy − Random schedule (cold) | +5.87 [+3.72, +8.03]; 91%; 5/5, SD 2.21 | +7.22 [+5.75, +8.68]; 95%; 5/5, SD 1.10 | +4.92 [+3.26, +6.56]; 90%; 5/5, SD 1.56 | +14.92 [+13.26, +16.65]; 99%; 5/5, SD 0.84 |

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
| Extreme warm-start policy − Cosine (extreme-retuned) | SGD | +8.77 [+4.14, +12.87] | +0.42 | +4.02 [+1.28, +7.29] | +0.27 |
| Extreme warm-start policy − Cosine (extreme-retuned) | Adam | -0.03 [-3.25, +2.98] | -0.00 | +0.93 [-1.12, +3.18] | +0.06 |
| Extreme warm-start policy − Cosine (extreme-retuned) | AdamW | +0.15 [-3.46, +3.45] | +0.01 | +0.98 [-1.11, +3.28] | +0.07 |
| Extreme warm-start policy − Cosine (extreme-retuned) | RMSprop | -2.30 [-6.74, +2.81] | -0.08 | +0.49 [-2.44, +3.98] | +0.02 |
| Cold extreme policy − Cosine (extreme-retuned) | SGD | +8.29 [+2.67, +13.38] | +0.26 | +4.93 [+1.19, +9.57] | +0.21 |
| Cold extreme policy − Cosine (extreme-retuned) | Adam | -2.55 [-7.44, +1.78] | -0.09 | +0.70 [-2.13, +3.73] | +0.03 |
| Cold extreme policy − Cosine (extreme-retuned) | AdamW | -2.89 [-8.07, +1.70] | -0.10 | +0.55 [-2.24, +3.46] | +0.03 |
| Cold extreme policy − Cosine (extreme-retuned) | RMSprop | -0.25 [-3.42, +2.51] | -0.01 | +0.87 [-0.99, +2.92] | +0.06 |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | SGD | +11.06 [+5.89, +15.63] | +0.50 | +4.98 [+1.92, +8.63] | +0.32 |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | Adam | +2.20 [-1.96, +6.01] | +0.11 | +1.84 [-0.45, +4.34] | +0.13 |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | AdamW | +2.13 [-2.19, +6.19] | +0.10 | +1.87 [-0.48, +4.44] | +0.13 |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | RMSprop | -1.95 [-6.03, +1.98] | -0.08 | +0.68 [-2.02, +3.75] | +0.04 |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | SGD | +10.58 [+4.50, +16.01] | +0.32 | +5.89 [+1.89, +10.77] | +0.25 |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | Adam | -0.32 [-6.16, +4.92] | -0.01 | +1.62 [-1.44, +4.93] | +0.08 |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | AdamW | -0.91 [-6.88, +4.42] | -0.03 | +1.44 [-1.55, +4.59] | +0.07 |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | RMSprop | +0.11 [-4.38, +3.70] | +0.01 | +1.05 [-1.24, +3.20] | +0.08 |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | SGD | -5.08 [-14.76, +5.53] | -0.10 | -5.52 [-11.80, +0.28] | -0.16 |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | Adam | +7.57 [-0.09, +13.48] | +0.18 | +0.13 [-5.22, +4.84] | +0.00 |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | AdamW | +7.70 [+0.04, +13.64] | +0.19 | +0.19 [-5.18, +4.91] | +0.01 |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | RMSprop | +3.62 [-2.79, +9.02] | +0.10 | -0.81 [-5.24, +3.01] | -0.03 |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | SGD | +25.66 [+17.13, +35.74] | +0.60 | +18.67 [+14.22, +23.00] | +0.61 |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | Adam | +3.85 [+0.20, +7.62] | +0.31 | +2.26 [+0.73, +3.98] | +0.26 |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | AdamW | +3.98 [+0.34, +7.72] | +0.32 | +2.31 [+0.78, +4.03] | +0.26 |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | RMSprop | +3.28 [-0.87, +6.70] | +0.21 | +1.70 [-0.25, +3.58] | +0.15 |
| Cosine (extreme-retuned) − Random schedule (cold) | SGD | +3.68 [-4.29, +11.82] | +0.09 | -1.34 [-6.63, +3.35] | -0.04 |
| Cosine (extreme-retuned) − Random schedule (cold) | Adam | +10.88 [+3.31, +17.65] | +0.28 | +1.91 [-3.64, +6.78] | +0.07 |
| Cosine (extreme-retuned) − Random schedule (cold) | AdamW | +9.53 [+1.97, +16.30] | +0.25 | +1.43 [-3.95, +6.20] | +0.05 |
| Cosine (extreme-retuned) − Random schedule (cold) | RMSprop | +12.02 [+3.72, +19.49] | +0.30 | +2.03 [-3.70, +7.22] | +0.07 |
| Cosine (extreme-retuned) − Random schedule (warm) | SGD | +3.68 [-4.29, +11.82] | +0.09 | -1.34 [-6.63, +3.35] | -0.04 |
| Cosine (extreme-retuned) − Random schedule (warm) | Adam | +8.74 [+1.05, +15.32] | +0.22 | +0.99 [-4.51, +5.95] | +0.04 |
| Cosine (extreme-retuned) − Random schedule (warm) | AdamW | +9.05 [+1.28, +15.49] | +0.23 | +1.20 [-4.11, +5.95] | +0.04 |
| Cosine (extreme-retuned) − Random schedule (warm) | RMSprop | +11.24 [+1.88, +19.04] | +0.24 | +1.40 [-4.86, +6.97] | +0.04 |
| Cosine (extreme-retuned) − Untrained policy (cold) | SGD | -1.05 [-11.55, +9.00] | -0.02 | -5.23 [-12.25, +0.99] | -0.13 |
| Cosine (extreme-retuned) − Untrained policy (cold) | Adam | +18.56 [+9.77, +26.14] | +0.31 | +5.05 [-1.80, +11.16] | +0.12 |
| Cosine (extreme-retuned) − Untrained policy (cold) | AdamW | +18.09 [+9.41, +25.41] | +0.31 | +4.59 [-2.13, +10.57] | +0.11 |
| Cosine (extreme-retuned) − Untrained policy (cold) | RMSprop | +2.13 [-5.95, +9.29] | +0.05 | -1.72 [-7.02, +2.94] | -0.05 |
| Cosine (extreme-retuned) − Untrained policy (warm) | SGD | -5.07 [-14.77, +5.54] | -0.10 | -5.52 [-11.80, +0.28] | -0.16 |
| Cosine (extreme-retuned) − Untrained policy (warm) | Adam | +7.54 [-0.11, +13.45] | +0.18 | +0.11 [-5.23, +4.80] | +0.00 |
| Cosine (extreme-retuned) − Untrained policy (warm) | AdamW | +7.70 [+0.04, +13.64] | +0.19 | +0.19 [-5.18, +4.91] | +0.01 |
| Cosine (extreme-retuned) − Untrained policy (warm) | RMSprop | +3.57 [-2.80, +8.87] | +0.10 | -0.82 [-5.25, +2.98] | -0.03 |
| Cosine (extreme-retuned) − Random fixed config | SGD | +3.50 [-13.63, +21.30] | +0.09 | +3.55 [-6.32, +15.15] | +0.13 |
| Cosine (extreme-retuned) − Random fixed config | Adam | +16.12 [+7.88, +26.03] | +0.49 | +5.59 [+0.23, +11.75] | +0.24 |
| Cosine (extreme-retuned) − Random fixed config | AdamW | +13.98 [+3.41, +24.20] | +0.35 | +3.98 [-2.62, +9.90] | +0.14 |
| Cosine (extreme-retuned) − Random fixed config | RMSprop | +22.67 [+6.87, +40.24] | +0.47 | +11.53 [+1.02, +22.53] | +0.33 |
| Cosine (regular-tuned) − Random schedule (cold) | SGD | -21.98 [-32.82, -13.91] | -0.38 | -20.01 [-26.26, -14.56] | -0.49 |
| Cosine (regular-tuned) − Random schedule (cold) | Adam | +7.03 [-0.98, +14.17] | +0.18 | -0.35 [-5.74, +4.38] | -0.01 |
| Cosine (regular-tuned) − Random schedule (cold) | AdamW | +5.55 [-2.29, +12.50] | +0.15 | -0.88 [-6.00, +3.67] | -0.03 |
| Cosine (regular-tuned) − Random schedule (cold) | RMSprop | +8.73 [+1.13, +16.35] | +0.24 | +0.33 [-5.00, +5.24] | +0.01 |
| Extreme warm-start policy − Random schedule (warm) | SGD | +12.45 [+4.82, +19.01] | +0.37 | +2.68 [-2.10, +7.00] | +0.11 |
| Extreme warm-start policy − Random schedule (warm) | Adam | +8.71 [+1.01, +14.91] | +0.31 | +1.92 [-2.44, +6.18] | +0.10 |
| Extreme warm-start policy − Random schedule (warm) | AdamW | +9.19 [+0.70, +15.65] | +0.33 | +2.18 [-2.31, +6.49] | +0.11 |
| Extreme warm-start policy − Random schedule (warm) | RMSprop | +8.94 [+3.32, +14.21] | +0.34 | +1.89 [-1.81, +5.29] | +0.10 |
| Cold extreme policy − Random schedule (cold) | SGD | +11.97 [+4.71, +17.87] | +0.41 | +3.59 [-1.08, +7.80] | +0.17 |
| Cold extreme policy − Random schedule (cold) | Adam | +8.34 [+0.14, +14.49] | +0.33 | +2.61 [-2.02, +6.67] | +0.15 |
| Cold extreme policy − Random schedule (cold) | AdamW | +6.64 [-1.94, +13.02] | +0.27 | +1.98 [-2.47, +6.06] | +0.11 |
| Cold extreme policy − Random schedule (cold) | RMSprop | +11.77 [+3.45, +18.85] | +0.39 | +2.90 [-2.25, +7.70] | +0.13 |

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
| Extreme warm-start policy − Cosine (extreme-retuned) | SGD | -1.74 [-4.08, +0.02] | -1.40 [-3.13, -0.00] | -0.98 [-2.55, +0.41] | -0.97 [-2.17, +0.15] | +3.22 [+0.91, +5.84] |
| Extreme warm-start policy − Cosine (extreme-retuned) | Adam | -1.11 [-3.30, +0.59] | -1.33 [-3.28, +0.28] | -0.86 [-3.04, +1.10] | -1.74 [-3.29, -0.19] | -0.64 [-2.75, +1.67] |
| Extreme warm-start policy − Cosine (extreme-retuned) | AdamW | -1.10 [-3.35, +0.59] | -1.33 [-3.33, +0.26] | -0.88 [-3.06, +1.06] | -1.81 [-3.39, -0.30] | -0.70 [-2.80, +1.59] |
| Extreme warm-start policy − Cosine (extreme-retuned) | RMSprop | -1.60 [-4.63, +0.73] | -1.72 [-4.59, +0.54] | -2.09 [-5.85, +0.90] | -2.41 [-5.50, +0.32] | -1.85 [-5.46, +1.43] |
| Cold extreme policy − Cosine (extreme-retuned) | SGD | -4.38 [-9.98, -0.32] | -3.08 [-6.99, -0.34] | -2.75 [-5.68, -0.29] | -3.14 [-5.79, -1.12] | +0.85 [-2.22, +4.03] |
| Cold extreme policy − Cosine (extreme-retuned) | Adam | -3.76 [-7.12, -1.04] | -4.08 [-7.13, -1.63] | -4.38 [-7.63, -1.58] | -4.55 [-6.57, -2.78] | -5.57 [-8.48, -2.78] |
| Cold extreme policy − Cosine (extreme-retuned) | AdamW | -3.63 [-6.73, -1.12] | -4.04 [-6.94, -1.70] | -4.57 [-7.95, -1.77] | -4.61 [-6.84, -2.74] | -5.85 [-9.17, -2.65] |
| Cold extreme policy − Cosine (extreme-retuned) | RMSprop | -1.70 [-3.44, -0.35] | -2.09 [-3.78, -0.76] | -2.11 [-4.62, -0.06] | -2.70 [-4.81, -0.78] | -1.93 [-3.95, +0.03] |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | SGD | -1.68 [-3.99, +0.06] | -1.17 [-2.85, +0.19] | -0.49 [-1.92, +0.84] | -0.42 [-1.59, +0.66] | +4.55 [+1.93, +7.53] |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | Adam | -1.22 [-3.38, +0.45] | -1.33 [-3.19, +0.23] | -0.73 [-2.62, +1.07] | -1.55 [-2.94, -0.13] | +0.29 [-1.98, +2.77] |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | AdamW | -1.16 [-3.39, +0.52] | -1.30 [-3.25, +0.28] | -0.70 [-2.71, +1.14] | -1.55 [-3.01, -0.12] | +0.19 [-2.03, +2.66] |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | RMSprop | -2.37 [-5.38, -0.10] | -2.54 [-5.36, -0.32] | -2.59 [-5.96, +0.06] | -3.00 [-5.65, -0.74] | -2.65 [-5.92, +0.33] |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | SGD | -4.32 [-9.88, -0.28] | -2.85 [-6.72, -0.15] | -2.27 [-5.09, +0.12] | -2.59 [-5.25, -0.57] | +2.18 [-1.11, +5.67] |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | Adam | -3.87 [-7.21, -1.17] | -4.07 [-7.05, -1.69] | -4.25 [-7.20, -1.66] | -4.36 [-6.21, -2.72] | -4.65 [-7.70, -1.72] |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | AdamW | -3.69 [-6.81, -1.19] | -4.01 [-6.88, -1.68] | -4.39 [-7.58, -1.73] | -4.34 [-6.46, -2.54] | -4.96 [-8.37, -1.64] |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | RMSprop | -2.47 [-4.19, -1.17] | -2.92 [-4.54, -1.63] | -2.62 [-4.60, -1.01] | -3.30 [-4.75, -1.98] | -2.73 [-5.32, -0.32] |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | SGD | +7.81 [+3.73, +12.76] | +7.45 [+3.96, +11.67] | +6.56 [+3.35, +10.15] | +7.45 [+4.09, +11.34] | +3.44 [-1.75, +9.21] |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | Adam | +5.96 [+2.60, +10.06] | +6.57 [+3.33, +10.51] | +5.91 [+2.60, +9.79] | +6.82 [+4.06, +9.92] | +8.68 [+4.81, +12.88] |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | AdamW | +5.98 [+2.61, +10.08] | +6.59 [+3.35, +10.52] | +5.94 [+2.61, +9.83] | +6.85 [+4.08, +9.97] | +8.76 [+4.88, +12.97] |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | RMSprop | +3.95 [+1.41, +7.05] | +4.35 [+1.93, +7.34] | +4.83 [+1.59, +8.74] | +5.07 [+2.29, +8.16] | +4.91 [+1.38, +8.61] |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | SGD | -1.04 [-1.75, -0.23] | +0.63 [-0.58, +1.98] | +4.52 [+1.74, +7.83] | +11.47 [+7.95, +15.45] | +19.39 [+14.91, +24.01] |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | Adam | -3.02 [-3.13, -2.90] | -2.85 [-3.09, -2.58] | -2.83 [-3.17, -2.45] | -1.33 [-2.11, -0.47] | -0.56 [-2.64, +1.73] |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | AdamW | -3.00 [-3.12, -2.89] | -2.84 [-3.07, -2.57] | -2.80 [-3.15, -2.41] | -1.30 [-2.08, -0.44] | -0.47 [-2.55, +1.83] |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | RMSprop | -3.15 [-3.43, -2.94] | -2.86 [-3.20, -2.53] | -2.24 [-3.45, -0.88] | -1.28 [-2.79, +0.42] | -1.22 [-3.45, +1.00] |
| Cosine (extreme-retuned) − Random schedule (cold) | SGD | +7.52 [+3.79, +12.10] | +7.48 [+4.14, +11.53] | +7.39 [+4.39, +11.02] | +9.75 [+6.60, +13.51] | +8.93 [+4.74, +13.60] |
| Cosine (extreme-retuned) − Random schedule (cold) | Adam | +10.65 [+6.68, +15.47] | +10.56 [+7.40, +14.30] | +10.48 [+7.46, +13.82] | +11.73 [+8.84, +15.19] | +15.48 [+11.35, +19.92] |
| Cosine (extreme-retuned) − Random schedule (cold) | AdamW | +8.49 [+4.85, +13.03] | +9.03 [+6.11, +12.48] | +8.20 [+5.46, +11.27] | +9.52 [+6.72, +12.87] | +12.55 [+8.53, +16.98] |
| Cosine (extreme-retuned) − Random schedule (cold) | RMSprop | +15.33 [+10.94, +20.29] | +14.75 [+11.09, +18.68] | +17.18 [+13.42, +21.19] | +18.01 [+14.25, +22.19] | +21.14 [+16.36, +25.99] |
| Cosine (extreme-retuned) − Random schedule (warm) | SGD | +7.52 [+3.79, +12.10] | +7.48 [+4.14, +11.53] | +7.39 [+4.39, +11.02] | +9.75 [+6.60, +13.51] | +8.93 [+4.74, +13.60] |
| Cosine (extreme-retuned) − Random schedule (warm) | Adam | +10.51 [+6.79, +15.05] | +10.81 [+7.72, +14.48] | +10.31 [+7.39, +13.61] | +11.50 [+8.68, +14.92] | +14.20 [+10.17, +18.59] |
| Cosine (extreme-retuned) − Random schedule (warm) | AdamW | +9.62 [+6.03, +14.05] | +9.91 [+6.80, +13.61] | +9.32 [+6.50, +12.62] | +10.65 [+7.94, +13.85] | +13.43 [+9.47, +17.69] |
| Cosine (extreme-retuned) − Random schedule (warm) | RMSprop | +9.67 [+5.77, +14.46] | +10.25 [+6.69, +14.50] | +10.53 [+6.86, +14.89] | +12.05 [+8.39, +16.24] | +14.66 [+9.62, +20.02] |
| Cosine (extreme-retuned) − Untrained policy (cold) | SGD | +7.50 [+2.46, +13.69] | +6.56 [+2.28, +11.75] | +5.58 [+1.73, +10.20] | +6.12 [+2.22, +10.89] | +3.99 [-1.29, +10.04] |
| Cosine (extreme-retuned) − Untrained policy (cold) | Adam | +6.67 [+1.53, +13.19] | +6.90 [+2.40, +12.39] | +7.85 [+3.51, +12.91] | +9.64 [+5.54, +14.48] | +16.36 [+11.51, +21.78] |
| Cosine (extreme-retuned) − Untrained policy (cold) | AdamW | +6.61 [+1.51, +13.12] | +6.85 [+2.40, +12.32] | +7.86 [+3.54, +12.88] | +9.56 [+5.46, +14.42] | +16.03 [+11.27, +21.33] |
| Cosine (extreme-retuned) − Untrained policy (cold) | RMSprop | +3.69 [+0.28, +7.89] | +3.62 [+0.51, +7.42] | +3.67 [+0.04, +8.05] | +3.87 [+0.59, +7.61] | +3.26 [-0.94, +7.86] |
| Cosine (extreme-retuned) − Untrained policy (warm) | SGD | +7.80 [+3.72, +12.76] | +7.45 [+3.96, +11.67] | +6.55 [+3.34, +10.14] | +7.44 [+4.08, +11.34] | +3.44 [-1.75, +9.21] |
| Cosine (extreme-retuned) − Untrained policy (warm) | Adam | +5.97 [+2.60, +10.07] | +6.56 [+3.33, +10.49] | +5.91 [+2.60, +9.79] | +6.80 [+4.04, +9.89] | +8.67 [+4.80, +12.87] |
| Cosine (extreme-retuned) − Untrained policy (warm) | AdamW | +5.98 [+2.61, +10.08] | +6.59 [+3.35, +10.52] | +5.94 [+2.61, +9.83] | +6.85 [+4.08, +9.97] | +8.76 [+4.88, +12.97] |
| Cosine (extreme-retuned) − Untrained policy (warm) | RMSprop | +3.96 [+1.43, +7.05] | +4.35 [+1.94, +7.34] | +4.87 [+1.61, +8.78] | +5.10 [+2.31, +8.19] | +4.90 [+1.40, +8.58] |
| Cosine (extreme-retuned) − Random fixed config | SGD | +27.24 [+12.12, +46.97] | +28.45 [+14.94, +46.39] | +30.45 [+18.51, +45.30] | +32.89 [+20.66, +46.80] | +32.22 [+23.85, +41.41] |
| Cosine (extreme-retuned) − Random fixed config | Adam | +11.44 [+5.99, +17.19] | +13.40 [+8.16, +18.90] | +15.35 [+9.93, +20.10] | +17.62 [+12.60, +22.68] | +23.18 [+16.83, +29.25] |
| Cosine (extreme-retuned) − Random fixed config | AdamW | +7.01 [+3.18, +11.30] | +8.50 [+4.30, +12.65] | +10.29 [+4.96, +15.43] | +12.06 [+7.31, +16.31] | +16.36 [+8.10, +23.88] |
| Cosine (extreme-retuned) − Random fixed config | RMSprop | +23.27 [+6.82, +48.40] | +26.63 [+8.39, +54.33] | +28.73 [+10.36, +56.08] | +32.64 [+12.72, +60.77] | +39.73 [+17.24, +64.08] |
| Cosine (regular-tuned) − Random schedule (cold) | SGD | +8.56 [+4.71, +13.24] | +6.86 [+3.04, +11.39] | +2.88 [-1.70, +7.69] | -1.72 [-6.91, +3.79] | -10.46 [-15.64, -5.21] |
| Cosine (regular-tuned) − Random schedule (cold) | Adam | +13.66 [+9.70, +18.51] | +13.41 [+10.26, +17.14] | +13.31 [+10.38, +16.56] | +13.06 [+10.15, +16.54] | +16.05 [+11.69, +20.52] |
| Cosine (regular-tuned) − Random schedule (cold) | AdamW | +11.49 [+7.84, +16.03] | +11.87 [+8.99, +15.28] | +11.00 [+8.37, +13.95] | +10.82 [+8.05, +14.21] | +13.03 [+8.82, +17.42] |
| Cosine (regular-tuned) − Random schedule (cold) | RMSprop | +18.49 [+14.14, +23.46] | +17.62 [+13.98, +21.57] | +19.43 [+15.83, +23.24] | +19.29 [+15.88, +23.18] | +22.36 [+17.82, +26.95] |
| Extreme warm-start policy − Random schedule (warm) | SGD | +5.78 [+3.65, +8.49] | +6.08 [+4.01, +8.61] | +6.42 [+4.40, +8.94] | +8.78 [+5.98, +12.07] | +12.14 [+7.87, +16.89] |
| Extreme warm-start policy − Random schedule (warm) | Adam | +9.39 [+6.98, +12.35] | +9.48 [+7.46, +11.87] | +9.45 [+7.65, +11.50] | +9.76 [+7.61, +12.40] | +13.56 [+9.61, +17.80] |
| Extreme warm-start policy − Random schedule (warm) | AdamW | +8.52 [+6.33, +11.35] | +8.57 [+6.59, +10.95] | +8.44 [+6.73, +10.46] | +8.83 [+6.75, +11.33] | +12.73 [+8.80, +17.00] |
| Extreme warm-start policy − Random schedule (warm) | RMSprop | +8.06 [+5.41, +11.21] | +8.53 [+5.95, +11.69] | +8.45 [+6.05, +11.27] | +9.65 [+6.73, +12.85] | +12.81 [+9.45, +16.77] |
| Cold extreme policy − Random schedule (cold) | SGD | +3.14 [-0.06, +6.35] | +4.40 [+2.22, +6.98] | +4.64 [+2.33, +7.50] | +6.61 [+3.61, +10.00] | +9.77 [+5.18, +14.87] |
| Cold extreme policy − Random schedule (cold) | Adam | +6.89 [+4.24, +9.85] | +6.48 [+4.45, +8.61] | +6.10 [+4.22, +8.05] | +7.18 [+4.88, +9.79] | +9.91 [+5.63, +14.13] |
| Cold extreme policy − Random schedule (cold) | AdamW | +4.86 [+2.60, +7.54] | +4.99 [+2.96, +7.10] | +3.63 [+1.54, +5.80] | +4.91 [+2.57, +7.47] | +6.70 [+2.28, +11.26] |
| Cold extreme policy − Random schedule (cold) | RMSprop | +13.64 [+10.01, +17.73] | +12.66 [+9.65, +15.92] | +15.07 [+12.03, +18.32] | +15.31 [+12.11, +18.80] | +19.21 [+14.52, +24.01] |

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
| Extreme warm-start policy − Cosine (extreme-retuned) | SGD | +7.54 [-1.66, +18.49] | 3/7 | -0.37 [-1.28, +0.49] | 56% | neither CI excludes 0 |
| Extreme warm-start policy − Cosine (extreme-retuned) | Adam | +1.77 [-6.51, +9.56] | 5/7 | -1.14 [-2.09, -0.19] | 57% | **opposite sign, generated CI excludes 0** |
| Extreme warm-start policy − Cosine (extreme-retuned) | AdamW | +2.21 [-6.23, +9.96] | 4/7 | -1.17 [-2.13, -0.28] | 56% | **opposite sign, generated CI excludes 0** |
| Extreme warm-start policy − Cosine (extreme-retuned) | RMSprop | +1.37 [-4.05, +6.48] | 3/7 | -1.93 [-4.10, +0.04] | 60% | neither CI excludes 0 |
| Cold extreme policy − Cosine (extreme-retuned) | SGD | +5.21 [-3.88, +15.77] | 4/7 | -2.49 [-4.87, -0.51] | 46% | **opposite sign, generated CI excludes 0** |
| Cold extreme policy − Cosine (extreme-retuned) | Adam | -3.48 [-11.74, +4.79] | 2/7 | -4.47 [-5.85, -3.19] | 15% | same sign, now CI excludes 0 |
| Cold extreme policy − Cosine (extreme-retuned) | AdamW | -3.56 [-11.35, +4.15] | 2/7 | -4.54 [-6.11, -3.11] | 10% | same sign, now CI excludes 0 |
| Cold extreme policy − Cosine (extreme-retuned) | RMSprop | +2.70 [-1.51, +7.22] | 3/7 | -2.11 [-3.08, -1.25] | 24% | **opposite sign, generated CI excludes 0** |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | SGD | +8.46 [-1.40, +21.08] | 3/7 | +0.16 [-0.76, +1.07] | 61% | neither CI excludes 0 |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | Adam | +3.86 [-5.94, +13.59] | 5/7 | -0.91 [-1.84, +0.03] | 56% | neither CI excludes 0 |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | AdamW | +3.67 [-6.11, +12.87] | 5/7 | -0.90 [-1.85, -0.01] | 56% | **opposite sign, generated CI excludes 0** |
| Extreme warm-start policy − Warmup-cosine (extreme-retuned) | RMSprop | -0.74 [-9.97, +8.35] | 3/7 | -2.63 [-4.71, -0.72] | 34% | same sign, now CI excludes 0 |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | SGD | +6.13 [-3.57, +18.39] | 4/7 | -1.96 [-4.34, +0.03] | 48% | neither CI excludes 0 |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | Adam | -1.39 [-11.07, +8.94] | 2/7 | -4.24 [-5.59, -2.99] | 12% | same sign, now CI excludes 0 |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | AdamW | -2.10 [-11.02, +6.96] | 2/7 | -4.28 [-5.83, -2.85] | 9% | same sign, now CI excludes 0 |
| Cold extreme policy − Warmup-cosine (extreme-retuned) | RMSprop | +0.59 [-6.26, +7.86] | 2/7 | -2.81 [-3.72, -1.99] | 12% | **opposite sign, generated CI excludes 0** |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | SGD | +0.95 [-18.31, +17.47] | 5/7 | +6.54 [+4.68, +8.52] | 82% | same sign, now CI excludes 0 |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | Adam | +6.79 [-4.81, +18.98] | 4/7 | +6.79 [+5.24, +8.49] | 90% | same sign, now CI excludes 0 |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | AdamW | +6.78 [-4.82, +19.02] | 4/7 | +6.83 [+5.27, +8.52] | 90% | same sign, now CI excludes 0 |
| Cosine (extreme-retuned) − Fixed (extreme-retuned) | RMSprop | +0.05 [-5.31, +6.11] | 3/7 | +4.62 [+3.26, +6.08] | 78% | same sign, now CI excludes 0 |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | SGD | +14.61 [+3.22, +28.11] | 6/7 | +7.00 [+5.35, +8.78] | 64% | replicates |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | Adam | +6.88 [+0.49, +13.99] | 5/7 | -2.12 [-2.58, -1.58] | 14% | **reverses** |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | AdamW | +6.87 [+0.50, +13.96] | 5/7 | -2.08 [-2.54, -1.55] | 12% | **reverses** |
| Cosine (extreme-retuned) − Cosine (regular-tuned) | RMSprop | +1.87 [-3.35, +6.98] | 4/7 | -2.15 [-2.77, -1.50] | 15% | **opposite sign, generated CI excludes 0** |
| Cosine (extreme-retuned) − Random schedule (cold) | SGD | +7.43 [-4.09, +19.65] | 5/7 | +8.22 [+6.54, +10.06] | 92% | same sign, now CI excludes 0 |
| Cosine (extreme-retuned) − Random schedule (cold) | Adam | +17.37 [+10.19, +24.48] | 7/7 | +11.78 [+10.11, +13.54] | 98% | replicates |
| Cosine (extreme-retuned) − Random schedule (cold) | AdamW | +15.43 [+8.24, +22.51] | 7/7 | +9.56 [+7.98, +11.25] | 97% | replicates |
| Cosine (extreme-retuned) − Random schedule (cold) | RMSprop | +20.54 [+13.96, +28.20] | 7/7 | +17.28 [+15.43, +19.24] | 100% | replicates |
| Cosine (extreme-retuned) − Random schedule (warm) | SGD | +7.43 [-4.09, +19.65] | 5/7 | +8.22 [+6.54, +10.06] | 92% | same sign, now CI excludes 0 |
| Cosine (extreme-retuned) − Random schedule (warm) | Adam | +14.82 [+7.36, +22.17] | 6/7 | +11.47 [+9.88, +13.16] | 98% | replicates |
| Cosine (extreme-retuned) − Random schedule (warm) | AdamW | +13.56 [+6.14, +20.94] | 6/7 | +10.58 [+9.02, +12.25] | 98% | replicates |
| Cosine (extreme-retuned) − Random schedule (warm) | RMSprop | +13.35 [+7.51, +19.86] | 7/7 | +11.44 [+9.54, +13.43] | 96% | replicates |
| Cosine (extreme-retuned) − Untrained policy (cold) | SGD | +4.84 [-9.55, +18.70] | 5/7 | +5.94 [+3.79, +8.25] | 61% | same sign, now CI excludes 0 |
| Cosine (extreme-retuned) − Untrained policy (cold) | Adam | +23.35 [+10.14, +39.36] | 7/7 | +9.49 [+7.29, +11.94] | 70% | replicates |
| Cosine (extreme-retuned) − Untrained policy (cold) | AdamW | +23.46 [+10.22, +39.47] | 7/7 | +9.39 [+7.21, +11.82] | 68% | replicates |
| Cosine (extreme-retuned) − Untrained policy (cold) | RMSprop | +1.39 [-4.95, +8.05] | 4/7 | +3.62 [+1.96, +5.42] | 34% | same sign, now CI excludes 0 |
| Cosine (extreme-retuned) − Untrained policy (warm) | SGD | +0.95 [-18.32, +17.46] | 5/7 | +6.53 [+4.67, +8.52] | 82% | same sign, now CI excludes 0 |
| Cosine (extreme-retuned) − Untrained policy (warm) | Adam | +6.77 [-4.80, +18.95] | 4/7 | +6.79 [+5.24, +8.48] | 90% | same sign, now CI excludes 0 |
| Cosine (extreme-retuned) − Untrained policy (warm) | AdamW | +6.78 [-4.82, +19.02] | 4/7 | +6.83 [+5.27, +8.52] | 90% | same sign, now CI excludes 0 |
| Cosine (extreme-retuned) − Untrained policy (warm) | RMSprop | +0.12 [-5.26, +6.12] | 4/7 | +4.64 [+3.28, +6.09] | 78% | same sign, now CI excludes 0 |
| Cosine (extreme-retuned) − Random fixed config | SGD | +27.05 [+12.22, +43.00] | 6/7 | +30.25 [+19.27, +44.84] | 100% | replicates |
| Cosine (extreme-retuned) − Random fixed config | Adam | +22.45 [+11.39, +35.91] | 7/7 | +16.21 [+11.80, +20.57] | 100% | replicates |
| Cosine (extreme-retuned) − Random fixed config | AdamW | +17.64 [+6.21, +29.97] | 7/7 | +10.85 [+7.12, +14.09] | 100% | replicates |
| Cosine (extreme-retuned) − Random fixed config | RMSprop | +33.87 [+14.07, +56.42] | 7/7 | +30.22 [+12.20, +56.21] | 100% | replicates |
| Cosine (regular-tuned) − Random schedule (cold) | SGD | -7.18 [-16.95, +1.40] | 2/7 | +1.21 [-1.14, +3.52] | 58% | neither CI excludes 0 |
| Cosine (regular-tuned) − Random schedule (cold) | Adam | +10.48 [+2.78, +17.84] | 5/7 | +13.90 [+12.23, +15.63] | 98% | replicates |
| Cosine (regular-tuned) − Random schedule (cold) | AdamW | +8.56 [+0.65, +15.87] | 5/7 | +11.64 [+10.07, +13.26] | 96% | replicates |
| Cosine (regular-tuned) − Random schedule (cold) | RMSprop | +18.67 [+9.71, +27.60] | 7/7 | +19.43 [+17.67, +21.26] | 98% | replicates |
| Extreme warm-start policy − Random schedule (warm) | SGD | +14.98 [+9.05, +20.86] | 7/7 | +7.84 [+6.45, +9.37] | 94% | replicates |
| Extreme warm-start policy − Random schedule (warm) | Adam | +16.59 [+11.19, +22.02] | 7/7 | +10.33 [+9.06, +11.66] | 99% | replicates |
| Extreme warm-start policy − Random schedule (warm) | AdamW | +15.77 [+10.18, +21.38] | 7/7 | +9.42 [+8.23, +10.69] | 100% | replicates |
| Extreme warm-start policy − Random schedule (warm) | RMSprop | +14.72 [+9.57, +20.69] | 7/7 | +9.50 [+7.56, +11.55] | 100% | replicates |
| Cold extreme policy − Random schedule (cold) | SGD | +12.65 [+7.42, +18.11] | 7/7 | +5.72 [+3.40, +7.95] | 90% | replicates |
| Cold extreme policy − Random schedule (cold) | Adam | +13.88 [+8.24, +19.95] | 7/7 | +7.31 [+5.85, +8.80] | 95% | replicates |
| Cold extreme policy − Random schedule (cold) | AdamW | +11.87 [+6.47, +17.70] | 7/7 | +5.02 [+3.39, +6.69] | 90% | replicates |
| Cold extreme policy − Random schedule (cold) | RMSprop | +23.24 [+16.92, +30.46] | 7/7 | +15.17 [+13.53, +16.92] | 100% | replicates |
