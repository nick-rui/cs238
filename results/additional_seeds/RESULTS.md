# Five-policy-seed results

Accuracy is in percent. Policy values are mean ± sample standard deviation of
episode-averaged accuracy across policy seeds 338–342. Fixed values reuse the
original tuned baselines. Wins count policy seeds whose episode-averaged
accuracy strictly exceeds fixed, not individual paired episodes. Biased means
equally weight nine regular or seven extreme conditions, excluding `mixed`.
These descriptive summaries are not significance tests.

| Suite | Optimizer | Mixed policy | Mixed fixed | Wins / 5 | Biased policy | Biased fixed | Wins / 5 |
|---|---|---:|---:|---:|---:|---:|---:|
| Regular | SGD | 92.48 ± 0.42 | 92.27 | 3 | 89.61 ± 0.51 | 88.36 | 5 |
| Regular | Adam | 92.72 ± 0.23 | 92.63 | 4 | 90.31 ± 0.66 | 91.50 | 0 |
| Regular | AdamW | 92.67 ± 0.38 | 92.63 | 2 | 90.34 ± 0.80 | 91.50 | 0 |
| Regular | RMSprop | 93.56 ± 0.12 | 93.19 | 5 | 92.57 ± 0.07 | 92.37 | 5 |
| Extreme | SGD | 92.55 ± 0.49 | 92.03 | 4 | 51.66 ± 1.23 | 47.55 | 5 |
| Extreme | Adam | 92.81 ± 0.27 | 92.26 | 5 | 46.05 ± 10.02 | 63.36 | 0 |
| Extreme | AdamW | 92.76 ± 0.39 | 92.26 | 5 | 49.04 ± 10.77 | 63.36 | 0 |
| Extreme | RMSprop | 93.64 ± 0.14 | 93.37 | 5 | 67.56 ± 0.92 | 67.00 | 4 |

Additional seeds preserve SGD's biased-stream advantage and RMSprop's smaller
regular advantage. They expose substantial extreme-order seed variability for
Adam and AdamW. Neither beats its fixed baseline on the aggregate biased suite
for any of the five policy seeds. The full condition breakdown is in
`summary.csv`; individual seed means and failure counts are in
`policy_seed_summary.csv`.

## Measured execution cost

Four concurrent processes, each with one Torch intra-op thread, on the measured
8-vCPU Xeon CPU machine. Times include development checks and checkpoint I/O;
per-job times include contention with other training jobs.

| Optimizer | Seed 340 seconds | Seed 341 seconds | Seed 342 seconds |
|---|---:|---:|---:|
| SGD | 1376.62 | 1265.42 | 1258.99 |
| Adam | 1985.59 | 1757.90 | 1763.37 |
| AdamW | 1803.96 | 1707.15 | 1697.49 |
| RMSprop | 1649.15 | 1615.93 | 1575.22 |

Total training-phase wall time: **5277.65 seconds (87.96 minutes)**.
Full pipeline including evaluations, replays, summaries and validation:
**5915.83 seconds (98.60 minutes)**.

## Validation and failures

- All twelve trainings completed 500 updates / 4000 episodes.
- All 24 new evaluation grids are complete: 200 regular or 160 extreme episodes
  per checkpoint, and 40 action-trace records per episode. No failed episodes
  were dropped.
- All 24 representative replays passed the original reproduction script's
  metric/action tolerances, including the second memory coefficient. Maximum
  absolute metric discrepancy was **5.96e-7**.
- No new regular evaluation episode failed. Extreme failures: SGD seed 341 had
  one `class_sorted` failure; Adam seeds 341/342 had eight/four `staircase`
  failures. AdamW and RMSprop had none. Original failure handling is retained.
- SHA256 validation confirms all **62 original tracked result files** remain
  byte-identical. Original code checks were run with output redirected outside
  the original results directories.

Exact timings, failures, checkpoint selections, and replay errors are retained
in the per-job JSON records. Training histories and intermediate checkpoints
are not committed, matching the original studies' artifact policy.
