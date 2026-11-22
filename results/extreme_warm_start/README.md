# Warm-started controllers trained on extreme orderings

This experiment combines the two changes that each helped in the paper:
**warm-start initialization** (`results/warm_start`) and **retraining on the
extreme orderings** (`results/extreme_retrained_five_seeds`). Twenty new
policies (seeds 338–342, four optimizers) are trained on the extreme training
distribution. Each starts from the **extreme-retuned fixed configuration**
(`results/extreme_retrained/<family>_baseline.json`), which was selected only
on extreme development episodes. This initialization was possible for all
four families. The policies are then compared, paired, with three controls.

## Design

- **Training** calls `retrain.train` unchanged: 500 updates × 8 episodes, with
  the same alternating extreme training halves (first/last four `ORDERS`, each
  repeated twice, episode *i* paired with *i*+4 as the baseline) and the same
  training-episode seed `6000000 + 10000·policy_seed + 8·update + episode`. So
  each warm policy sees exactly the episodes its seed-matched cold-retrained
  policy saw. Policy Adam lr is 0.003, with gradient clipping at 1.
- **Warm start** uses the existing `warm=` path. The output-layer bias is set
  so the policy's initial mean action reproduces the config, and the rollout's
  optimizer state starts at that config through each engine's `warm_settings`
  clamps. Warm configs: SGD lr 0.1, momentum 0, wd 0 (clamped to 1e-6); Adam
  and AdamW lr 1e-3, β1 0.5, β2 0.99, wd 0 or 1e-4; RMSprop lr 1e-4,
  momentum 0, α 0.9999, wd 0.
- **Checkpoint selection** uses only development data: extreme dev seeds
  1930000–1930015 (each ordering twice) and dev images, minimum mean
  `b.score` at updates 0, 25, 50, 100, …, 500. Test accuracy is never used.
- **Test**: the full extreme suite, eight orderings × seeds 1920000–1920019,
  on the official MNIST test set, with batches of four episodes and one torch
  thread. Failed episodes are kept at their recorded accuracy.
- **Controls**, all re-evaluated fresh in this environment on the same
  episodes:
  - `fixed_retuned`: the extreme-retuned fixed config.
  - `policy_retrained`: cold-start extreme-retrained policies. Seeds 338–339
    come from `results/extreme_retrained`, 340–342 from
    `results/extreme_retrained_five_seeds`.
  - `warm_regular`: the regular-distribution warm-start policies from
    `results/warm_start`, warm from the regular-tuned config.
- **Pairing**: differences are taken per policy seed, so the cold and regular
  warm controls are seed-matched. Each seed's difference averages the
  orderings and the 20 test seeds. Reported as mean ± sample SD over the five
  seeds, with wins (strictly positive differences) out of 5. "biased" means
  the seven non-`mixed` orderings; "all" means all eight.

## Results

### Seven biased orderings

Mean test accuracy (%) ± policy-seed SD [min–max]; fixed has no seed variation.

| Optimizer | Extreme-retuned fixed | Cold retrained | Regular warm start | **Extreme warm start** |
|---|---:|---:|---:|---:|
| SGD | 61.32 | 67.48 ± 1.86 [65.27–68.97] | 50.10 ± 3.84 [43.90–53.40] (40 failed) | 69.81 ± 1.23 [68.00–71.28] |
| Adam | 65.91 | 69.22 ± 2.81 [66.68–73.54] | 66.39 ± 2.78 [63.76–69.91] | 74.47 ± 1.20 [72.37–75.37] |
| AdamW | 65.91 | 69.13 ± 3.47 [64.50–74.21] | 67.00 ± 2.01 [64.66–69.53] | 74.90 ± 1.80 [72.45–77.48] |
| RMSprop | 70.22 | 72.98 ± 1.05 [72.12–74.67] | 66.95 ± 2.33 [63.32–69.23] | 71.65 ± 1.37 [70.36–73.49] |

Extreme warm start minus control (pp, mean ± SD over the five matched policy seeds; wins out of 5):

| Optimizer | vs extreme-retuned fixed | vs cold retrained | vs regular warm start |
|---|---:|---:|---:|
| SGD | +8.50 ± 1.23 (5/5) | +2.33 ± 1.63 (5/5) | +19.71 ± 3.95 (5/5) |
| Adam | +8.56 ± 1.20 (5/5) | +5.25 ± 2.21 (5/5) | +8.08 ± 2.50 (5/5) |
| AdamW | +8.99 ± 1.80 (5/5) | +5.78 ± 1.72 (5/5) | +7.91 ± 2.91 (5/5) |
| RMSprop | +1.42 ± 1.37 (5/5) | -1.33 ± 2.07 (2/5) | +4.69 ± 3.64 (5/5) |

### All eight orderings (including mixed)

Mean test accuracy (%) ± policy-seed SD [min–max]; fixed has no seed variation.

| Optimizer | Extreme-retuned fixed | Cold retrained | Regular warm start | **Extreme warm start** |
|---|---:|---:|---:|---:|
| SGD | 64.76 | 70.57 ± 1.73 [68.50–71.98] | 55.48 ± 3.41 [49.95–58.39] (40 failed) | 72.50 ± 1.08 [70.91–73.80] |
| Adam | 68.94 | 71.99 ± 2.45 [69.81–75.82] | 69.81 ± 2.42 [67.53–72.88] | 76.72 ± 0.98 [75.02–77.49] |
| AdamW | 68.94 | 71.90 ± 3.05 [67.85–76.39] | 70.33 ± 1.73 [68.34–72.55] | 77.11 ± 1.54 [75.06–79.33] |
| RMSprop | 72.78 | 75.26 ± 0.91 [74.53–76.75] | 70.29 ± 2.03 [67.11–72.27] | 74.20 ± 1.31 [72.90–75.90] |

Extreme warm start minus control (pp, mean ± SD over the five matched policy seeds; wins out of 5):

| Optimizer | vs extreme-retuned fixed | vs cold retrained | vs regular warm start |
|---|---:|---:|---:|
| SGD | +7.74 ± 1.08 (5/5) | +1.93 ± 1.50 (5/5) | +17.02 ± 3.52 (5/5) |
| Adam | +7.78 ± 0.98 (5/5) | +4.73 ± 1.95 (5/5) | +6.91 ± 2.23 (5/5) |
| AdamW | +8.17 ± 1.54 (5/5) | +5.21 ± 1.55 (5/5) | +6.78 ± 2.54 (5/5) |
| RMSprop | +1.43 ± 1.31 (5/5) | -1.05 ± 1.88 (2/5) | +3.92 ± 3.29 (5/5) |

### Per ordering

Mean accuracy (%) of the extreme warm start over five seeds, then its paired difference (pp) vs extreme-retuned fixed / cold retrained / regular warm start, with wins out of 5.

**SGD**

| Ordering | Fixed | Cold | Reg. warm | Ext. warm | Δ fixed | Δ cold | Δ reg. warm |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 88.86 | 92.14 | 93.09 | 91.28 | +2.42 (5) | -0.86 (1) | -1.80 (0) |
| single_max | 80.40 | 82.24 | 81.70 | 81.34 | +0.94 (5) | -0.90 (1) | -0.37 (1) |
| single_late | 82.10 | 79.76 | 72.80 | 80.97 | -1.13 (0) | +1.22 (3) | +8.17 (5) |
| pair_max | 73.45 | 78.83 | 70.39 | 83.25 | +9.80 (5) | +4.41 (4) | +12.85 (5) |
| half_block | 49.33 | 74.11 | 46.12 | 78.67 | +29.34 (5) | +4.55 (4) | +32.55 (5) |
| staircase | 75.33 | 57.03 | 38.88 | 60.03 | -15.30 (0) | +3.00 (3) | +21.16 (5) |
| class_sorted | 9.87 | 22.55 | 9.86 | 24.04 | +14.17 (5) | +1.49 (2) | +14.18 (5) |
| alternating_halves | 58.74 | 77.86 | 30.96 | 80.41 | +21.66 (5) | +2.54 (4) | +49.44 (5) |

**Adam**

| Ordering | Fixed | Cold | Reg. warm | Ext. warm | Δ fixed | Δ cold | Δ reg. warm |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 90.18 | 91.40 | 93.72 | 92.50 | +2.32 (5) | +1.10 (5) | -1.23 (0) |
| single_max | 81.88 | 83.77 | 84.80 | 84.27 | +2.39 (5) | +0.50 (4) | -0.53 (2) |
| single_late | 84.79 | 68.95 | 84.56 | 82.57 | -2.22 (0) | +13.63 (4) | -1.99 (2) |
| pair_max | 79.21 | 81.87 | 77.60 | 85.52 | +6.31 (5) | +3.65 (5) | +7.93 (5) |
| half_block | 52.07 | 65.66 | 48.32 | 68.14 | +16.07 (5) | +2.48 (4) | +19.82 (5) |
| staircase | 83.10 | 78.55 | 86.85 | 81.89 | -1.21 (2) | +3.34 (4) | -4.96 (0) |
| class_sorted | 13.80 | 24.59 | 12.26 | 38.64 | +24.84 (5) | +14.05 (5) | +26.37 (5) |
| alternating_halves | 66.52 | 81.15 | 70.36 | 80.26 | +13.74 (5) | -0.89 (3) | +9.90 (5) |

**AdamW**

| Ordering | Fixed | Cold | Reg. warm | Ext. warm | Δ fixed | Δ cold | Δ reg. warm |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 90.18 | 91.31 | 93.67 | 92.57 | +2.40 (5) | +1.27 (5) | -1.10 (0) |
| single_max | 81.87 | 83.84 | 84.86 | 84.13 | +2.26 (5) | +0.30 (4) | -0.73 (0) |
| single_late | 84.79 | 71.51 | 85.75 | 83.23 | -1.56 (0) | +11.72 (5) | -2.53 (1) |
| pair_max | 79.21 | 79.89 | 78.54 | 85.07 | +5.86 (5) | +5.18 (5) | +6.53 (5) |
| half_block | 52.07 | 66.20 | 49.14 | 69.72 | +17.65 (5) | +3.52 (3) | +20.59 (5) |
| staircase | 83.10 | 76.75 | 87.19 | 82.47 | -0.63 (2) | +5.72 (3) | -4.72 (0) |
| class_sorted | 13.80 | 24.24 | 11.86 | 39.88 | +26.09 (5) | +15.65 (5) | +28.02 (5) |
| alternating_halves | 66.52 | 81.47 | 71.64 | 79.82 | +13.29 (5) | -1.65 (2) | +8.18 (4) |

**RMSprop**

| Ordering | Fixed | Cold | Reg. warm | Ext. warm | Δ fixed | Δ cold | Δ reg. warm |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | 90.66 | 91.24 | 93.62 | 92.10 | +1.45 (4) | +0.86 (4) | -1.52 (0) |
| single_max | 82.90 | 84.10 | 84.71 | 84.50 | +1.60 (4) | +0.39 (3) | -0.22 (2) |
| single_late | 86.40 | 81.74 | 83.49 | 86.64 | +0.24 (3) | +4.89 (5) | +3.14 (4) |
| pair_max | 81.19 | 84.25 | 77.22 | 83.42 | +2.23 (5) | -0.83 (2) | +6.20 (5) |
| half_block | 56.02 | 67.30 | 50.13 | 58.94 | +2.93 (5) | -8.36 (1) | +8.81 (5) |
| staircase | 86.55 | 85.78 | 87.41 | 87.25 | +0.70 (4) | +1.48 (4) | -0.15 (1) |
| class_sorted | 25.05 | 24.28 | 14.56 | 23.68 | -1.37 (1) | -0.59 (2) | +9.12 (5) |
| alternating_halves | 73.45 | 83.38 | 71.13 | 77.10 | +3.65 (5) | -6.28 (0) | +5.96 (5) |

**Summary.** On the biased average, the extreme warm start beats the
extreme-retuned fixed config for every seed of every optimizer: +8.5 to
+9.0 pp for SGD, Adam and AdamW, and +1.4 pp for RMSprop. It also beats the
regular warm start in 20/20 seed pairs. Against cold-start extreme
retraining, it wins 5/5 for SGD (+2.3 pp), Adam (+5.3) and AdamW (+5.8). It
does not help RMSprop: −1.3 ± 2.1 pp, 2/5 wins. Seed SD is lower than cold
retraining for SGD, Adam and AdamW (1.2–1.8 vs 1.9–3.5 pp).

The gains are not uniform across orderings. As with cold retraining, the
extreme warm start loses to the fixed config on `single_late` for all five
seeds of SGD, Adam and AdamW, and on SGD `staircase` (−15.3 pp). On `mixed`,
the shuffled reference, the regular warm start stays best (0/5 wins for the
extreme warm start in every family). The largest gains over cold retraining
are on Adam/AdamW `class_sorted` (+14 to +16 pp) and `single_late` (+12 to
+14 pp). Even so, `class_sorted` accuracy stays below 40%. RMSprop's loss
against cold retraining comes from `half_block` (−8.4) and
`alternating_halves` (−6.3). Its extreme-retuned config sits at the lowest
allowed lr (1e-4) and the highest α. That may limit what the warm start
offers, but this experiment does not test that explanation.

## Stretch: held-out ordering types (two seeds)

See [held_out/README.md](held_out/README.md). Each fold starts from its own retuned fixed config, chosen without the held-out type. Warm held-out policies beat that config for both seeds in all four families. All-type gains: SGD +1.99, Adam +9.33, AdamW +7.66, RMSprop +1.14 pp. Against the cold held-out policies, results are mixed and only two seeds were run: SGD +1.60 (1/2), Adam +2.26 (2/2), AdamW +0.91 (1/2), RMSprop −0.50 (1/2).

## Checks and caveats

- **Fresh vs committed controls** (`drift.csv`):
  - Extreme-retuned fixed: 634/640 episodes are bit-identical, max |Δacc| 0.0002.
  - Cold retrained: 3150/3200 identical, max 0.0104.
  - All fixed and cold aggregates match `extreme_retrained_five_seeds/RESULTS.md`
    to two decimals.
  - Regular warm-start policies drift more (2840/3200 identical, max per-episode
    0.103, SGD seed 341). Their aggregate means move by at most 0.03 pp from the
    committed `results/warm_start/*_extreme_evaluation.csv` (SGD: 41 → 40
    failures). This matches the unexplained drift documented in
    `results/warm_start/README.md`. All comparisons here use only the fresh rows.
- **Failures**: none for the new policies, fixed or cold. The regular SGD warm
  start has 40 failed test episodes, kept in its mean. Training failures for
  the new policies are 0–1 of 4000 episodes (`training_summary.csv`).
- **Selected checkpoints**: 15 of 20 policies were selected at the final update
  (500), and the others at 300–400. Longer training might change the numbers,
  but it was not tried.
- **Scope**: the ordering types are in-distribution (training and test share
  the eight types), with disjoint episode seeds. This is not a test of
  generalization to unseen orderings or a new test set. The MNIST test images
  were used in earlier exploratory work. There are five policy seeds and one
  network. SD and wins are descriptive, and no significance test is claimed.
- The comparison mixes two changes, the warm-start config and the training
  distribution. The regular warm start differs in both its config and its
  training data. The cold-retrained control isolates the initialization only.
- Seed disjointness of the training, development and test pools is asserted
  at launch, and three unit tests pass (`unit_tests.txt`).

## Commands

```bash
export MNIST_DATA_DIR=/path/to/data   # parent of MNIST/
python experiments/extreme_warm_start/test_extreme_warm.py
python experiments/extreme_warm_start/experiment.py launch --smoke          # 2-update pipeline check
python experiments/extreme_warm_start/experiment.py launch --workers 12     # train, evaluate, analyze (resumable)
python experiments/extreme_warm_start/tables.py                             # tables.md
```

Individual steps: `experiment.py train --family F --seed S`,
`experiment.py evaluate --family F --variant V --seed S` (`--seed -1` for
fixed), and `experiment.py analyze`. Every job runs one thread with
`OMP_NUM_THREADS=MKL_NUM_THREADS=1`.

**Compute**: CPU VM, 12 concurrent one-thread jobs on a shared 32-vCPU
machine. Training took 1700–2700 s per policy. The whole run, including 84
evaluation jobs, took 1 h 45 min wall clock (20:53–22:38 EST, 2026-11-21).

## Files

- `summary.csv`: mean, SD, min and max per family, scope (biased, all, each
  ordering) and variant, with failures.
- `comparisons.csv`: paired differences, SD, wins, and each seed's difference.
- `policy_seed_scores.csv`, `training_summary.csv`, `drift.csv`, `design.json`.
- `evaluation/`: per-episode test rows for every variant, plus
  compressed action traces for the new policies.
- `<Family>_gradient_<seed>/`: selected checkpoint (`best.pt`), development
  scores, and `done.json` (with the warm config).
