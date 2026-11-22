# Stretch: held-out ordering types with warm starts (two seeds)

This repeats the leave-one-type-out protocol of `results/extreme_held_out` with warm-started policies for seeds 338 and 339.

- **Training** calls `held_out.train` unchanged. In each fold:
  - Training uses the seven seen types, with training seeds `7000000 + 10000·seed + 8·update + episode`.
  - Checkpoint selection uses development seeds 1940000–1940015, with the held-out type removed.
- **Initialization**: each policy starts from **its fold's retuned fixed config** (`results/extreme_held_out/<family>/<held>/fixed_retuned.json`). That config was selected without the held-out type, so the held-out type never informs training, selection or initialization. The extreme-retuned config used in the main experiment saw all eight types, so it would have leaked the held-out type.
- **Test**: the held-out type only, seeds 1920000–1920019. It is paired against fresh replays of the fold-retuned fixed config and the cold held-out policies (same seeds).
- **Fresh replay vs committed rows**: 1891/1920 episodes are bit-identical, with max |Δacc| 0.0012 (`drift.json`). There were no failed test episodes, and training failures were 0–4 of 4000 episodes per policy.
- **Compute**: 64 policies plus 32 fold evaluations, 3 h 16 min wall clock at 12 workers (22:40–01:56 EST). Mean training time was 1924 s per policy, max 2584 s.

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

**Summary.** With only two seeds, this is descriptive.
- The warm held-out policy beats the fold-retuned fixed config for both seeds in every family. On the all-type average the gains are +2.0 pp (SGD), +9.3 (Adam), +7.7 (AdamW) and +1.1 (RMSprop).
- Against the cold held-out policy, the result is mixed:
  - Adam: +2.3 pp, 2/2 wins.
  - SGD: +1.6 pp on average, but seed 339 loses (−1.6).
  - AdamW: +0.9 pp, 1/2.
  - RMSprop: −0.5 pp, 1/2.
- `single_late` and SGD `staircase` remain worse than fixed when held out.

Commands:

```bash
python experiments/extreme_warm_start/held_out_warm.py launch --smoke --families SGD Adam   # pipeline check
python experiments/extreme_warm_start/held_out_warm.py launch --workers 12                  # train, evaluate, analyze
```

Files under `held_out/`:
- `summary.csv` and `comparisons.csv`, by scope (all, biased, each held-out type), with per-seed differences.
- `drift.json`.
- `<family>/<held_out>/evaluation.csv`.
- `<family>/<held_out>/<seed>/` with `best.pt`, `development.csv` and `training_summary.json` (including the warm config).
