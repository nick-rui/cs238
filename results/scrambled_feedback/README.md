# Scrambled-feedback control

The paper's open-loop schedule control removes the policy entirely. This
control keeps the frozen policy running, but feeds it **wrong feedback**. That
tests whether the closed-loop gains depend on the policy observing *this
episode's* actual trajectory.

## Design

- **Policies and episodes.** All 20 frozen regular-suite policies (SGD,
  Adam, AdamW, RMSprop × seeds 338–342) are run on the original regular
  evaluation episodes. That is all ten orderings × episode seeds
  1900000–1900019, with the same checkpoints, rollout batches of four, and
  original thread counts (two threads for SGD 338/339, one otherwise).
- **What is scrambled.** At each of the 40 decisions, the five feedback
  inputs are replaced: relative control loss, recent loss improvement, log
  gradient norm, gradient–memory cosine, and gradient–previous-gradient
  cosine. Elapsed time and the policy's own current settings stay real. The
  chosen settings act on the real classifier. The engines are unchanged: a
  pass-through wrapper around the policy module substitutes those columns.
- **Donor traces.** Replacement values are what *the same policy* observed at
  the same decision of a different evaluation episode. They come from the
  real-feedback re-evaluation in [`results/clean_test`](../clean_test/README.md)
  (same environment), so they are realistic, on-policy feedback values, just
  not about the current episode. There are two fixed donor maps, built with
  NumPy seed 238 and shared by every policy ([`donors.csv`](donors.csv)):
  - **cross**: the donor has a different episode seed *and* a different
    ordering. It is a permutation of the 200 episodes, so each donor trace is
    used once.
  - **same_order**: the donor has the same ordering but a different episode
    seed: a derangement within each ordering. Burst timing therefore matches,
    but the initialization, digits and minibatches do not.
- **Comparators**, all evaluated fresh in the same environment on both test
  sets: real feedback, the open-loop mean schedules (seeds 338–339 only, as
  in `results/open_loop`) and the tuned fixed configuration.
- **Checks.** Feeding each episode its own recorded feedback through the
  wrapper reproduces real-feedback results exactly. With no donor, the
  wrapper reproduces the plain engine (`experiments/clean_test/run.py check`).
  The donor maps are verified to be permutations with the stated
  constraints. Every batch asserts that the captured final classifier
  reproduces the engine's own official accuracy.
- **Statistics.** These follow the paper: orderings are weighted equally,
  with mean ± sample SD over policy seeds and wins out of N seeds. 95% CIs come
  from [the extended paired bootstrap](../paired_bootstrap_extended/README.md),
  which resamples policy seeds and then episode seeds, 10,000 times, with
  seed 238. Failed episodes are retained.

## Commands

```bash
export MNIST_DATA_DIR=/path/to/data
.venv/bin/python experiments/clean_test/run.py launch --cpus 6 --only 'regular/(policy|fixed|open_loop)/.*'
.venv/bin/python experiments/scrambled_feedback/run.py donors
.venv/bin/python experiments/scrambled_feedback/run.py evaluate --smoke --donor cross --family Adam --seed 338
nohup .venv/bin/python experiments/scrambled_feedback/run.py launch --cpus 6 > results/scrambled_feedback/launch.log 2>&1 &
.venv/bin/python experiments/scrambled_feedback/run.py summarize
.venv/bin/python scripts/paired_bootstrap_extended.py --groups scrambled
```

## Results

Accuracy (%) on the nine biased regular orderings. Differences are in pp, as
the mean [95% CI] (wins / seeds).

**Official test, regular biased mean (nine orderings).** Arms: mean ± SD over policy seeds.

| Optimizer | Fixed | Open loop (338–339) | Real feedback | Scrambled, cross | Scrambled, same order |
|---|---:|---:|---:|---:|---:|
| SGD | 88.35 | 88.66 ± 0.19 | 89.61 ± 0.51 | 88.55 ± 0.87 | 89.57 ± 0.42 |
| Adam | 91.50 | 90.22 ± 0.52 | 90.31 ± 0.66 | 90.13 ± 0.60 | 90.31 ± 0.60 |
| AdamW | 91.50 | 90.08 ± 0.38 | 90.35 ± 0.80 | 90.04 ± 0.94 | 90.36 ± 0.81 |
| RMSprop | 92.37 | 92.53 ± 0.07 | 92.57 ± 0.07 | 92.48 ± 0.06 | 92.57 ± 0.08 |

| Optimizer | Real − cross | Real − same order | Cross − fixed | Same order − fixed | Cross − open loop (338–339) | Real − open loop (338–339) |
|---|---:|---:|---:|---:|---:|---:|
| SGD | +1.06 [+0.39, +2.14] (5/5) | +0.04 [-0.10, +0.19] (4/5) | +0.20 [-0.78, +0.86] (4/5) | +1.22 [+0.80, +1.65] (5/5) | +0.02 [-0.16, +0.22] (1/2) | +0.46 [+0.28, +0.65] (2/2) |
| Adam | +0.18 [-0.19, +0.72] (2/5) | -0.01 [-0.13, +0.08] (2/5) | -1.37 [-1.94, -0.81] (0/5) | -1.19 [-1.77, -0.65] (0/5) | -0.07 [-0.22, +0.01] (0/2) | +0.47 [-0.23, +1.22] (1/2) |
| AdamW | +0.31 [-0.13, +0.81] (2/5) | -0.02 [-0.11, +0.07] (1/5) | -1.46 [-2.22, -0.63] (1/5) | -1.14 [-1.82, -0.44] (1/5) | -0.07 [-0.19, +0.01] (0/2) | +0.43 [-0.23, +1.15] (1/2) |
| RMSprop | +0.10 [+0.04, +0.16] (5/5) | +0.00 [-0.02, +0.03] (3/5) | +0.11 [-0.01, +0.21] (5/5) | +0.20 [+0.09, +0.30] (5/5) | -0.02 [-0.05, +0.02] (0/2) | +0.04 [-0.04, +0.11] (1/2) |

**Clean test, regular biased mean (nine orderings).** Arms: mean ± SD over policy seeds.

| Optimizer | Fixed | Open loop (338–339) | Real feedback | Scrambled, cross | Scrambled, same order |
|---|---:|---:|---:|---:|---:|
| SGD | 88.16 | 88.45 ± 0.23 | 89.40 ± 0.52 | 88.33 ± 0.86 | 89.37 ± 0.45 |
| Adam | 91.49 | 90.09 ± 0.46 | 90.22 ± 0.67 | 90.03 ± 0.60 | 90.23 ± 0.60 |
| AdamW | 91.49 | 89.94 ± 0.33 | 90.26 ± 0.82 | 89.93 ± 0.95 | 90.27 ± 0.83 |
| RMSprop | 92.26 | 92.48 ± 0.08 | 92.53 ± 0.09 | 92.41 ± 0.06 | 92.52 ± 0.08 |

| Optimizer | Real − cross | Real − same order | Cross − fixed | Same order − fixed | Cross − open loop (338–339) | Real − open loop (338–339) |
|---|---:|---:|---:|---:|---:|---:|
| SGD | +1.07 [+0.39, +2.19] (5/5) | +0.03 [-0.11, +0.17] (4/5) | +0.17 [-0.83, +0.84] (4/5) | +1.21 [+0.77, +1.66] (5/5) | +0.03 [-0.15, +0.24] (1/2) | +0.46 [+0.29, +0.64] (2/2) |
| Adam | +0.19 [-0.18, +0.75] (2/5) | -0.01 [-0.13, +0.09] (2/5) | -1.46 [-2.05, -0.88] (0/5) | -1.26 [-1.86, -0.71] (0/5) | -0.08 [-0.25, +0.02] (0/2) | +0.49 [-0.22, +1.26] (1/2) |
| AdamW | +0.33 [-0.10, +0.83] (2/5) | -0.00 [-0.09, +0.08] (2/5) | -1.56 [-2.33, -0.71] (0/5) | -1.22 [-1.92, -0.49] (0/5) | -0.08 [-0.19, -0.00] (0/2) | +0.45 [-0.22, +1.18] (1/2) |
| RMSprop | +0.12 [+0.05, +0.20] (5/5) | +0.02 [-0.01, +0.05] (3/5) | +0.15 [+0.04, +0.26] (5/5) | +0.26 [+0.15, +0.37] (5/5) | -0.03 [-0.07, +0.01] (0/2) | +0.04 [-0.04, +0.12] (1/2) |


**Reading.**

- **SGD.** Feedback from a mismatched ordering removes nearly all of the
  closed-loop gain. Real minus cross is +1.06 pp [+0.39, +2.14], with all 5
  seeds positive. Cross-scrambled SGD is not distinguishable from fixed
  (+0.20 [−0.78, +0.86]) or from the open-loop schedule (+0.02, two seeds).
  Feedback from *another episode with the same ordering* keeps the gain: real
  minus same order is +0.04 [−0.10, +0.19], and same order minus fixed is
  +1.22 [+0.80, +1.65], 5/5. The SGD gain therefore depends on feedback
  that tracks the ordering's shift timing. It does not measurably depend on
  feedback being accurate for the specific episode (its initialization,
  digits and minibatches).
  Per-seed effects of cross scrambling vary widely (+0.41 to +2.72 pp).
  On the shuffled ordering alone, the cross-minus-real effect is small and
  mixed (+0.04 pp, 3/5).
- **Adam and AdamW.** Scrambling makes no detectable difference.
  Real minus cross is +0.18 [−0.19, +0.72] and +0.31 [−0.13, +0.81], with
  2/5 wins each; real minus same order is within ±0.02 pp. All policy arms
  stay about 1.2–1.5 pp below fixed. One seed (339) drives most of the
  cross effect for both families.
- **RMSprop.** The effect is small but consistent. Real minus cross is
  +0.10 [+0.04, +0.17], 5/5, and real minus same order is +0.00. Cross-scrambled
  RMSprop still beats fixed by +0.11 pp, 5/5. Its CI is [−0.01, +0.21] on the
  official test and [+0.04, +0.26] on the clean test. Most of RMSprop's small
  gain over fixed therefore does not need any feedback about the episode.
- **Clean test.** All conclusions are the same on the 6,000 clean images.
  Every difference moves by at most 0.10 pp; the real-vs-scrambled contrasts move by at most 0.04 pp. The only changes in a CI's
  relation to zero are two borderline cases: RMSprop cross minus fixed
  (above) and AdamW cross minus open loop (−0.07; CI upper bound +0.008 on
  official, −0.001 on clean).

**Diagnostics** ([`diagnostics.csv`](diagnostics.csv)). The mean absolute
difference between the fed and the realized feedback inputs (normalized
observation units) is 0.25–0.36 for cross donors and 0.14–0.17 for same-order
donors. Cross-scrambled SGD had 3 failed episodes out of 1,000 (real feedback:
0). They are retained at their recorded accuracy. No other arm failed.

## Caveats

- "Same order" donors share the ordering's event timing by construction.
  The result shows that matching timing is sufficient here. It does not show
  which features carry the timing signal (loss or gradient), and it does not
  test unseen timings.
- Cross donors provide realistic but inconsistent feedback; this is not the
  same as removing feedback. Its similarity to the open-loop schedule is
  measured only for the two original seeds.
- The donor maps are a single fixed draw. Paired per-seed SDs are reported,
  but donor-assignment variability is not resampled.
- Only the regular suite was run; no extreme-suite scrambling.

## Files

`arms.csv` (means and SDs per arm, test set and scope: biased, shuffled, all) ·
`contrasts.csv` (paired differences with per-seed values) ·
`policy_seed_accuracy.csv` · `diagnostics.csv` · `donors.csv` ·
`summary_table.md` · `units/<donor>/<family>_<seed>/` (per-episode
`evaluation.csv` with donor identity and feedback-error diagnostics, and
`done.json`).

## Runtime

40 units took 1.62 CPU-hours. Wall time was about 46 min: 30 adaptive-family
units ran on two threads alongside the clean-test SGD units, then 10 SGD units
ran on six threads.
