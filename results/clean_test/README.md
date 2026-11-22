# Clean held-out test split

The paper's 10,000 official MNIST test images were used during earlier
exploration. The seed-238 permutation of the 60,000 training images assigns
50,000 to classifier training, 2,000 to control feedback and 2,000 to
development; the remaining **6,000 images (permutation positions
54,000–59,999) are unused**. This experiment verifies that, then re-scores
every saved policy and fixed baseline on those 6,000 images. Nothing is
trained, tuned or selected; no test result feeds back into anything.

## 1. Audit: the 6,000 images are untouched

`experiments/clean_test/run.py audit` writes [`audit.json`](audit.json). It
passed (re-run at the end; 54 revisions scanned):

- **Split.** `torch.randperm(60000, seed 238)` is recomputed and equals
  `engine.ix`. The four slices (50,000 / 2,000 / 2,000 / 6,000) are disjoint
  and cover all 60,000 indices. `engine.x`, `engine.xc` and `engine.xd`
  equal the first three slices exactly. The stream generators index only
  `engine.x`: every per-class index used to build orderings is below 50,000.
- **Static scan of every committed revision.** Every pushed revision (`main`
  and all PR branches) plus this branch was searched for any
  access to the full training split (`X[`, `Y[`, `ix[`, `e.X`, `e.Y`,
  `e.ix`, `train=True`, `randperm(60000`). Only the five split-definition
  lines in `experiments/recovery_study/engine.py` and the CIFAR-10 timing
  harness's own split code (`scripts/time_cifar10_episode.py`, which uses
  the same three slices and reports timing only) match, plus a CIFAR-10-only
  CNN driver on a pushed branch (`experiments/cnn_cifar10/run.py`) that splits
  CIFAR-10's own arrays (allowed by path; any such line mentioning MNIST
  would fail). The split lines were
  introduced in the first commit and never changed. Training, control,
  development, tuning, test evaluation, ablations, open loop, warm start,
  retraining, held-out types, Fashion-MNIST and the reproduction checks
  all read `x`/`xc`/`xd` or the official test split.
- **Runtime poisoning.** `engine.X`, `engine.Y`, `engine.ix` and
  `engine.data` were replaced by objects that raise on any use. Then SGD and
  all three adaptive families ran a policy training update, development
  evaluation, test evaluation and fixed tuning/test rollouts, plus an
  extreme-stream rollout. None touched the full split.
- **Duplicates.** None of the 6,000 images is a pixel-exact duplicate of a
  training, control or development image, or of an official test image.
  The clean split's class counts are 568/689/592/650/582/521/589/629/557/623.

What the audit cannot cover: exploration that happened before the first
commit, outside the repository. The 6,000 images are also from MNIST's
*training* population, which shares writers with the classifier-training
images, whereas the official test split uses different writers. Accuracy
levels on the two splits are therefore not strictly comparable. In practice
they came out within about 0.4 pp, with the clean split slightly lower. The useful quantity is whether **paired
method differences** survive on images no-one has used.

## 2. Design

- **Same trained classifiers, two test sets.** Each unit re-runs the original
  evaluation episodes for one saved policy or fixed configuration: the
  same episode seeds (regular 1900000–1900019, extreme 1920000–1920019),
  orderings, checkpoints and rollout batches (four episodes per rollout;
  fixed SGD uses its original single-network `baseline`). Thread counts match
  the runs that produced the saved records: two for regular-suite fixed SGD,
  original SGD policies 338/339 and warm SGD, and one otherwise.
  The final classifier of every episode is scored on both the official test
  split and the 6,000 clean images. For engine rollouts the final parameters
  are captured by temporarily wrapping `engine.forward`; the engines are
  unchanged. Each batch asserts that the official accuracy recomputed from
  the captured parameters equals the engine's own value. This proves the
  clean score belongs to the same final classifier. Fixed SGD is run twice
  with identical training (once per test set). The clean score uses the
  engines' failure conventions: adaptive-family failures score 0% accuracy,
  as in the official metric.
- **Fresh official replay.** Comparisons of clean with official use the fresh
  official numbers from the same run, not the committed CSVs, because full
  re-evaluation drifts from some saved rows (documented in
  `results/warm_start/README.md`). Drift against the committed CSVs is reported
  per unit in [`replay_drift.csv`](replay_drift.csv).
- **Units (120).** Regular suite: fixed, frozen policies 338–342 (from
  `results/optimizer_families`, `results/recovery_study` and
  `results/additional_seeds`), warm-start policies 338–342 and open-loop
  schedules for 338–339. Extreme suite: saved fixed, extreme-retuned fixed,
  frozen policies 338–342, extreme-retrained policies 338–342 (two original
  seeds and the five-seed extension) and warm-start policies 338–342. All
  four optimizers. Not included: the held-out-ordering-type folds, the
  ablations and Fashion-MNIST.
- **Aggregation.** As in the paper: orderings weighted equally, averaged
  within a policy seed. Mean ± sample SD across policy seeds. Wins count
  policy seeds whose paired mean difference is positive. Failed episodes are
  retained.
- **Observation traces.** Regular frozen-policy units also save the
  observations each policy saw (`observations.npz`). These are the donor
  traces for the [scrambled-feedback control](../scrambled_feedback/README.md).

## Commands

```bash
export MNIST_DATA_DIR=/path/to/data   # parent of MNIST/
.venv/bin/python experiments/clean_test/run.py audit
.venv/bin/python experiments/clean_test/run.py check     # capture == engine; wrapper is a no-op
.venv/bin/python experiments/clean_test/run.py evaluate --smoke --suite regular --method policy --family Adam --seed 340
nohup .venv/bin/python experiments/clean_test/run.py launch --cpus 6 > results/clean_test/launch.log 2>&1 &
.venv/bin/python experiments/clean_test/run.py summarize
```

`launch` skips units that have a `done.json` file. Two-thread units take two
of the `--cpus` slots. One unit is `evaluate --suite S --method M --family F --seed N`.

## Results

Mean final accuracy over policy seeds, with the paper's aggregation. "Biased"
means the nine regular or seven extreme biased orderings; the extreme warm-start
row uses all eight extreme orderings, as in the paper. Each cell shows the fresh
official-test replay → the 6,000 clean images, for **the same trained
classifiers**. Differences are method minus comparator, paired by episode
(and by policy seed where both arms are seeded). The 95% CIs come from
[the extended bootstrap](../paired_bootstrap_extended/README.md). Wins count
policy seeds with a positive paired difference. Comparisons labelled
338–339 use two policy seeds.

| Comparison | Suite (orderings) | Optimizer | Comparator, official → clean (%) | Method mean ± SD, official → clean (%) | Difference, official [95% CI] (pp) | Difference, clean [95% CI] (pp) | Wins, official / clean |
|---|---|---|---:|---:|---:|---:|---:|
| frozen policy vs fixed | regular (biased) | SGD | 88.35 → 88.16 | 89.61 ± 0.51 → 89.40 ± 0.52 | +1.26 [+0.78, +1.76] | +1.24 [+0.75, +1.74] | 5/5 / 5/5 |
| frozen policy vs fixed | regular (biased) | Adam | 91.50 → 91.49 | 90.31 ± 0.66 → 90.22 ± 0.67 | -1.20 [-1.81, -0.63] | -1.27 [-1.90, -0.68] | 0/5 / 0/5 |
| frozen policy vs fixed | regular (biased) | AdamW | 91.50 → 91.49 | 90.35 ± 0.80 → 90.26 ± 0.82 | -1.16 [-1.83, -0.47] | -1.23 [-1.93, -0.51] | 0/5 / 0/5 |
| frozen policy vs fixed | regular (biased) | RMSprop | 92.37 → 92.26 | 92.57 ± 0.07 → 92.53 ± 0.09 | +0.20 [+0.10, +0.31] | +0.27 [+0.16, +0.39] | 5/5 / 5/5 |
| frozen policy vs fixed | extreme (biased) | SGD | 47.46 → 47.36 | 51.63 ± 1.23 → 51.43 ± 1.23 | +4.18 [+2.74, +5.52] | +4.07 [+2.62, +5.42] | 5/5 / 5/5 |
| frozen policy vs fixed | extreme (biased) | Adam | 63.35 → 63.27 | 46.05 ± 10.01 → 45.95 ± 9.96 | -17.30 [-25.47, -9.79] | -17.31 [-25.41, -9.84] | 0/5 / 0/5 |
| frozen policy vs fixed | extreme (biased) | AdamW | 63.35 → 63.27 | 49.07 ± 10.78 → 49.02 ± 10.73 | -14.27 [-23.34, -6.56] | -14.25 [-23.26, -6.58] | 0/5 / 0/5 |
| frozen policy vs fixed | extreme (biased) | RMSprop | 67.01 → 66.87 | 67.56 ± 0.91 → 67.44 ± 0.89 | +0.55 [-0.34, +1.23] | +0.57 [-0.30, +1.26] | 4/5 / 4/5 |
| retrained vs extreme-retuned fixed | extreme (biased) | SGD | 61.32 → 61.03 | 67.48 ± 1.86 → 67.06 ± 1.86 | +6.17 [+4.48, +7.76] | +6.03 [+4.35, +7.63] | 5/5 / 5/5 |
| retrained vs extreme-retuned fixed | extreme (biased) | Adam | 65.91 → 65.63 | 69.22 ± 2.81 → 68.85 ± 2.73 | +3.31 [+1.15, +5.72] | +3.22 [+1.13, +5.58] | 5/5 / 5/5 |
| retrained vs extreme-retuned fixed | extreme (biased) | AdamW | 65.91 → 65.63 | 69.13 ± 3.47 → 68.76 ± 3.39 | +3.22 [+0.46, +6.12] | +3.13 [+0.42, +5.98] | 4/5 / 4/5 |
| retrained vs extreme-retuned fixed | extreme (biased) | RMSprop | 70.22 → 69.92 | 72.98 ± 1.05 → 72.60 ± 1.02 | +2.75 [+1.80, +3.90] | +2.68 [+1.74, +3.81] | 5/5 / 5/5 |
| retrained vs frozen (seed-matched) | extreme (biased) | SGD | 51.63 → 51.43 | 67.48 ± 1.86 → 67.06 ± 1.86 | +15.85 [+14.53, +17.04] | +15.62 [+14.30, +16.83] | 5/5 / 5/5 |
| retrained vs frozen (seed-matched) | extreme (biased) | Adam | 46.05 → 45.95 | 69.22 ± 2.81 → 68.85 ± 2.73 | +23.17 [+17.07, +29.73] | +22.90 [+16.79, +29.46] | 5/5 / 5/5 |
| retrained vs frozen (seed-matched) | extreme (biased) | AdamW | 49.07 → 49.02 | 69.13 ± 3.47 → 68.76 ± 3.39 | +20.05 [+13.42, +26.93] | +19.75 [+13.09, +26.65] | 5/5 / 5/5 |
| retrained vs frozen (seed-matched) | extreme (biased) | RMSprop | 67.56 → 67.44 | 72.98 ± 1.05 → 72.60 ± 1.02 | +5.42 [+4.42, +6.40] | +5.16 [+4.23, +6.10] | 5/5 / 5/5 |
| warm vs fixed | regular (biased) | SGD | 88.35 → 88.16 | 90.86 ± 0.38 → 90.73 ± 0.38 | +2.52 [+2.03, +3.02] | +2.57 [+2.08, +3.08] | 5/5 / 5/5 |
| warm vs fixed | regular (biased) | Adam | 91.50 → 91.49 | 92.69 ± 0.26 → 92.67 ± 0.27 | +1.18 [+0.94, +1.44] | +1.18 [+0.91, +1.45] | 5/5 / 5/5 |
| warm vs fixed | regular (biased) | AdamW | 91.50 → 91.49 | 92.66 ± 0.29 → 92.64 ± 0.30 | +1.15 [+0.90, +1.44] | +1.15 [+0.87, +1.44] | 5/5 / 5/5 |
| warm vs fixed | regular (biased) | RMSprop | 92.37 → 92.26 | 92.63 ± 0.08 → 92.58 ± 0.08 | +0.26 [+0.18, +0.34] | +0.32 [+0.23, +0.41] | 5/5 / 5/5 |
| warm vs fixed | extreme (all 8) | SGD | 53.03 → 52.94 | 55.48 ± 3.41 → 55.29 ± 3.41 | +2.45 [-0.67, +4.95] | +2.35 [-0.77, +4.85] | 4/5 / 4/5 |
| warm vs fixed | extreme (all 8) | Adam | 66.96 → 66.89 | 69.81 ± 2.42 → 69.70 ± 2.41 | +2.85 [+0.93, +4.86] | +2.81 [+0.89, +4.82] | 5/5 / 5/5 |
| warm vs fixed | extreme (all 8) | AdamW | 66.96 → 66.89 | 70.33 ± 1.73 → 70.22 ± 1.73 | +3.37 [+1.95, +4.87] | +3.34 [+1.90, +4.85] | 5/5 / 5/5 |
| warm vs fixed | extreme (all 8) | RMSprop | 70.31 → 70.18 | 70.29 ± 2.03 → 70.18 ± 2.03 | -0.02 [-1.78, +1.45] | -0.00 [-1.76, +1.48] | 3/5 / 3/5 |
| frozen policy vs open loop (seeds 338-339) | regular (biased) | SGD | 88.66 → 88.45 | 89.12 ± 0.23 → 88.90 ± 0.26 | +0.46 [+0.28, +0.65] | +0.46 [+0.29, +0.64] | 2/2 / 2/2 |
| frozen policy vs open loop (seeds 338-339) | regular (biased) | Adam | 90.22 → 90.09 | 90.69 ± 0.38 → 90.58 ± 0.45 | +0.47 [-0.23, +1.22] | +0.49 [-0.22, +1.26] | 1/2 / 1/2 |
| frozen policy vs open loop (seeds 338-339) | regular (biased) | AdamW | 90.08 → 89.94 | 90.51 ± 0.48 → 90.39 ± 0.54 | +0.43 [-0.23, +1.15] | +0.45 [-0.22, +1.18] | 1/2 / 1/2 |
| frozen policy vs open loop (seeds 338-339) | regular (biased) | RMSprop | 92.53 → 92.48 | 92.57 ± 0.13 → 92.52 ± 0.16 | +0.04 [-0.04, +0.11] | +0.04 [-0.04, +0.12] | 1/2 / 1/2 |

**What moved.** Every sign, every win count, and every bootstrap CI's
relation to zero (for the biased and all-order aggregates above) is unchanged
between the official and clean test sets, with one exception in win counts: frozen
AdamW's regular *shuffled* gain over fixed went from 2/5 to 3/5 seed wins (a
+0.03 → +0.12 pp difference; see `summary.csv`). The headline differences move
by at most 0.33 pp. The largest moves are SGD retrained vs saved fixed (+20.03
→ +19.70) and retrained vs frozen (−0.23 to −0.30 pp across families). Regular-suite
biased differences move by at most 0.13 pp. Mean accuracies on the clean split
range from 0.43 pp lower to 0.04 pp higher than official; most are lower. RMSprop's frozen
extreme-biased CI still includes zero ([−0.30, +1.26]), as does SGD's warm
extreme all-order CI ([−0.77, +4.85]).

**Replay drift against the committed records.** The fresh official numbers
are not bit-identical to every saved row. Across 21,280 re-evaluated episodes,
19,075 match the committed accuracy exactly. The largest single-episode
difference is 10.3 pp (extreme suite), and no unit's mean moves by more than
0.10 pp. One failure flag differs (extreme warm SGD 340). This is the same
pre-existing drift documented in `results/warm_start/README.md`. Fresh rows
reproduce Table I's five-seed means to within 0.1 pp (e.g. SGD extreme fixed
47.46 vs 47.55 saved). All clean-vs-official comparisons above use the fresh
pair. See [`replay_drift.csv`](replay_drift.csv).

**Failures.** 53 of 21,280 episodes failed, all on the extreme suite: frozen
Adam (12), frozen SGD (1), and warm-start SGD (40 across five seeds). They are
retained at the engines' recorded accuracy on both test sets, with the
adaptive-family failures at 0%.

## Files

- [`summary.csv`](summary.csv): every comparison with official and clean
  means, SDs, differences and wins, and the clean-minus-official shift;
  [`summary_table.md`](summary_table.md) is its rendered table.
- [`policy_seed_differences.csv`](policy_seed_differences.csv): per policy seed.
- [`per_order.csv`](per_order.csv): per unit and ordering.
- [`replay_drift.csv`](replay_drift.csv): fresh official replay vs committed rows.
- `units/<suite>/<method>/<family>[_<seed>]/`: per-episode `evaluation.csv`
  (official and clean metrics, plus the recomputed official accuracy) and
  `done.json` (runtime, threads, checkpoint, drift); `observations.npz` for
  regular frozen policies.
- [`audit.json`](audit.json), [`checks.json`](checks.json).

## Caveats

- This is a new *image* sample, not a new task: same orderings, episode seeds,
  checkpoints and architecture. It removes test-image reuse, not the
  possibility that design choices were tuned to MNIST or to these orderings.
- The clean images come from MNIST's training writers (see the audit).
- At 6,000 images, the clean split's per-episode accuracy noise is about 1.3×
  the official split's. The bootstrap resamples episodes and policy seeds, not
  test images.
- The held-out-ordering-type folds, the ablations and Fashion-MNIST were not
  re-scored.

## Runtime

120 units took 4.64 CPU-hours (sum of per-unit wall times). That includes the
second pass that fixed SGD needs for the clean set. They ran on a 32-vCPU
machine shared with three other experiment sessions, at no more than six
concurrent threads. Running time was about 1.4 h. A launcher deadlock, now
fixed, idled the run for another ~75 min: two-thread jobs could each hold one
CPU slot while waiting for a second.
