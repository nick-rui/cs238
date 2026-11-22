# Generalization to held-out ordering types

Each of the eight extreme orderings is held out in turn, including the shuffled
reference (`mixed`). Each fold starts SGD, Adam, AdamW and RMSprop policies from
scratch for seeds 338 and 339. It trains for 500 updates with eight episodes per
update: four pairs of episodes matched by ordering. Over a seven-update cycle,
each seen type appears equally often. The held-out type is excluded from policy
training, checkpoint selection and fixed-config selection.

Checkpoint selection and tuning use two independent development episodes per
seen type. The fixed search uses the same 144 SGD and 288 adaptive-optimizer
candidates as the earlier extreme retraining study, including the original
fixed config in each grid. Candidate rollouts can be reused across folds; the
fold's selection score uses only the 14 seen-type episodes. The selection
metric is mean final log loss plus mean log trajectory loss on the development
images, not test accuracy.

The original frozen policies and fixed config are re-evaluated in the same CPU
environment as the new methods, on the same 20 episode seeds for each held-out
type. Each paired episode uses the same classifier initialization and ordered
examples; every ordering rearranges the same class-balanced bag. The official
MNIST test split is reused, not a new dataset. These are held-out **ordering
types**, not a test of generalization to unrelated shifts: some types are
structurally similar (for example, `single_max` and `single_late`).

## Results

Mean final test accuracy, weighting each held-out type equally:

| Optimizer | Retrained policy | Frozen policy | Frozen fixed | Fold-retuned fixed | Retrained minus fold-retuned fixed (pp) |
|---|---:|---:|---:|---:|---:|
| SGD | 65.15% | 56.69% | 53.15% | 64.76% | +0.39 ± 0.37 |
| Adam | 75.26% | 58.96% | 66.98% | 68.19% | +7.06 ± 0.33 |
| AdamW | 75.20% | 60.33% | 66.98% | 68.45% | +6.76 ± 0.33 |
| RMSprop | 73.75% | 71.32% | 70.30% | 72.11% | +1.64 ± 0.27 |

The deltas report mean ± one paired standard error, calculated before rounding.
On biased types alone, the gains over the fold-retuned fixed configs were
+0.23, +8.08, +7.71 and +1.92 points, respectively. Adam and AdamW beat their
retuned fixed configs for all 20 episode seeds after averaging over held-out
types; RMSprop did for 19. SGD beat its retuned fixed config for 11 seeds in
the all-type average, and its two retrained policy seeds averaged 62.56% and
67.74%, bracketing the 64.76% fixed result.

Transfer was not uniform: all four retrained policies beat every comparator
on `pair_max`, `half_block`, `class_sorted` and `alternating_halves`, but none
beat the best comparator on `mixed`, `single_late` or `staircase`. In particular,
the Adam and AdamW policy gains on `class_sorted` are large relative to the
baselines, but both policies still finish below 36% accuracy. No final test
episode failed.

## Files

- [Design and seeds](design.json)
- [Accuracy and failure counts by optimizer and holdout](summary.csv)
- [Separate results for each policy seed](policy_seed_summary.csv)
- [Paired accuracy deltas, wins and standard errors](paired_differences.csv)
- Under each optimizer/holdout: full paired episode results, fixed-grid scores
  and selected config, and both selected checkpoints with development scores.

The paired unit is the episode seed. For each seed, average over the included
ordering types and (for policies) the two training seeds before subtracting the
comparator. The standard errors are conditional on these orderings, policy
seeds and test images; they do not measure optimizer-to-optimizer variation or
uncertainty across new ordering types. `biased` excludes only `mixed`.
