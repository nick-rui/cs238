# Paired bootstrap CIs for the later experiments

`scripts/paired_bootstrap.py` covers the earlier comparisons. This adds
intervals for the four later experiments the paper lists as uncovered: the
start-held ablations, five-seed extreme retraining, warm start, and held-out
ordering types. It also covers the two new evaluation-only experiments,
[clean test](../clean_test/README.md) and
[scrambled feedback](../scrambled_feedback/README.md).

## Method (unchanged)

`scripts/paired_bootstrap_extended.py` imports `paired` and `bootstrap` from
`scripts/paired_bootstrap.py` unchanged. Pairing is strict: by ordering and
episode seed, and by policy seed when both arms are seeded. A missing,
duplicate or nonfinite pair raises an error. Fixed arms broadcast across policy
seeds. Each row draws 10,000 percentile resamples with NumPy
`default_rng(238)`: first policy seeds, then episode seeds, sharing episode
draws across selected seeds and across orderings with the same episode-seed
grid. Orderings get equal weight. Failed episodes keep their stored accuracy.
`p_le_0` is the fraction of resampled means ≤ 0, not a calibrated p-value.

Experiment-specific choices:

- **Start-held ablations.** Arms are `results/ablations_start_values/`, the
  full policy and fixed configuration from `results/optimizer_families/`,
  and the tuned-held ablation from `results/ablations/`. All are saved
  official-test rows from seeds 338–339. The headline is all ten orderings, as
  in Table III.
- **Five-seed retraining.** Retrained seeds 338–339 come from
  `results/extreme_retrained/` and 340–342 from
  `results/extreme_retrained_five_seeds/`. Comparators are the extreme-retuned
  fixed configuration, the saved fixed configuration, and seed-matched frozen
  policies (`results/extreme_orderings/` and
  `results/additional_seeds/extreme/`). The headline is the seven biased
  orderings.
- **Warm start.** Arms come from `results/warm_start/*_{regular,extreme}_evaluation.csv`,
  whose fixed and full rows were re-evaluated with the warm policies. Warm vs
  fixed uses five seeds; warm vs full uses the matched seeds 338–339. The
  headline is regular biased, and all eight extreme orderings as in the paper.
- **Held-out types.** Each ordering type is scored only by the fold that held
  it out (`results/extreme_held_out/<family>/<type>/evaluation.csv`, rows
  with `order == <type>`). Fold policies with the same seed label share a
  policy-seed draw. The headline is all eight types; the seven-biased mean is
  also listed.
- **Clean test and scrambled feedback.** Rows are freshly re-evaluated in
  `results/clean_test/units/` and `results/scrambled_feedback/units/`. Each
  comparison is run separately on the fresh official-test accuracy and on the
  clean-split accuracy (`test` column).

```bash
.venv/bin/python scripts/paired_bootstrap_extended.py            # all groups whose inputs exist
.venv/bin/python scripts/paired_bootstrap_extended.py --groups warm held_out
cd scripts && ../.venv/bin/python -m unittest test_paired_bootstrap_extended test_paired_bootstrap
```

The tests check that every mean difference reproduces the experiment's own
committed summary. That covers retraining `comparisons.csv`, warm-start
`summary.csv` and held-out `paired_differences.csv` (to 1e-9 pp), and
start-held `summary.md` (to its rounding). They also check cross-fold pairing
and reproducibility.

## Headline intervals (official test, saved results)

### Start-held ablations (Table III; all ten orderings)

| Comparison | Suite | Optimizer | Condition | Seeds | A − B, pp [95% CI] | Seeds beating | Episode win rate | Failed A/B |
|---|---|---|---|---:|---:|---:|---:|---:|
| start_lr_only_vs_full_policy | regular | SGD | all | 2 | +0.078 [-0.320, +0.406] | 1/2 | 0.500 | 0/0 |
| start_lr_only_vs_fixed | regular | SGD | all | 2 | +0.762 [+0.470, +1.055] | 2/2 | 0.636 | 0/0 |
| start_lr_only_vs_tuned_held_lr_only | regular | SGD | all | 2 | +0.664 [-0.234, +1.966] | 1/2 | 0.491 | 0/1 |
| start_fixed_memory_vs_full_policy | regular | SGD | all | 2 | -0.605 [-0.953, -0.282] | 0/2 | 0.251 | 0/0 |
| start_fixed_memory_vs_fixed | regular | SGD | all | 2 | +0.079 [-0.200, +0.348] | 2/2 | 0.485 | 0/0 |
| start_fixed_memory_vs_tuned_held_fixed_memory | regular | SGD | all | 2 | +0.157 [-1.143, +1.355] | 1/2 | 0.355 | 0/3 |
| start_lr_only_vs_full_policy | regular | Adam | all | 2 | -1.055 [-1.440, -0.664] | 0/2 | 0.429 | 0/0 |
| start_lr_only_vs_fixed | regular | Adam | all | 2 | -1.800 [-2.156, -1.442] | 0/2 | 0.333 | 0/0 |
| start_lr_only_vs_tuned_held_lr_only | regular | Adam | all | 2 | -2.942 [-3.330, -2.557] | 0/2 | 0.075 | 0/0 |
| start_fixed_memory_vs_full_policy | regular | Adam | all | 2 | -0.878 [-1.182, -0.519] | 0/2 | 0.430 | 0/0 |
| start_fixed_memory_vs_fixed | regular | Adam | all | 2 | -1.624 [-2.053, -1.260] | 0/2 | 0.265 | 0/0 |
| start_fixed_memory_vs_tuned_held_fixed_memory | regular | Adam | all | 2 | -2.685 [-3.007, -2.319] | 0/2 | 0.034 | 0/0 |

### Five-seed extreme retraining (seven biased extreme orderings)

| Comparison | Suite | Optimizer | Condition | Seeds | A − B, pp [95% CI] | Seeds beating | Episode win rate | Failed A/B |
|---|---|---|---|---:|---:|---:|---:|---:|
| retrained_vs_retuned_fixed | extreme | SGD | biased_mean | 5 | +6.167 [+4.477, +7.759] | 5/5 | 0.711 | 0/0 |
| retrained_vs_saved_fixed | extreme | SGD | biased_mean | 5 | +19.929 [+18.049, +21.789] | 5/5 | 0.919 | 0/0 |
| retrained_vs_frozen | extreme | SGD | biased_mean | 5 | +15.823 [+14.498, +17.019] | 5/5 | 0.851 | 0/1 |
| retrained_vs_retuned_fixed | extreme | Adam | biased_mean | 5 | +3.310 [+1.149, +5.718] | 5/5 | 0.653 | 0/0 |
| retrained_vs_saved_fixed | extreme | Adam | biased_mean | 5 | +5.855 [+3.720, +8.261] | 5/5 | 0.644 | 0/0 |
| retrained_vs_frozen | extreme | Adam | biased_mean | 5 | +23.164 [+17.051, +29.737] | 5/5 | 0.929 | 0/12 |
| retrained_vs_retuned_fixed | extreme | AdamW | biased_mean | 5 | +3.219 [+0.462, +6.122] | 4/5 | 0.637 | 0/0 |
| retrained_vs_saved_fixed | extreme | AdamW | biased_mean | 5 | +5.765 [+3.035, +8.651] | 5/5 | 0.639 | 0/0 |
| retrained_vs_frozen | extreme | AdamW | biased_mean | 5 | +20.088 [+13.484, +26.954] | 5/5 | 0.904 | 0/0 |
| retrained_vs_retuned_fixed | extreme | RMSprop | biased_mean | 5 | +2.753 [+1.797, +3.902] | 5/5 | 0.638 | 0/0 |
| retrained_vs_saved_fixed | extreme | RMSprop | biased_mean | 5 | +5.977 [+5.059, +7.010] | 5/5 | 0.652 | 0/0 |
| retrained_vs_frozen | extreme | RMSprop | biased_mean | 5 | +5.413 [+4.406, +6.416] | 5/5 | 0.629 | 0/0 |

### Warm start (regular: nine biased; extreme: all eight, as in the paper)

| Comparison | Suite | Optimizer | Condition | Seeds | A − B, pp [95% CI] | Seeds beating | Episode win rate | Failed A/B |
|---|---|---|---|---:|---:|---:|---:|---:|
| warm_vs_fixed | regular | SGD | biased_mean | 5 | +2.514 [+2.027, +3.021] | 5/5 | 0.922 | 0/0 |
| warm_vs_full | regular | SGD | biased_mean | 2 | +1.774 [+0.931, +2.596] | 2/2 | 0.847 | 0/0 |
| warm_vs_fixed | extreme | SGD | all | 5 | +2.319 [-0.790, +4.815] | 4/5 | 0.649 | 41/0 |
| warm_vs_full | extreme | SGD | all | 2 | -2.542 [-6.990, +1.474] | 1/2 | 0.419 | 26/0 |
| warm_vs_fixed | regular | Adam | biased_mean | 5 | +1.183 [+0.938, +1.439] | 5/5 | 0.955 | 0/0 |
| warm_vs_full | regular | Adam | biased_mean | 2 | +1.863 [+1.330, +2.415] | 2/2 | 0.992 | 0/0 |
| warm_vs_fixed | extreme | Adam | all | 5 | +2.849 [+0.941, +4.846] | 5/5 | 0.759 | 0/0 |
| warm_vs_full | extreme | Adam | all | 2 | +11.936 [+9.103, +14.589] | 2/2 | 0.838 | 0/0 |
| warm_vs_fixed | regular | AdamW | biased_mean | 5 | +1.152 [+0.899, +1.434] | 5/5 | 0.950 | 0/0 |
| warm_vs_full | regular | AdamW | biased_mean | 2 | +2.034 [+1.513, +2.573] | 2/2 | 0.997 | 0/0 |
| warm_vs_fixed | extreme | AdamW | all | 5 | +3.340 [+1.906, +4.845] | 5/5 | 0.814 | 0/0 |
| warm_vs_full | extreme | AdamW | all | 2 | +10.666 [+9.223, +12.021] | 2/2 | 0.827 | 0/0 |
| warm_vs_fixed | regular | RMSprop | biased_mean | 5 | +0.262 [+0.179, +0.344] | 5/5 | 0.766 | 0/0 |
| warm_vs_full | regular | RMSprop | biased_mean | 2 | +0.054 [-0.036, +0.151] | 2/2 | 0.538 | 0/0 |
| warm_vs_fixed | extreme | RMSprop | all | 5 | -0.025 [-1.768, +1.443] | 3/5 | 0.597 | 0/0 |
| warm_vs_full | extreme | RMSprop | all | 2 | -0.111 [-0.646, +0.528] | 1/2 | 0.489 | 0/0 |

### Held-out ordering types (each type scored by its own fold; all eight types, then seven biased)

| Comparison | Suite | Optimizer | Condition | Seeds | A − B, pp [95% CI] | Seeds beating | Episode win rate | Failed A/B |
|---|---|---|---|---:|---:|---:|---:|---:|
| retrained_vs_fold_retuned_fixed | extreme_held_out | SGD | all | 2 | +0.390 [-2.666, +3.574] | 1/2 | 0.731 | 0/0 |
| retrained_vs_fold_retuned_fixed | extreme_held_out | SGD | biased_mean | 2 | +0.229 [-3.066, +3.673] | 1/2 | 0.718 | 0/0 |
| retrained_vs_frozen_policy | extreme_held_out | SGD | all | 2 | +8.461 [+5.963, +10.774] | 2/2 | 0.752 | 0/0 |
| retrained_vs_frozen_policy | extreme_held_out | SGD | biased_mean | 2 | +9.941 [+7.344, +12.326] | 2/2 | 0.832 | 0/0 |
| retrained_vs_frozen_fixed | extreme_held_out | SGD | all | 2 | +12.004 [+8.689, +15.406] | 2/2 | 0.797 | 0/0 |
| retrained_vs_frozen_fixed | extreme_held_out | SGD | biased_mean | 2 | +13.952 [+10.366, +17.639] | 2/2 | 0.886 | 0/0 |
| retrained_vs_fold_retuned_fixed | extreme_held_out | Adam | all | 2 | +7.062 [+5.203, +8.781] | 2/2 | 0.666 | 0/0 |
| retrained_vs_fold_retuned_fixed | extreme_held_out | Adam | biased_mean | 2 | +8.082 [+5.903, +10.098] | 2/2 | 0.696 | 0/0 |
| retrained_vs_frozen_policy | extreme_held_out | Adam | all | 2 | +16.298 [+14.308, +18.277] | 2/2 | 0.797 | 0/0 |
| retrained_vs_frozen_policy | extreme_held_out | Adam | biased_mean | 2 | +18.986 [+16.642, +21.323] | 2/2 | 0.911 | 0/0 |
| retrained_vs_frozen_fixed | extreme_held_out | Adam | all | 2 | +8.280 [+6.494, +9.997] | 2/2 | 0.581 | 0/0 |
| retrained_vs_frozen_fixed | extreme_held_out | Adam | biased_mean | 2 | +9.776 [+7.684, +11.791] | 2/2 | 0.664 | 0/0 |
| retrained_vs_fold_retuned_fixed | extreme_held_out | AdamW | all | 2 | +6.755 [+5.522, +7.991] | 2/2 | 0.709 | 0/0 |
| retrained_vs_fold_retuned_fixed | extreme_held_out | AdamW | biased_mean | 2 | +7.708 [+6.272, +9.151] | 2/2 | 0.714 | 0/0 |
| retrained_vs_frozen_policy | extreme_held_out | AdamW | all | 2 | +14.873 [+14.118, +15.622] | 2/2 | 0.758 | 0/0 |
| retrained_vs_frozen_policy | extreme_held_out | AdamW | biased_mean | 2 | +17.319 [+16.454, +18.204] | 2/2 | 0.866 | 0/0 |
| retrained_vs_frozen_fixed | extreme_held_out | AdamW | all | 2 | +8.225 [+7.041, +9.436] | 2/2 | 0.594 | 0/0 |
| retrained_vs_frozen_fixed | extreme_held_out | AdamW | biased_mean | 2 | +9.690 [+8.312, +11.106] | 2/2 | 0.679 | 0/0 |
| retrained_vs_fold_retuned_fixed | extreme_held_out | RMSprop | all | 2 | +1.645 [+1.004, +2.233] | 2/2 | 0.600 | 0/0 |
| retrained_vs_fold_retuned_fixed | extreme_held_out | RMSprop | biased_mean | 2 | +1.921 [+1.210, +2.572] | 2/2 | 0.646 | 0/0 |
| retrained_vs_frozen_policy | extreme_held_out | RMSprop | all | 2 | +2.427 [+1.786, +3.023] | 2/2 | 0.509 | 0/0 |
| retrained_vs_frozen_policy | extreme_held_out | RMSprop | biased_mean | 2 | +3.243 [+2.559, +3.893] | 2/2 | 0.582 | 0/0 |
| retrained_vs_frozen_fixed | extreme_held_out | RMSprop | all | 2 | +3.453 [+2.837, +4.033] | 2/2 | 0.550 | 0/0 |
| retrained_vs_frozen_fixed | extreme_held_out | RMSprop | biased_mean | 2 | +4.385 [+3.699, +5.027] | 2/2 | 0.629 | 0/0 |

Reading, with no claims beyond these intervals:

- **Start-held ablations.** For SGD, LR-only is indistinguishable from full
  control (+0.08 [−0.32, +0.41]), while holding memory is worse
  (−0.61 [−0.95, −0.28]). For Adam, both restricted variants are below full
  control: −1.06 [−1.44, −0.66] and −0.88 [−1.18, −0.52]. Only two policy
  seeds are available, so the seed level of the resampling is very coarse.
- **Five-seed retraining.** All four families' intervals over the
  extreme-retuned fixed configuration exclude zero: SGD +6.17 [+4.48, +7.76],
  Adam +3.31 [+1.15, +5.72], AdamW +3.22 [+0.46, +6.12] and RMSprop
  +2.75 [+1.80, +3.90]. AdamW's lower bound is close to zero, consistent with
  its one losing seed.
- **Warm start.** All four regular-biased gains over fixed exclude zero
  (+2.51, +1.18, +1.15, +0.26). On all eight extreme orderings, the SGD
  (+2.32 [−0.79, +4.82]) and RMSprop (−0.03 [−1.77, +1.44]) intervals include
  zero; Adam and AdamW exclude it. Warm vs full SGD on extreme, −2.54
  [−6.99, +1.47], is a two-seed interval that includes zero.
- **Held-out types.** SGD's gain over fold-retuned fixed, +0.39
  [−2.67, +3.57], includes zero. Adam (+7.06), AdamW (+6.76) and RMSprop
  (+1.64) exclude zero. These intervals use two seeds, and resample only the
  20 episode seeds within each of the eight fixed types. They are not
  uncertainty over new ordering types. The paper's ± values are episode-level
  standard errors; these intervals also resample policy seeds and are wider.

Clean-test and scrambled-feedback intervals, on both test sets, are tabulated
in their own READMEs and in [`summary_table.md`](summary_table.md).

## Caveats

These are the same caveats as `results/paired_bootstrap/`. The CIs are
marginal, with no multiplicity adjustment, and conditional on saved checkpoints
and tuning. Two-seed intervals cannot robustly characterize training-seed
uncertainty. Fixed-vs-fixed rows are not included.

## Files

`bootstrap.csv` holds every comparison × condition (each single ordering,
`biased_mean`, `all`) with `n_policy_seeds`, `n_episodes`, `mean_diff_pp`,
the CI, `p_le_0`, `episode_win_rate`, `seeds_beating` and failure counts.
`headline` marks the row quoted for that comparison. `inputs.json` records
each input file's SHA-256 and the HEAD commit. `summary_table.md` lists every
headline row.
