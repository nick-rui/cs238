# Warm-start run record

Twenty policies were trained using the existing training loops without changing
the episode seeds, checkpoints, update count, or development-selection rule.
Training used eight concurrent CPU jobs at most, each with one torch thread.
Evaluation used eight concurrent jobs, each with one torch thread, four episodes
per policy rollout batch (fixed SGD uses its original single-episode baseline).

Training seconds per job, excluding import overhead:

| Family | Minimum | Maximum | Sum (five jobs) |
|---|---:|---:|---:|
| SGD | 2446.0 | 2998.9 | 13182.3 |
| Adam | 3198.2 | 4024.1 | 18196.1 |
| AdamW | 2698.2 | 3349.4 | 14951.0 |
| RMSprop | 1859.3 | 2552.7 | 10178.5 |

Paired evaluation job wall times (seconds): SGD regular/extreme 624/509;
Adam 854/702; AdamW 847/705; RMSprop 818/668.

`summary.csv` reports final accuracy (fractions), policy-seed mean paired
differences, and policy-seed win counts. `seed_differences.csv` retains each
policy seed's paired difference. The extreme group includes all eight orders,
including mixed; regular groups separate mixed from the nine biased orders.
The warm mean uses all five seeds; warm-minus-full uses only warm/full seeds
338 and 339. Thus that paired difference is not generally the difference between
the displayed warm and full means. No significance claim is made.

`*_sanity.json` compares update-zero warm policies with the unclamped tuned
fixed configs on the same sixteen development episodes. Small discrepancies
are expected from finite inverse-tanh logits and weight-decay clamping. AdamW
matches exactly here; SGD's score difference is about 0.0155.

## Verification and drift

- Initialization unit tests: three passed (`unit_tests.log`).
- Regular reproduction: all twelve representative checks passed, including
  action traces (`reproduction_regular.log`).
- Extreme reproduction: fails on fixed RMSprop, accuracy difference
  0.006100058555603027. The identical assertion occurs on unchanged main
  (`reproduction_extreme.log`, `reproduction_extreme_main.log`). No tolerance
  was changed.
- Full re-evaluation reveals additional drift not covered by that representative
  reproduction check. Each `*_drift.csv` reports maximum absolute errors against
  the saved CSVs, separately for fixed and each original full policy. In
  particular, SGD extreme fixed/full-338 maximum accuracy errors are 0.0980 and
  0.0907. Regular SGD fixed/full-338 errors reach 0.0214 and 0.0189.
- `drift_probe_main.json` replays the worst accuracy cases for SGD and Adam
  fixed/full-338 on unchanged main, using the original trace-enabled path and
  original rollout batch size. At one thread all eight probed main accuracies
  equal the warm branch's re-evaluations exactly. Two-thread probes do not
  restore the saved numbers (one extreme SGD fixed result changes). This rules
  out these warm-engine edits and simple one-vs-two-thread mismatch as the cause
  for those probed cases; the origin of the saved-result drift remains unknown.
  Other drift cases were measured but not individually replayed on main.

The summary uses only re-evaluated, paired controls. Existing results were never
overwritten. Small best-policy checkpoints are retained, matching existing
repository conventions; intermediate checkpoints and training logs/CSVs follow
the existing ignore rules. Verification logs are explicitly retained.
