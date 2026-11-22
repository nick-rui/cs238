# Regular-suite open-loop control

Eight frozen schedules, 1,600 paired replay episodes, 64,000 control-step trace
rows. Each policy's schedule is the arithmetic mean of its 200 saved regular
evaluation trajectories, pooled across conditions. The same schedule runs on
every condition without policy calls or feedback-dependent hyperparameters.
Existing evaluations/checkpoints/traces are unchanged. See the
[runner documentation](../../../experiments/open_loop/README.md) for commands
and the retrospective-control limitations.

## Mean final MNIST accuracy (%)

Both policy seeds, 20 episode seeds per condition; all 10 conditions are equally
weighted. Delta is closed-loop minus open-loop, in percentage points.

| Optimizer | Tuned fixed | Closed-loop | Open-loop | Delta (pp) |
|---|---:|---:|---:|---:|
| SGD | 88.7541 | 89.4385 | 89.0115 | +0.4269 |
| Adam | 91.6169 | 90.8717 | 90.4486 | +0.4231 |
| AdamW | 91.6169 | 90.7029 | 90.3119 | +0.3910 |
| RMSprop | 92.4524 | 92.6653 | 92.6220 | +0.0433 |

Nine biased conditions only:

| Optimizer | Tuned fixed | Closed-loop | Open-loop | Delta (pp) |
|---|---:|---:|---:|---:|
| SGD | 88.3638 | 89.1264 | 88.6613 | +0.4651 |
| Adam | 91.5040 | 90.6890 | 90.2216 | +0.4674 |
| AdamW | 91.5040 | 90.5145 | 90.0838 | +0.4307 |
| RMSprop | 92.3703 | 92.5664 | 92.5325 | +0.0340 |

## Seed sensitivity

All 10 conditions, final accuracy (%). These are two policy-training replicates,
not hundreds of independent policy-training seeds.

| Optimizer | Policy seed | Closed-loop | Open-loop | Delta (pp) |
|---|---:|---:|---:|---:|
| SGD | 338 | 89.6183 | 89.1629 | +0.4555 |
| SGD | 339 | 89.2587 | 88.8602 | +0.3984 |
| Adam | 338 | 90.6130 | 90.7646 | -0.1516 |
| Adam | 339 | 91.1304 | 90.1325 | +0.9979 |
| AdamW | 338 | 90.3829 | 90.5412 | -0.1583 |
| AdamW | 339 | 91.0229 | 90.0826 | +0.9403 |
| RMSprop | 338 | 92.5695 | 92.5793 | -0.0097 |
| RMSprop | 339 | 92.7611 | 92.6648 | +0.0963 |

## Interpretation

Closed-loop improves mean accuracy over this particular pooled mean schedule,
most clearly for SGD and policy seed 339 of Adam/AdamW. It does not consistently
improve each saved policy: seed 338 of Adam, AdamW, and RMSprop favors replay.
RMSprop's aggregate gap is small, and its mean trajectory log loss slightly
favors open-loop (-0.859381 vs -0.855285; lower is better). Neither adaptive nor
open-loop Adam/AdamW beats the existing tuned fixed baseline on average.

This is limited evidence for trajectory-specific adaptation, not proof that
feedback causes all observed policy-versus-fixed gains. A time-varying open-loop
schedule retains some gains, averaging nonlinear settings changes their
distribution, schedules use the same evaluation trajectories retrospectively,
and only two policy seeds are available. No significance claim is made.

## Validation and artifacts

- `checks.json` records source-mean checks, duplicate/missing-source rejection,
  exact historical settings replay for all eight policies, a policy object that
  raises if called, and verification of every stored replay setting.
- `closed_loop_reproduction.json` is the existing regular reproduction check:
  fixed plus both closed-loop seeds for all four families (12 checks).
- All open-loop and saved closed-loop episodes retained; zero failed episodes.
- `paired_differences.csv`, `summary.csv`, `policy_seed_summary.csv`, and
  `condition_summary.csv` contain full-precision accuracy and loss comparisons.
- Each policy directory includes its schedule, final evaluation, compressed
  trace, and input/checkpoint hashes plus runtime metadata.

An optional broader check (`scripts/check_reproduction.py --suite extreme`)
fails on the existing RMSprop fixed representative (83.89% replay vs 84.50%
saved). The unmodified main-branch engine produces exactly the same row as the
modified engine for that case; it is not a schedule-path regression. No extreme
results were regenerated or modified for this regular-suite control.
