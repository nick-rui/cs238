# Table and figure number audit

All line numbers below refer to the preparatory digest
`/home/ubuntu/final_report_digest.md` (Step 1), not to the LaTeX file.
The relevant source lines are quoted verbatim below so this audit remains
useful without that machine-local file. Pinned source commits are in README.md.
No table cell is calculated by subtracting already-rounded means.

## Table I: all 36 numerical cells

Columns are regular shuffled fixed, policy mean ± sample SD; regular biased
fixed, policy mean ± SD; frozen extreme biased fixed, policy mean ± SD.
All numbers are accuracy percentages.

| Row | Shuffled fixed / mean / SD | Regular biased fixed / mean / SD | Extreme biased fixed / mean / SD | Digest lines |
|---|---|---|---|---|
| SGD | 92.27 / 92.48 / 0.42 | 88.36 / 89.61 / 0.51 | 47.55 / 51.66 / 1.23 | 768, 772 |
| Adam | 92.63 / 92.72 / 0.23 | 91.50 / 90.31 / 0.66 | 63.36 / 46.05 / 10.02 | 769, 773 |
| AdamW | 92.63 / 92.67 / 0.38 | 91.50 / 90.34 / 0.80 | 63.36 / 49.04 / 10.77 | 770, 774 |
| RMSprop | 93.19 / 93.56 / 0.12 | 92.37 / 92.57 / 0.07 | 67.00 / 67.56 / 0.92 | 771, 775 |

Repository source: PR #4,
`results/additional_seeds/RESULTS.md`, five-policy-seed results table
at commit `4e24b070613967eaf1b8e47d93ac25c20c560116`.
Underlying detailed records: `summary.csv` and `policy_seed_summary.csv`
in the same directory.

Caption numerals: five policy seeds 338–342 (digest lines 759–760);
20 paired episode seeds (test-seed ranges at digest line 64, with the explicit
20-seed evaluation protocol at line 41); nine regular and seven extreme
biased types (lines 762–763). SD is sample SD across episode-averaged policy seeds,
not a CI.

## Table II: all 24 numerical cells

Each entry is mean delta / lower bound / upper bound, in percentage points.
Bounds are the reported marginal 95% hierarchical paired bootstrap CI.

| Row | Regular biased delta / lower / upper | Frozen extreme biased delta / lower / upper | Digest lines |
|---|---|---|---|
| SGD | 1.246 / 0.770 / 1.752 | 4.106 / 2.679 / 5.421 | 1201, 1213 |
| Adam | -1.199 / -1.810 / -0.625 | -17.309 / -25.454 / -9.810 | 1204, 1216 |
| AdamW | -1.161 / -1.847 / -0.471 | -14.323 / -23.379 / -6.619 | 1207, 1219 |
| RMSprop | 0.202 / 0.096 / 0.306 | 0.563 / -0.320 / 1.242 | 1210, 1222 |

Repository source: PR #6, `results/paired_bootstrap/README.md`,
Key results table, at `80fb3ad7d2e38bce16918720a3843a83eb16cb05`.
Underlying exact intervals: `results/paired_bootstrap/bootstrap.csv`,
comparison `policy_vs_fixed`, condition `biased_mean`.
The caption's five seeds and 95% interval level are documented by digest
lines 1138, 1141, and 1178–1183. These are not the later #9/#10/#11 treatments.

## Table III: all 8 numerical cells

All numbers are accuracy percentages over all ten regular orders.
No confidence intervals are reported for this table.

| Variant | SGD | Digest line | Adam | Digest line |
|---|---|---|---|---|
| Full policy | 89.44 | 1294 | 90.87 | 1297 |
| LR only, start-held | 89.52 | 1295 | 89.82 | 1298 |
| LR + WD, start-held memory | 88.83 | 1296 | 89.99 | 1299 |
| No gradient features | 89.50 | 614 | 89.70 | 619 |

Start-held source: PR #7, `results/ablations_start_values/summary.md`,
at `601913154463fb61ba11760df706164b1002fc78`.
No-gradient-feature source: PR #3, `results/ablations/summary.md`,
at `05e17f4e40c4e3af8e3d62ce163342f80c9ac48c`.
The latter removes only gradient norm and the two cosine features, not
loss feedback. Two policy seeds and 20 episode seeds per ordering are
specified in the source note at digest line 1301.
The three removed gradient features are enumerated in the PR #3 description
and the method at digest line 57.

## Figure 2: all 20 bar values (pp)

Optimizer order is SGD, Adam, AdamW, RMSprop.

| Series | Bar values | Digest source lines | Repository source |
|---|---|---|---|
| Frozen regular biased | 1.246, -1.199, -1.161, 0.202 | 1201, 1204, 1207, 1210 | #6 `results/paired_bootstrap/README.md` |
| Warm regular biased | 2.51, 1.18, 1.15, 0.26 | 1945, 1948, 1951, 1954 | #10 PR summary, independently in `results/warm_start/summary.csv` (fraction × 100, rounded to two decimals) |
| Extreme retrained | 6.17, 3.31, 3.22, 2.75 | 1824–1827 | #9 PR summary and `results/extreme_retrained_five_seeds/RESULTS.md` |
| Held-out types | 0.39, 7.06, 6.76, 1.64 | 2117–2120 | #11 `results/extreme_held_out/README.md` |
| Fashion-MNIST | 1.65, 0.39, -0.26, 0.21 | 1345–1348 | #8 PR summary and `results/fashion_mnist/optimizer_families/aggregates.csv` |

Warm and frozen use the regular nine-biased mean; extreme retrained uses
seven biased types versus extreme-retuned fixed; held-out uses all eight
types versus fold-retuned fixed; Fashion uses all ten regular types.
Series do not share one test distribution or comparator. No CI/error bar
is implied by the plot. In particular Fashion AdamW's rounded displayed
means differ by -0.25, but its stored unrounded paired delta rounds to -0.26.

Figure 1 is the unchanged original `reports/project_update/momentum_block.pdf`.
Its caption (20 episode seeds × two policy seeds, median/IQR, middle 50%)
comes from `reports/project_update/main.tex` and digest line 51.

## Quoted digest source rows

```text
768: | Regular | SGD | 92.48 ± 0.42 | 92.27 | 3 | 89.61 ± 0.51 | 88.36 | 5 |
769: | Regular | Adam | 92.72 ± 0.23 | 92.63 | 4 | 90.31 ± 0.66 | 91.50 | 0 |
770: | Regular | AdamW | 92.67 ± 0.38 | 92.63 | 2 | 90.34 ± 0.80 | 91.50 | 0 |
771: | Regular | RMSprop | 93.56 ± 0.12 | 93.19 | 5 | 92.57 ± 0.07 | 92.37 | 5 |
772: | Extreme | SGD | 92.55 ± 0.49 | 92.03 | 4 | 51.66 ± 1.23 | 47.55 | 5 |
773: | Extreme | Adam | 92.81 ± 0.27 | 92.26 | 5 | 46.05 ± 10.02 | 63.36 | 0 |
774: | Extreme | AdamW | 92.76 ± 0.39 | 92.26 | 5 | 49.04 ± 10.77 | 63.36 | 0 |
775: | Extreme | RMSprop | 93.64 ± 0.14 | 93.37 | 5 | 67.56 ± 0.92 | 67.00 | 4 |
1201: | policy_vs_fixed | regular | SGD | +1.246 [+0.770, +1.752] | +0.214 [-0.224, +0.671] |
1204: | policy_vs_fixed | regular | Adam | -1.199 [-1.810, -0.625] | +0.088 [-0.320, +0.520] |
1207: | policy_vs_fixed | regular | AdamW | -1.161 [-1.847, -0.471] | +0.039 [-0.426, +0.534] |
1210: | policy_vs_fixed | regular | RMSprop | +0.202 [+0.096, +0.306] | +0.367 [+0.200, +0.523] |
1213: | policy_vs_fixed | extreme | SGD | +4.106 [+2.679, +5.421] | +0.519 [-0.013, +1.067] |
1216: | policy_vs_fixed | extreme | Adam | -17.309 [-25.454, -9.810] | +0.548 [+0.162, +0.945] |
1219: | policy_vs_fixed | extreme | AdamW | -14.323 [-23.379, -6.619] | +0.491 [+0.061, +0.949] |
1222: | policy_vs_fixed | extreme | RMSprop | +0.563 [-0.320, +1.242] | +0.272 [+0.109, +0.421] |
1294: | SGD | gradient | 0.3450 | 89.44 | -0.3714 | 0/400 | +0.0000 | +0.00 | +0.0000 | — | — | — |
1295: | SGD | lr_only | 0.3471 | 89.52 | -0.3475 | 0/400 | +0.0021 | +0.08 | +0.0239 | -0.0200 | +0.66 | +0.0093 |
1296: | SGD | fixed_memory | 0.3648 | 88.83 | -0.3336 | 0/400 | +0.0198 | -0.60 | +0.0378 | -0.0041 | +0.16 | +0.0097 |
1297: | Adam | gradient | 0.3027 | 90.87 | -0.5880 | 0/400 | +0.0000 | +0.00 | +0.0000 | — | — | — |
1298: | Adam | lr_only | 0.3487 | 89.82 | -0.4482 | 0/400 | +0.0459 | -1.05 | +0.1398 | +0.1112 | -2.94 | +0.2296 |
1299: | Adam | fixed_memory | 0.3378 | 89.99 | -0.5113 | 0/400 | +0.0351 | -0.88 | +0.0766 | +0.0977 | -2.69 | +0.1928 |
614: | SGD | no_feedback | 0.3432 | 89.50 | -0.3661 | 0/400 | -0.0018 | +0.06 | +0.0053 |
619: | Adam | no_feedback | 0.3454 | 89.70 | -0.4950 | 0/400 | +0.0426 | -1.17 | +0.0930 |
1345: | SGD | 73.89 | 75.54 | **+1.65** | 2/2 | +0.68 |
1346: | Adam | 77.33 | 77.72 | +0.39 | 2/2 | −0.75 |
1347: | AdamW | 77.81 | 77.56 | −0.26 | 0/2 | −0.91 |
1348: | RMSprop | 78.30 | 78.51 | +0.21 | 2/2 | +0.21 |
1824: | SGD | 67.48 ± 1.86 | +6.17 pp | 5/5 · 5/5 · 5/5 |
1825: | Adam | 69.22 ± 2.81 | +3.31 pp | 5/5 · 5/5 · 5/5 |
1826: | AdamW | 69.13 ± 3.47 | +3.22 pp | 5/5 · 4/5 · 5/5 |
1827: | RMSprop | 72.98 ± 1.05 | +2.75 pp | 5/5 · 5/5 · 5/5 |
1945: | SGD | Regular biased | 88.35 | 89.13 | 90.86 | +2.51 (5) | +1.77 (2) |
1948: | Adam | Regular biased | 91.51 | 90.69 | 92.69 | +1.18 (5) | +1.86 (2) |
1951: | AdamW | Regular biased | 91.51 | 90.51 | 92.66 | +1.15 (5) | +2.03 (2) |
1954: | RMSprop | Regular biased | 92.37 | 92.57 | 92.63 | +0.26 (5) | +0.05 (2) |
2117: | SGD | 65.15% | 56.69% | 53.15% | 64.76% | +0.39 ± 0.37 |
2118: | Adam | 75.26% | 58.96% | 66.98% | 68.19% | +7.06 ± 0.33 |
2119: | AdamW | 75.20% | 60.33% | 66.98% | 68.45% | +6.76 ± 0.33 |
2120: | RMSprop | 73.75% | 71.32% | 70.30% | 72.11% | +1.64 ± 0.27 |

```

## Sources for numbers added after the merge of PRs #14–#19

Numbers added in the "new results" revision come from these committed files. All are on `main`.

| Paper location | Source |
|---|---|
| Table II, clean-test column; clean-test sentence | `results/clean_test/README.md` |
| Table III, dev-mean schedule and tuned cosine rows; extreme schedule comparison (+3.6, +4.0) | `results/prospective_schedules/README.md` (regular and extreme tables) |
| Table III, scrambled-feedback row; same-order +0.04 | `results/scrambled_feedback/README.md` |
| Reactive rule never firing | `results/reactive_rule/README.md` |
| Ablation and warm-start CIs; five-seed retraining CIs; held-out SGD interval | `results/paired_bootstrap_extended/README.md` |
| Warm extreme policy vs extreme-retuned fixed (seven orderings) | `results/extreme_warm_start/README.md` |
| Table IV left half; warm/cold vs extreme cosine (seven orderings); cosine vs random | `results/extreme_baselines/README.md` (method means; P1, P2, E1) |
| Table IV right half; generated-suite differences, win shares, severity quintiles, replication | `results/extreme_generated/README.md` (method means, paired differences, severity, replication) |
| Warm-started held-out types (+1.99, +9.33, +7.66, +1.14) | `results/extreme_warm_start/README.md` (held-out section) |
| CIFAR-10 differences, LR-floor share | `results/cifar10/README.md` |
| SGD engine-replay offset (0.42, 2.38) | `results/prospective_schedules/README.md` (caveats) |
