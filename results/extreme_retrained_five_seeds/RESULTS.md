# Extreme-ordering retraining: five policy seeds

The aggregate improvement over the extreme-retuned fixed baseline holds for
all four optimizer families. AdamW seed 341 is the only loss on the biased
average against that baseline (64.50% vs 65.91%). Retrained controllers beat
the original fixed and the seed-matched frozen controllers for every policy
seed on the biased average.

## Seven biased orderings

Mean test accuracy (%), with sample SD (percentage points) and min–max across
policy seeds. The fixed columns have no policy-seed variation.

| Optimizer | Saved fixed | Extreme-retuned fixed | Frozen mean ± SD [min–max] | Retrained mean ± SD [min–max] |
|---|---:|---:|---:|---:|
| SGD | 47.55 | 61.32 | 51.66 ± 1.23 [50.10–52.68] | 67.48 ± 1.86 [65.27–68.97] |
| Adam | 63.36 | 65.91 | 46.05 ± 10.02 [33.05–54.38] | 69.22 ± 2.81 [66.68–73.54] |
| AdamW | 63.36 | 65.91 | 49.04 ± 10.77 [32.95–57.82] | 69.13 ± 3.47 [64.50–74.21] |
| RMSprop | 67.00 | 70.22 | 67.56 ± 0.92 [65.97–68.16] | 72.98 ± 1.05 [72.12–74.67] |

Paired retrained-minus-control accuracy (mean ± sample SD, percentage points;
wins out of five policy seeds in parentheses). A win requires a strictly
positive average over the same orderings and 20 test seeds, with frozen
policies additionally matched by policy seed.

| Optimizer | vs saved fixed | vs extreme-retuned fixed | vs frozen |
|---|---:|---:|---:|
| SGD | +19.93 ± 1.86 (5/5) | +6.17 ± 1.86 (5/5) | +15.82 ± 0.64 (5/5) |
| Adam | +5.86 ± 2.81 (5/5) | +3.31 ± 2.81 (5/5) | +23.16 ± 8.19 (5/5) |
| AdamW | +5.76 ± 3.47 (5/5) | +3.22 ± 3.47 (4/5) | +20.09 ± 8.64 (5/5) |
| RMSprop | +5.98 ± 1.05 (5/5) | +2.75 ± 1.05 (5/5) | +5.41 ± 1.13 (5/5) |

The wins are not uniform across orderings: every family gains on
`half_block` and `alternating_halves` but loses on average on `single_late`
and `staircase` against the extreme-retuned fixed baseline.

## All eight orderings and shuffled reference

| Optimizer | All-order retrained mean ± SD (%) | All-order wins vs saved / retuned / frozen | Shuffled delta vs frozen (pp) | Shuffled wins vs frozen |
|---|---:|---:|---:|---:|
| SGD | 70.57 ± 1.73 | 5/5 / 5/5 / 5/5 | −0.41 | 2/5 |
| Adam | 71.99 ± 2.45 | 5/5 / 5/5 / 5/5 | −1.42 | 0/5 |
| AdamW | 71.90 ± 3.05 | 5/5 / 4/5 / 5/5 | −1.45 | 0/5 |
| RMSprop | 75.26 ± 0.91 | 5/5 / 5/5 / 5/5 | −2.40 | 0/5 |

## Design and checks

- Seeds 340–342 add 12 policies trained for 500 updates × 8 episodes, with
  training episode seed `6000000 + 10000 * policy_seed + 8 * update + episode`.
  Development (1930000–1930015) selects checkpoints; test (1920000–1920019)
  is separate. The launcher checks the seed sets for overlap.
- Retrained seeds 338–339, both fixed controls, and frozen seeds 338–339 are
  reused from the original experiments. Frozen seeds 340–342 come from the
  [additional-seed extreme evaluations](https://github.com/nick-rui/cs238/tree/4e24b070613967eaf1b8e47d93ac25c20c560116/results/additional_seeds/extreme).
  New evaluation CSVs and traces only contain the three new retrained runs
  per optimizer; [the cached rows](frozen_added_seeds.csv) are frozen controls.
- Failed test episodes are included at their recorded accuracy. There were
  no retrained or fixed test failures; frozen SGD had one, frozen Adam had twelve.
- This measures adaptation to the same ordering types with new episodes,
  not generalization to unseen orderings or a new MNIST test set. The five
  policy-seed pairs are descriptive, not 100 independent test pairs.
- Regular reproduction passed all 12 comparisons. Extreme reproduction still
  fails on the original RMSprop fixed row, with absolute accuracy error
  0.0061000586 as on unchanged main; no tolerances or saved rows were changed.

[Summary by scope and method](summary.csv) · [Paired mean, SD, range and wins](comparisons.csv) ·
[Scores by policy seed](policy_seed_scores.csv) · [Paired test-seed wins within policy seed](policy_seed_differences.csv) ·
[Seed and source manifest](design.json)
