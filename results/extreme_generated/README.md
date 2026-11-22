# Procedurally generated extreme orderings

The paper's extreme suite has seven hand-designed biased orderings (plus
shuffled), so its conclusions rest on seven orderings whose per-ordering results
vary sharply. This experiment re-evaluates the same frozen methods on **200
procedurally generated biased orderings** and treats the ordering as the unit
of resampling. No policy is trained and no configuration is tuned.

The design (generator, sampling distribution, seeds, severity measure, nearness
rule, analysis plan) was committed in `d3dcd01` ("Prespecify procedurally
generated extreme orderings") and pushed before any episode of the generated
suite ran. Everything below follows that plan. Results labeled *exploratory*
were not prespecified.

## Design

**Generator** (`experiments/extreme_orderings/generated.py`). An ordering is a spec:

| Parameter | Distribution |
|---|---|
| `k` classes per biased block | uniform on {1, 2, 3, 5} |
| `n` blocks | uniform on {1..10} |
| class mode (n > 1) | uniform on {repeat (same k digits every block), change (next k digits of a cyclic walk through the episode's digit permutation), alternate (two disjoint k-digit sets alternate)}. For k = 5, change and alternate are the same stream. |
| `R` shuffled recovery after the last block | 0 with probability 0.2, else uniform on {1..100} |
| `L` block length (batches) | uniform on {max(2, ⌈Lmax/2⌉)..Lmax}, where Lmax is the longest block that keeps every digit within its 1,280 draws and fits in 200 − R batches |
| `G` gap between blocks (mixed batches) | 0 for n = 1; repeat: uniform on {1..Gmax}; change/alternate: 0 with probability 0.5, else uniform on {1..Gmax} |
| start of the first block | derived: 200 − R − (nL + (n−1)G) |

Streams keep the exact invariants of `run.stream`:
- the same seeded balanced bag: 1,280 draws per digit, 12,800 examples;
- digits permuted per episode;
- block labels spread across each block with `np.resize`;
- the rest of the bag shuffled into the non-block positions;
- a within-batch shuffle, giving 200 batches of 64.

The random-number calls happen in the same order as in `run.stream`. So the seven original orderings, written as specs (`ORIGINAL_SPECS`), reproduce the original streams **exactly** (tested). Under a fixed episode seed, every method gets the same multiset, classifier initialization and probe set.

**Seeds.**
- Generator seeds: 8100000–8100199, one ordering each, `gen000`–`gen199`.
- Episode seeds: five per ordering, distinct across orderings, `1986000 + 5i + j` (1986000–1986999).
- `check_design()` verifies both ranges are disjoint from every earlier dev, test and training seed pool, and from the stream RNG offsets.

**Severity.** Primary: `mean_kl`, the mean over the 200 batches of KL(expected batch class frequencies ‖ uniform), computed from the spec. Block batches have k equal classes. Non-block batches have the expected composition of the remaining bag, which is depleted of block digits. Secondary: `biased_batches` = nL.

The generated family is **milder on average than the seven originals**:

| | Generated (200) | Seven originals |
|---|---|---|
| Median mean KL | 0.42 (range 0.13–2.07) | 0.69 |
| Mean biased batches | 64 | 106 |

Per-original values: single_max and single_late 0.33, pair_max 0.50, half_block and alternating_halves 0.69, staircase 1.84, class_sorted 2.30 (`original_specs.csv`).

**Nearness to the seven originals.**
- Distance to each original: the mean over batches of the total-variation distance between expected class-frequency profiles (`orderings.csv`: `nearest`, `nearest_distance`).
- Near-duplicate rule (prespecified): same k, n and mode, with L, start and R each within 10 batches. Only **2/200** qualify (gen000 ≈ single_late, gen198 ≈ single_max).
- Secondary exclusion: profile distance ≤ 0.15, which catches **11/200**.
- Results are reported for all 200, without the 2, and without the 11. The conclusions do not change.

**Methods** (all four optimizers, all re-run fresh here: one torch thread, official MNIST test set, failed episodes kept):

| Name | Method | Source |
|---|---|---|
| `fixed_regular` | fixed configuration tuned on regular data | `results/optimizer_families/*_baseline.json` |
| `fixed_extreme` | fixed configuration retuned on extreme data | `results/extreme_retrained/*_baseline.json` |
| `cosine_regular` | cosine LR schedule tuned on regular data | PR #14 `results/prospective_schedules/selected` |
| `policy_frozen` | regular-trained policies, seeds 338–342 | recovery_study / optimizer_families, additional_seeds |
| `policy_retrained` | cold extreme-retrained policies, seeds 338–342 | extreme_retrained (338–339), extreme_retrained_five_seeds (340–342) |
| `warm_extreme` | extreme warm-start policies, seeds 338–342 | PR #15 `results/extreme_warm_start`, `warm=` from the extreme-retuned config |

Each ordering's five episodes are one rollout. As in every saved SGD fixed row, SGD fixed configurations run through `torch.optim.SGD`; everything else uses the batched engines.

**Analysis** (`experiments/extreme_orderings/analyze_generated.py`):
- **Mean accuracy:** the mean over orderings of the per-ordering mean over episode seeds and policy seeds, with the policy-seed SD and range.
- **Pairing:** policy − policy differences are seed-matched; a policy minus an unseeded comparator broadcasts the comparator.
- **Bootstrap:** 10,000 resamples with NumPy `default_rng(238)`. Each resample draws orderings with replacement, then the five episode seeds within each drawn ordering, then the five policy seeds. 95% percentile CIs; 90% CIs are in `comparisons.csv`.
- **Win rates:** the share of orderings where A > B (per-ordering mean difference), and policy seeds with a positive mean difference out of 5.
- **Severity:** quintile bins of `mean_kl` (with CIs) and OLS slopes of the per-ordering difference on `mean_kl` and on `biased_batches`, with CIs from the same bootstrap.
- **Replication:** the same code on a fresh replay of the seven originals (seeds 1920000–1920019) in this environment.

## Results

### Mean final test accuracy, all 200 generated orderings (%)

± policy-seed SD [min–max]; failed episodes in parentheses (of 1,000 per method, or 5,000 for policies).

| Method | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| Fixed (regular-tuned) | 68.70 | 82.21 | 82.21 | 85.64 |
| Fixed (extreme-retuned) | 76.99 | 80.10 | 80.10 | 82.83 |
| Cosine (regular-tuned) | 76.53 (7) | **89.01** | **89.01** | **89.60** |
| Frozen regular policy | 75.35 ± 0.90 [73.78–76.07] (16) | 74.93 ± 7.39 [66.07–81.30] (36) | 77.68 ± 6.14 [67.22–81.94] (2) | 86.46 ± 0.82 [85.67–87.42] |
| Cold extreme policy | 81.04 ± 2.24 [78.13–83.07] | 82.42 ± 0.78 [81.73–83.63] | 82.39 ± 1.25 [80.93–83.90] | 85.34 ± 0.50 [84.83–86.07] |
| Extreme warm-start policy | **83.16 ± 0.39** [82.70–83.66] | 85.76 ± 0.59 [85.32–86.74] | 85.77 ± 0.54 [85.12–86.41] | 85.52 ± 2.25 [82.83–87.82] |

The regular fixed and cosine configurations have weight decay 0, so Adam and AdamW coincide for those rows.

### Paired differences, all 200 orderings (pp)

Mean [95% two-level bootstrap CI]; share of orderings with A > B; policy seeds with a positive mean difference (of 5).

| A − B | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| Warm − fixed (regular) | +14.46 [+12.03, +17.04]; 88%; 5/5 | +3.55 [+1.80, +5.43]; 38%; 5/5 | +3.56 [+1.82, +5.47]; 38%; 5/5 | −0.12 [−1.97, +1.80]; 21%; 3/5 |
| Warm − fixed (extreme) | +6.17 [+4.69, +7.72]; 88%; 5/5 | +5.65 [+4.52, +6.84]; 94%; 5/5 | +5.66 [+4.56, +6.88]; 94%; 5/5 | +2.69 [+0.94, +4.50]; 99%; 4/5 |
| Warm − cosine (regular) | +6.63 [+4.64, +8.71]; 64%; 5/5 | **−3.25 [−4.28, −2.20]; 14%; 0/5** | **−3.25 [−4.29, −2.22]; 14%; 0/5** | **−4.08 [−6.13, −2.18]; 11%; 0/5** |
| Warm − frozen | +7.81 [+5.77, +10.03]; 71%; 5/5 | +10.83 [+4.58, +17.81]; 72%; 5/5 | +8.08 [+3.77, +13.69]; 70%; 5/5 | −0.94 [−3.28, +1.41]; 19%; 2/5 |
| Warm − cold | +2.12 [+0.63, +3.95]; 70%; 5/5 | +3.33 [+2.37, +4.31]; 97%; 5/5 | +3.38 [+2.46, +4.36]; 98%; 5/5 | +0.18 [−1.88, +2.22]; 76%; 3/5 |
| Cold − fixed (regular) | +12.34 [+9.58, +15.22]; 86%; 5/5 | +0.22 [−1.43, +1.91]; 23%; 3/5 | +0.18 [−1.70, +2.06]; 22%; 2/5 | −0.30 [−1.45, +0.94]; 22%; 1/5 |
| Cold − fixed (extreme) | +4.04 [+1.86, +6.19]; 82%; 5/5 | +2.32 [+1.18, +3.44]; 81%; 5/5 | +2.28 [+0.87, +3.66]; 76%; 5/5 | +2.51 [+1.62, +3.51]; 86%; 5/5 |
| Cold − cosine (regular) | +4.51 [+1.58, +7.19]; 60%; 5/5 | −6.59 [−7.94, −5.29]; 7%; 0/5 | −6.62 [−8.19, −5.15]; 6%; 0/5 | −4.26 [−5.19, −3.36]; 10%; 0/5 |
| Cold − frozen | +5.69 [+3.29, +8.13]; 62%; 5/5 | +7.50 [+1.66, +14.11]; 58%; 5/5 | +4.70 [+0.51, +9.79]; 52%; 4/5 | −1.12 [−2.13, +0.05]; 19%; 0/5 |
| Frozen − fixed (regular) | +6.64 [+5.39, +7.97]; 96%; 5/5 | −7.28 [−13.86, −1.67]; 26%; 0/5 | −4.52 [−10.24, −0.60]; 28%; 0/5 | +0.82 [+0.12, +1.66]; 72%; 5/5 |
| Frozen − cosine (regular) | −1.18 [−3.82, +1.43]; 51%; 0/5 | −14.08 [−20.78, −8.17]; 2%; 0/5 | −11.33 [−17.23, −6.91]; 4%; 0/5 | −3.14 [−4.58, −1.83]; 18%; 0/5 |
| Fixed (extreme) − fixed (regular) | +8.29 [+6.56, +10.16]; 70% | −2.10 [−3.00, −1.14]; 16% | −2.10 [−3.00, −1.14]; 16% | −2.81 [−3.28, −2.29]; 8% |

Policy-seed SDs are in [`tables.md`](tables.md).

Excluding near-duplicates barely matters:
- Without the 2 near-duplicates (N = 198), every difference moves by at most 0.38 pp.
- Without the 11 loose-near orderings (N = 189), every difference moves by at most 1.3 pp; the largest shifts are the SGD comparisons against cosine.
- The only change in a significance call: RMSprop cold − frozen, whose CI (upper end +0.05 on all 200) excludes 0 in both subsets (−1.23 and −1.36 pp).

See `tables.md`.

### Severity

The prespecified bins are in [`severity_bins.csv`](severity_bins.csv) and the figure; the regressions are in [`severity_regression.csv`](severity_regression.csv).

![difference vs severity](figures/difference_vs_severity.png)

- **Warm − extreme-retuned fixed** is positive in every severity quintile for all four optimizers, roughly flat for SGD and RMSprop, and larger in the top quintile for Adam/AdamW (+8.0 vs about +5). Slope vs mean KL: Adam +7.5 pp/nat [+0.2, +13.2], AdamW +7.8 [−0.0, +13.9], SGD +3.7 [−5.5, +11.9], RMSprop +1.3 [−0.6, +3.7].
- **Warm − cold** is flat across severity: positive in every quintile for SGD/Adam/AdamW, about 0 for RMSprop.
- **Policy vs regular-tuned cosine depends strongly on severity for SGD:**

  | Quintile | Q1 | Q2 | Q3 | Q4 | Q5 |
  |---|---:|---:|---:|---:|---:|
  | Warm − cosine, SGD (pp) | −2.8 | −0.8 | +3.5 | +10.5 | +22.6 |

  Slope +34.4 pp/nat [+27.2, +42.9], r = 0.68. For Adam/AdamW/RMSprop, cosine beats the warm policy in Q1–Q4 by 3–5 pp, and only in Q5 does the CI reach 0: Adam −1.2 [−4.4, +2.1], RMSprop −3.1 [−6.4, +0.2].
- **Frozen regular policies get worse with severity for Adam/AdamW** relative to regular fixed (slope −28 and −20 pp/nat). The warm/cold minus frozen gaps grow correspondingly.

*Exploratory, not prespecified* (`structure_breakdown.csv`): the advantage of regular-tuned cosine over the warm policy is concentrated in orderings with **no recovery** (R = 0, 34 orderings): Adam −10.9 pp, RMSprop −14.4 pp. With 1–20 recovery batches it reverses (Adam +2.8, RMSprop +2.4). The cosine schedule's near-zero final learning rate limits damage from a block that ends the stream. Against extreme-retuned fixed, the warm policy gains most on R = 0 orderings (SGD +25.6, Adam +18.4).

### Do the seven-ordering conclusions replicate?

Same code, same environment, and the same bootstrap. For the seven originals, the bootstrap is over 7 orderings × 20 seeds × policy seeds, which gives wide CIs. "Replicates" means both 95% CIs exclude 0 with the same sign. Full table: [`tables.md`](tables.md#replication-seven-original-orderings-vs-generated-suite); `replication.csv`.

**Replicates** (23 of 48 optimizer × comparison cells):
- Warm − fixed (regular): SGD/Adam/AdamW.
- Warm − fixed (extreme): Adam/AdamW. SGD and RMSprop now also exclude 0: SGD +6.17, RMSprop +2.69.
- Warm − frozen and cold − frozen: SGD/Adam/AdamW.
- Warm − cold: SGD/Adam/AdamW. Seven orderings: +2.3/+5.3/+5.8. Generated: +2.1/+3.3/+3.4, winning on 70%/97%/98% of orderings.
- Cold − fixed (extreme): all four now exclude 0 (+2.3 to +4.0). On the seven, none did.
- Frozen − fixed (regular) and frozen − cosine (regular): Adam/AdamW, negative on both suites.
- SGD policies beat regular-tuned cosine on both suites.

**Does not replicate.** Stated plainly:
1. **Policies vs the regular-tuned cosine schedule (Adam, AdamW, RMSprop).**
   - On the seven originals, the warm policy is ahead of cosine: +8.7/+9.1/+3.2 pp, CIs including 0. On the generated suite it is **behind**: −3.25/−3.25/−4.08 pp, CIs excluding 0, losing on 86–89% of orderings and with 0/5 seeds ahead.
   - The cold policy is the same: +3.4/+3.3/+4.6 on the seven vs −6.6/−6.6/−4.3 generated.
   - On the generated suite, regular-tuned cosine is the best method for these three optimizers.
2. **Extreme-retuned fixed vs regular-tuned fixed (Adam, AdamW, RMSprop).** Retuning on extreme development data helps on the seven (+2.6/+2.6/+3.2, CIs about 0) but **hurts** on the generated suite (−2.1/−2.1/−2.8, CIs excluding 0). The extreme dev set uses exactly the seven ordering types, so the retuned configuration (low LR) is fitted to them.
3. **RMSprop policy gains over regular-tuned controls.**
   - On the seven: warm − fixed (regular) +4.6, warm − frozen +4.1, cold − fixed (regular) +6.0, cold − frozen +5.4, all CIs excluding 0.
   - On the generated suite: −0.1, −0.9, −0.3, −1.1, with CIs including 0 (cold − frozen's CI ends at +0.05).
   - RMSprop's only robust policy gain is over the extreme-retuned fixed configuration.
4. **Cold − fixed (regular), Adam/AdamW:** +5.9/+5.8 on the seven (CIs include 0) vs +0.2/+0.2 generated. There is no evidence of a gain on the broader family.

The generated family is milder than the seven on average, and several of these reversals shrink in the top severity quintile. The family as prespecified is one reasonable distribution over biased orderings, not the only one. A suite weighted toward severe orderings, like the seven, would favor the extreme-trained methods more.

## Commands

```bash
export MNIST_DATA_DIR=/workspace/cs238/data OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest experiments/extreme_orderings/test_generated.py   # stream invariants, design checks
python experiments/extreme_orderings/generated_eval.py design        # design.json, orderings.csv (prespecification)
python experiments/extreme_orderings/generated_eval.py smoke         # timing on smoke-only orderings/seeds
python experiments/extreme_orderings/check_generated_replay.py       # replay_check.json
nohup python -u experiments/extreme_orderings/generated_eval.py launch --workers 28 \
  > results/extreme_generated/logs/launch.out 2>&1 &                 # 720 jobs, resumable
python experiments/extreme_orderings/generated_originals.py launch --workers 28   # seven originals, same methods
python experiments/extreme_orderings/analyze_generated.py            # tables, CSVs, figure
```

## Files

| File | Contents |
|---|---|
| `design.json` | the prespecification |
| `orderings.csv` | spec, severity, nearness and episode seeds per ordering |
| `original_specs.csv` | the seven originals as specs |
| `evaluation/{family}_{method}_{seed}/chunk_XX.csv` | per-episode rows, 20 orderings per chunk |
| `evaluation_original/` | the seven-ordering replay |
| `summary.csv`, `policy_seed_scores.csv` | mean accuracy |
| `comparisons.csv` | paired differences with 95% and 90% CIs, win shares, seed wins and SD |
| `per_ordering_differences.csv` | per-ordering differences |
| `severity_bins.csv`, `severity_regression.csv` | severity analysis |
| `structure_breakdown.csv` | exploratory breakdown by structure |
| `seven_original_comparisons.csv`, `replication.csv` | replication check |
| `tables.md` | all tables |
| `replay_check.json` | replay against committed rows |
| `unit_tests.txt` | test output |

## Checks and caveats

- **Replay.** The runners reproduce committed extreme-suite rows exactly (0 difference) on pair_max and staircase × 4 seeds for warm, cold, extreme-retuned fixed, regular fixed and regular cosine, all four optimizers. Frozen regular policies differ by up to 0.21 pp on staircase. Those committed rows came from an earlier environment (the README notes a similar pre-existing drift), and all methods here are fresh in one environment.
- **Not new test images.** Episodes are new, but the MNIST test set is the same one used throughout. The policies, configurations and checkpoints are the saved ones; their selection never saw these episodes.
- **The sampling distribution is a choice.** It was fixed before evaluation, but a different choice would change the averages. The family is milder than the seven originals (see Severity), so the generated means are not directly comparable to the paper's seven-ordering means.
- **Pairing.** Five episode seeds per ordering. The policy-seed SDs show large seed variation for frozen Adam/AdamW (about 7 pp) and RMSprop warm (2.3 pp).
- **Failures.** Failed episodes are retained: frozen SGD 16, frozen Adam 36, frozen AdamW 2, cosine SGD 7; all other methods 0.
- **Multiplicity.** No adjustment; CIs are marginal and conditional on the saved checkpoints.
- **Sibling baselines.** The extreme-retuned cosine and random baselines from the sibling PR were not yet available when this was written (see the section below, if present).
