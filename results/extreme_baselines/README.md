# Extreme-ordering schedule and random baselines

Tests the claim "on the extreme orderings, the policy trained on extreme orderings
beats a tuned cosine schedule, and cosine is roughly no better than a random
baseline", for all four optimizers, on the full extreme suite (eight orderings ×
test seeds 1920000–1920019, the same paired episodes as every other extreme result).

## Prespecified design (committed before any test evaluation)

This section was committed before any test-suite episode of a new method was run.
Results added later do not change it; any deviation is listed under Caveats.

### Methods

1. **Cosine (and step, warmup-cosine) retuned on extreme.** The PR #14 schedule search
   (`experiments/prospective_schedules/run.py`: same three shapes, same axes, same
   candidates — the full 144-point SGD grid per shape and the same seeded 160-of-288
   sample per shape for Adam/AdamW/RMSprop), but each candidate is scored on the
   **extreme development episodes** the retrained policies used for checkpoint
   selection: seeds 1930000–1930015 × `ORDERS * 2` (each extreme ordering twice), dev
   images, mean `b.score` (the same development score). Lowest score wins; first
   minimum on ties. No test data are read. The comparator for the claim is the
   **cosine** shape; the dev-best shape over the three is reported too.
   The regular-tuned cosine rows from PR #14 (`results/prospective_schedules/extreme/`)
   are kept as a second schedule comparator.
2. **Random baselines.**
   - **(a) Random schedule.** At each of the 40 control steps (every 5 updates) every
     post-tanh action component `a` (3 for SGD, 4 for the adaptive families) is drawn
     i.i.d. uniform on [−1, 1] and fed through the exact policy action mapping of the
     engine (`lr ← clamp(lr·2^a₀)`, momentum/β1 = 0.95(a₁+1)/2, the β2/α and weight-decay
     maps), from the same starting settings as the policy. This is implemented as a
     stand-in policy module whose output `μ = atanh(a)` replaces the network, so the
     rollout path is the policy's own. Draws use NumPy
     `default_rng([238, 7, random_seed, episode_seed, order_index])`, so random seed
     *k* ∈ {0,…,4} uses the same draws on every family and batch, and the five random
     seeds are paired across episodes like policy seeds. **Primary:** cold start
     (the cold-retrained policy's starting settings). **Secondary:** warm start from the
     extreme-retuned fixed configuration (the warm-start policy's starting settings).
   - **(b) Untrained policy.** The update-0 policy (initial weights) for seeds
     338–342, built exactly as training builds it, cold start and warm start from the
     extreme-retuned fixed configuration (`warm=` path, as in PR #15). The rebuild is
     checked against the update-0 development rows saved by each training run.
     Because the engines zero-initialize the output layer, the untrained policy's
     actions do not depend on the seed or observations; it is a constant schedule
     (expected identical across seeds; checked and reported).
   - **(c) Random fixed configuration.** Five configurations, *k* ∈ {0,…,4}, each the
     grid point `default_rng([238, 11, family_index, k]).integers(len(grid))` of the
     fixed grid used to tune the extreme fixed baseline (`retrain.configs(family)`:
     144 SGD / 288 adaptive points), independent draws (duplicates allowed and
     reported). Replayed as a constant schedule through the same batched engine as the
     cosine schedules.
3. **Comparators.** Fresh rows from PR #15 (`results/extreme_warm_start/evaluation/`),
   which are paired on the same episodes with the same protocol: extreme-retuned fixed
   (`fixed_retuned`), cold extreme-retrained policies (`policy_retrained`, seeds
   338–342), extreme warm-start policies (`warm_extreme`, seeds 338–342). Frozen
   regular policies (seeds 338–342) are re-evaluated fresh here through the same path.
   The extreme-retuned fixed configuration is also replayed through the batched engine
   (`fixed_engine`), because the saved SGD fixed rows use `torch.optim.SGD`.

All new test evaluations: one torch thread, batches of four test seeds per ordering,
official MNIST test set, failed episodes kept at their recorded accuracy.

### Analysis

- Accuracy in %, biased mean = equal-weight mean of the seven non-`mixed` orderings,
  all-type mean = all eight. Per-ordering tables, failed episodes, policy/random-seed SD.
- Differences A − B in pp with `scripts/paired_bootstrap.py`'s paired hierarchical
  bootstrap (10,000 resamples, NumPy seed 238; seeds then episodes; unseeded rows
  broadcast), 95% percentile CIs; plus 90% percentile CIs from the same resamples.
  Seed wins = seeds whose biased-mean difference is > 0.
- Prespecified comparisons (biased mean is primary; all-type reported too):
  - P1 extreme warm-start policy − retuned cosine
  - P2 cold extreme-retrained policy − retuned cosine
  - E1 retuned cosine − random schedule (cold) — **primary random baseline**
  - E2 retuned cosine − random schedule (warm)
  - E3 retuned cosine − untrained policy (cold); E4 − untrained policy (warm)
  - E5 retuned cosine − random fixed configuration
  - F1 retuned cosine − extreme-retuned fixed (PR #15 rows; for SGD also same-engine)
  - Secondary: the same with the regular-tuned cosine (PR #14) and the dev-best shape;
    frozen regular policies − retuned cosine.
- **Decision rules, per optimizer.**
  - "Policy > tuned cosine" (P1, P2): **holds** if the 95% CI lies above 0; **fails** if
    it lies below 0; otherwise **inconclusive**.
  - "Cosine ≈ random" (E1–E5): equivalence margin **±1.0 pp on the biased mean**.
    **Holds** if the 90% CI lies inside [−1.0, +1.0] (TOST at α = 0.05 each side);
    **fails** if the 95% CI lies entirely outside the margin on one side (|difference|
    > 1 pp established); otherwise **inconclusive**. A 95% CI crossing zero is not
    evidence of equivalence. E1 decides the claim; E2–E5 are reported alongside.
- No multiplicity adjustment; all CIs are marginal and conditional on the saved
  checkpoints.

## Commands

```bash
export MNIST_DATA_DIR=/path/containing/MNIST OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest experiments/extreme_baselines/test_baselines.py
(cd scripts && python -m unittest test_paired_bootstrap)
python experiments/extreme_baselines/run.py smoke --output /tmp/extreme_baselines_smoke   # 1 candidate, 4 test seeds
nohup python -u experiments/extreme_baselines/run.py pipeline --workers 8 \
  > results/extreme_baselines/logs/pipeline.log 2>&1 < /dev/null &
python experiments/extreme_baselines/analyze.py
```

Single jobs: `run.py search --family F --kind {cosine,step,warmup_cosine}`, `run.py select --family F`,
`run.py check-untrained`, `run.py evaluate --family F --method M --seed S` (methods and seeds in `METHODS`).
Completed jobs are skipped on re-run. The smoke run also checks that this rollout path reproduces PR #15's
fresh cold-policy rows (32/32 episodes exact for SGD, Adam and RMSprop).

Layout: `search/{F}_{kind}.csv` (every candidate's extreme dev score), `selected/{F}.json` (chosen
configs, plus same-engine dev scores of the extreme-tuned fixed config and of PR #14's regular-tuned cosine on
the same 16 dev episodes), `evaluation/{F}_{method}_{seed}.{csv,json}` (test rows and metadata; traces for the
schedules, fixed replay and random seed 0; per-step mean/SD settings for random and untrained runs),
`untrained_check.csv`, `summary.csv`, `per_order.csv`, `comparisons.csv` (biased, all and every ordering, with
95% and 90% CIs), `drift.csv`, `verdicts.json`, [`tables.md`](tables.md) (all tables, including per-ordering),
`logs/`.

## Selected schedules (extreme development data only)

| Family | Cosine: LR0 / memory / second memory / WD (dev score) | Dev-best shape | Extreme-tuned fixed, same 16 episodes | Regular-tuned cosine, same 16 episodes |
|---|---|---|---:|---:|
| SGD | 0.3 / 0 / – / 0 (0.311) | cosine | 0.785 | 0.902 |
| Adam | 0.003 / 0 / 0.99 / 1e-4 (−0.044) | cosine | 0.281 | 0.477 |
| AdamW | 0.003 / 0 / 0.99 / 0 (−0.040) | cosine | 0.281 | 0.477 |
| RMSprop | 3e-4 / 0.5 / 0.999 / 0 (−0.032) | warmup-cosine (−0.088; LR0 1e-3, mom 0, α 0.999) | −0.025 | 0.300 |

Lower is better. Search cost: 432 candidates (SGD) / 480 (adaptive) × 16 episodes = 6,912 / 7,680 dev episodes.

## Results

Accuracy (%) on the extreme suite. ± is the SD over the five policy / random seeds. Untrained policies are
identical across seeds (checked: every episode's accuracy agrees across seeds 338–342). Failed episodes are
kept; counts are over all eight orderings.

### Method means

| Method | SGD biased | Adam biased | AdamW biased | RMSprop biased | SGD all | Adam all | AdamW all | RMSprop all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Cosine, extreme-tuned** | 62.27 | 72.70 | 72.69 | 70.27 | 65.77 | 75.01 | 74.99 | 72.88 |
| Warmup-cosine, extreme-tuned | 61.35 | 70.61 | 71.23 | 72.38 | 64.97 | 73.18 | 73.73 | 74.80 |
| Step, extreme-tuned | 59.88 | 67.86 | 67.94 | 67.16 | 63.66 | 70.75 | 70.82 | 70.20 |
| Cosine, regular-tuned (PR #14) | 47.66 (15 failed) | 65.82 | 65.82 | 68.41 | 53.35 | 69.35 | 69.35 | 71.62 |
| Fixed, extreme-tuned (PR #15) | 61.32 | 65.91 | 65.91 | 70.22 | 64.76 | 68.94 | 68.94 | 72.78 |
| **Warm extreme policy (5)** | 69.81 ± 1.23 | 74.47 ± 1.20 | 74.90 ± 1.80 | 71.65 ± 1.37 | 72.50 ± 1.08 | 76.72 ± 0.98 | 77.11 ± 1.54 | 74.20 ± 1.31 |
| **Cold extreme policy (5)** | 67.48 ± 1.86 | 69.22 ± 2.81 | 69.13 ± 3.47 | 72.98 ± 1.05 | 70.57 ± 1.73 | 71.99 ± 2.45 | 71.90 ± 3.05 | 75.26 ± 0.91 |
| Frozen regular policy (5, fresh) | 51.63 ± 1.23 (1) | 46.05 ± 10.01 (12) | 49.07 ± 10.78 | 67.56 ± 0.91 | 56.75 | 51.89 | 54.53 | 70.82 |
| **Random schedule, cold (5)** | 54.84 ± 0.72 (10) | 55.34 ± 1.79 (47) | 57.26 ± 1.70 (31) | 49.74 ± 1.50 (127) | 59.21 | 59.17 | 61.02 | 53.82 |
| Random schedule, warm (5) | 54.84 ± 0.72 (10) | 57.88 ± 1.39 (16) | 59.13 ± 1.08 (12) | 56.93 ± 1.44 (26) | 59.21 | 61.35 | 62.68 | 60.73 |
| Untrained policy, cold (5) | 57.43 ± 0 | 49.35 ± 0 | 49.22 ± 0 | 68.89 ± 0 | 61.51 | 54.59 | 54.49 | 71.77 |
| Untrained policy, warm (5) | 61.32 ± 0 | 65.93 ± 0 | 65.91 ± 0 | 70.15 ± 0 | 64.76 | 68.96 | 68.94 | 72.72 |
| Random fixed config (5) | 35.22 ± 5.78 (27) | 50.25 ± 11.91 | 55.05 ± 8.89 | 36.41 ± 23.51 (194) | 39.72 | 54.65 | 59.39 | 42.17 |

The batched-engine replay of the extreme-tuned fixed config equals PR #15's fixed rows on every episode for all
four families (difference 0.00; for SGD the tuned momentum is 0, so `torch.optim.SGD` and the engine agree).

### Prespecified comparisons, biased mean (pp, 95% CI [90% CI for E/F]; seed wins; decision)

| ID | A − B | SGD | Adam | AdamW | RMSprop |
|---|---|---|---|---|---|
| P1 | warm extreme policy − extreme cosine | +7.54 [+6.25, +8.83]; 5/5; **holds** | +1.77 [+0.58, +2.82]; 4/5; **holds** | +2.21 [+0.75, +3.80]; 4/5; **holds** | +1.37 [+0.26, +2.67]; 5/5; **holds** |
| P2 | cold extreme policy − extreme cosine | +5.21 [+3.45, +6.86]; 5/5; **holds** | −3.48 [−5.62, −1.05]; 1/5; **fails** | −3.56 [−6.30, −0.64]; 1/5; **fails** | +2.70 [+1.85, +3.71]; 5/5; **holds** |
| E1 | extreme cosine − random schedule (cold) | +7.43 [+6.05, +8.90] (90% +6.28, +8.62); 5/5; **fails** | +17.37 [+15.34, +19.30] (+15.71, +18.97); 5/5; **fails** | +15.43 [+13.51, +17.46] (+13.83, +17.12); 5/5; **fails** | +20.54 [+18.17, +23.10] (+18.56, +22.65); 5/5; **fails** |
| E2 | extreme cosine − random schedule (warm) | +7.43 [+6.05, +8.90]; **fails** | +14.82 [+12.69, +16.73]; **fails** | +13.56 [+11.66, +15.30]; **fails** | +13.35 [+11.44, +15.23]; **fails** |
| E3 | extreme cosine − untrained (cold) | +4.84 [+3.79, +5.84]; **fails** | +23.35 [+22.20, +24.39]; **fails** | +23.46 [+22.21, +24.66]; **fails** | +1.39 [+0.81, +1.96] (+0.90, +1.87); **inconclusive** |
| E4 | extreme cosine − untrained (warm) | +0.95 [+0.24, +1.64] (+0.36, +1.54); **inconclusive** | +6.77 [+6.02, +7.50]; **fails** | +6.78 [+6.05, +7.50]; **fails** | +0.12 [−0.48, +0.67] (−0.37, +0.59); **holds (equivalent)** |
| E5 | extreme cosine − random fixed config | +27.05 [+22.78, +32.00]; **fails** | +22.45 [+15.92, +33.20]; **fails** | +17.64 [+11.44, +25.33]; **fails** | +33.87 [+16.55, +52.33]; **fails** |
| F1 | extreme cosine − extreme-tuned fixed | +0.95 [+0.24, +1.64] (+0.37, +1.54) | +6.79 [+6.02, +7.55] | +6.78 [+6.05, +7.50] | +0.05 [−0.59, +0.69] (−0.49, +0.60) |

"Fails" for E1–E5 means the 95% CI lies entirely above +1 pp: cosine is better than that random baseline by
more than the margin. For E rows, wins count random seeds (or untrained policy seeds) that cosine beats.
All-type means give the same decisions (see [`tables.md`](tables.md)).

Secondary (biased mean, 95% CI; seed wins):

| ID | A − B | SGD | Adam | AdamW | RMSprop |
|---|---|---|---|---|---|
| S1 | warm extreme policy − regular-tuned cosine | +22.16 [+20.45, +23.85]; 5/5 | +8.65 [+7.29, +9.96]; 5/5 | +9.09 [+7.44, +10.78]; 5/5 | +3.24 [+1.80, +4.77]; 5/5 |
| S2 | cold extreme policy − regular-tuned cosine | +19.83 [+17.79, +21.76]; 5/5 | +3.40 [+1.21, +5.85]; 5/5 | +3.31 [+0.49, +6.21]; 4/5 | +4.57 [+3.35, +5.82]; 5/5 |
| S3 | regular-tuned cosine − random schedule (cold) | −7.18 [−8.73, −5.54] | +10.48 [+8.48, +12.42] | +8.56 [+6.67, +10.68] | +18.67 [+16.40, +21.20] |
| S4 | extreme cosine − regular-tuned cosine | +14.61 [+13.70, +15.51] | +6.88 [+5.86, +7.89] | +6.87 [+5.89, +7.83] | +1.87 [+1.07, +2.67] |
| S5 | warm extreme policy − dev-best shape | = P1 | = P1 | = P1 | −0.74 [−2.00, +0.65]; 2/5 |
| S6 | cold extreme policy − dev-best shape | = P2 | = P2 | = P2 | +0.59 [−0.41, +1.74]; 3/5 |
| S7 | frozen regular policy − extreme cosine | −10.64 [−11.88, −9.58]; 0/5 | −26.65 [−34.70, −19.11]; 0/5 | −23.61 [−32.70, −15.94]; 0/5 | −2.72 [−3.59, −2.01]; 0/5 |
| S10 | warm extreme policy − untrained (warm) | +8.49 [+7.19, +9.75]; 5/5 | +8.54 [+7.36, +9.55]; 5/5 | +8.99 [+7.53, +10.55]; 5/5 | +1.49 [+0.43, +2.75]; 5/5 |
| S11 | cold extreme policy − untrained (cold) | +10.05 [+8.37, +11.68]; 5/5 | +19.87 [+17.70, +22.34]; 5/5 | +19.90 [+17.10, +22.86]; 5/5 | +4.09 [+3.13, +5.19]; 5/5 |

S8/S9 (policy − random schedule) are in `tables.md`. They pair random seed *k* with policy seed 338+*k*; the
two seed sets are independent, so the pairing is arbitrary.

Per-ordering means and per-ordering differences for P1–F1 are in [`tables.md`](tables.md).

## Verdict on the claim, per optimizer

| Optimizer | Warm extreme policy > tuned cosine | Cold extreme policy > tuned cosine | Cosine ≈ random (E1, random schedule) |
|---|---|---|---|
| SGD | **holds** (+7.5 pp, 5/5) | **holds** (+5.2 pp, 5/5) | **fails**: cosine better by +7.4 pp |
| Adam | **holds** (+1.8 pp, 4/5) | **fails**: cosine better by 3.5 pp (1/5) | **fails**: cosine better by +17.4 pp |
| AdamW | **holds** (+2.2 pp, 4/5) | **fails**: cosine better by 3.6 pp (1/5) | **fails**: cosine better by +15.4 pp |
| RMSprop | **holds** (+1.4 pp, 5/5; vs the dev-best warmup-cosine −0.7 pp, inconclusive) | **holds** (+2.7 pp, 5/5; vs warmup-cosine inconclusive) | **fails**: cosine better by +20.5 pp |

- **"Cosine ≈ random" is not supported for any optimizer.** An extreme-tuned cosine beats the random schedule,
  the cold untrained policy and random fixed configurations by far more than the ±1 pp margin for all four
  optimizers (one exception: RMSprop vs the cold untrained policy, +1.4 pp, inconclusive). The only
  equivalence shown is RMSprop's cosine vs the warm untrained policy (+0.12 pp). That policy is in effect the
  extreme-tuned fixed config, so this says cosine ≈ tuned fixed for RMSprop, not cosine ≈ random. For SGD,
  cosine − warm untrained is +0.95 pp, inconclusive for the same reason (F1 SGD +0.95 pp).
- What *is* close to cosine is **tuned fixed** for SGD (+0.95 pp) and RMSprop (+0.05 pp). For Adam/AdamW the
  extreme-tuned cosine beats tuned fixed by 6.8 pp.
- **"Trained policy > tuned cosine" holds for the warm-start extreme policies on all four optimizers.** The
  margins are smaller than against tuned fixed (+1.4 to +2.2 pp for the adaptive families, +7.5 pp for SGD).
  For RMSprop the dev-selected shape was warmup-cosine, and against it the warm policy is −0.74 pp
  (inconclusive).
- For the **cold-start** extreme policies the claim holds for SGD and RMSprop but **fails for Adam and AdamW**:
  retuned cosine is 3.5 pp better, and 4/5 seeds lose.
- Retuning matters: the regular-tuned cosine from PR #14 is 1.9–14.6 pp worse on extreme than the extreme-tuned
  one (S4). Comparing extreme-trained policies to the regular-tuned cosine overstates their lead by 2–15 pp.
- What training adds: every extreme policy beats its own untrained (update-0) start by a clear margin (S10/S11),
  smallest for warm RMSprop (+1.5 pp).

## Caveats

- **Development budget.** Each schedule candidate is scored on all 16 extreme dev episodes (the policies'
  checkpoint-selection set), as specified. That is twice PR #14's 8-episode score; the candidate set is
  identical. The extreme-tuned fixed config (PR #15 comparator) was selected on the first 8 of those
  episodes. Schedule search: 6,912–7,680 dev episodes per family; fixed tuning: 1,152 (SGD) / 2,304. Policies
  were trained on 4,000 training episodes plus 128 dev episodes.
- **Grid edge.** The SGD cosine picked LR0 = 0.3, the top of the SGD LR axis (as in PR #14). A wider axis might
  move SGD's schedule result. The axes were kept as specified.
- **Random schedule = SGD cold and warm.** For SGD they are identical, because the extreme-tuned LR (0.1) equals
  the engine's cold-start LR and momentum/WD are set absolutely from the action at every step.
- **Untrained policies** are constant schedules: the output layer is zero-initialized, so seed and feedback
  have no effect at update 0. Their seed SD is 0 and each 5-seed row is effectively one condition. The warm
  one equals the tuned fixed config up to the warm clamps (SGD WD 0 → 1e-6).
- **Random fixed configs** are five independent grid draws. Their spread is large (RMSprop SD 23.5 pp, 194
  failed episodes), so their CIs are wide even though every decision is clear.
- **Replay drift.** Fresh frozen-regular rows vs the saved rows used elsewhere: 683–721 of 800 episodes are
  exactly equal per family, mean |Δ| 0.04–0.07 pp, single-episode max 9.7 pp (`drift.csv`). Group means agree
  with PR #14's saved values within 0.03 pp. PR #15's rows were reproduced exactly by this rollout path in the
  smoke check.
- **Reused rows.** The warm/cold extreme policies and tuned fixed rows are PR #15's fresh evaluations, and the
  regular-tuned cosine rows are PR #14's. Both used the same test episodes, batching (4) and one thread.
- The test seeds and MNIST test set are the project's usual ones, already seen by the authors in earlier
  results. No test metric chose any schedule, configuration or checkpoint here.
- CIs are marginal, with no multiplicity adjustment across the ~20 comparisons × 4 optimizers, and conditional
  on the saved checkpoints. Unseeded methods (schedules, fixed) carry episode-level uncertainty only.
- **Runtime.** The pipeline took 3,198 s wall on 8 one-thread workers on a shared 32-vCPU VM (load ≈ 13–15 from
  other sessions). Python 3.12, torch 2.5.1+cpu.
