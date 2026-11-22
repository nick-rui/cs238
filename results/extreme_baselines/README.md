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
