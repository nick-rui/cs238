# Prospective open-loop schedules (regular and extreme suites)

The earlier open-loop control (`results/open_loop/`) replays each policy's mean
schedule averaged over the **test** trajectories, so it is retrospective. This
experiment builds two open-loop controls fitted **without any test data** and
evaluates them on the regular suite (ten orderings × 20 paired episode seeds
1900000–1900019), plus the frozen extreme suite (eight orderings × seeds
1920000–1920019) as an extra. All four optimizers. Paired against the saved fixed
rows and the saved five-seed (338–342) frozen-policy rows.

## Design

**(a) Development mean schedule (`dev_mean`).** For each of the 20 saved
policies (4 families × seeds 338–342), roll the deterministic policy on its 16
development episodes (seeds 1810000–1810015, the four training regimes:
mixed/single_train/pair_train/double_train), take the arithmetic mean of each
actual setting (LR, momentum/β1, β2/α, WD) at each of the 40 control steps, then
replay that single schedule unchanged on every test condition and episode. The
policy is never called on test episodes. Each replay checks that every traced
setting equals the schedule exactly (float32).

**(b) Standard LR schedules, tuned on development data.** Three LR shapes over the
40 control steps (each step is five updates):

- `cosine`: `lr0 · ½(1 + cos(π t/40))`
- `step`: `lr0`, then ×0.1 at 50% and ×0.1 again at 75% of training
- `warmup_cosine`: linear warmup over the first 10% (`lr0·(t+1)/4` for t<4), then cosine over the rest

The peak LR and the other hyperparameters are searched on the **existing fixed-grid
axes**:

- SGD: LR {.001,.003,.01,.03,.1,.3} × momentum {0,.25,.5,.75,.9,.95} × WD {0,1e-6,1e-4,.01}. That is the full 144-point grid per shape, 432 candidates, 3,456 episodes.
- Adam/AdamW/RMSprop: LR {1e-4,3e-4,1e-3,3e-3,.01,.03} × β1/momentum {0,.5,.9,.95} × β2/α {.9,.99,.999,.9999} × WD {0,1e-4,.01}. For each shape, 160 of the 288 points are drawn without replacement (NumPy seed `[238, family, shape]`): 480 candidates, **3,840 episodes**, exactly the compute-matched adaptive budget (`results/compute_matched_baseline`). SGD stays under its 4,032-episode compute-matched budget.

Selection uses exactly the fixed-baseline protocol: the minimum mean development
score (log final dev loss + mean log dev-path loss) on seeds 1810000–1810007, two
per training regime. The headline "tuned schedule" is the development-best shape
per family. Each shape's own development-best row is also reported. Schedules run
through the policy engine's schedule path (`e.rollout(schedules=...)` for SGD,
`family_engine.rollout(schedule=...)` otherwise), with the same initialization,
streams, failure handling and evaluation.

**Same-engine fixed control (`fixed_engine`).** The tuned fixed configuration is
replayed as a constant schedule through the same engine path. For SGD the saved
fixed rows come from `torch.optim.SGD` one episode at a time (`study.baseline`),
while policies and schedules use the batched engine. This control measures that
implementation difference.

Development-selected schedules (all chose cosine; LR at or near the top of the grid):

| Family | Shape | LR0 | Memory | Second memory | WD | Dev score (fixed = same-engine fixed score) |
|---|---|---:|---:|---:|---:|---|
| SGD | cosine | 0.3 | 0.75 | – | 0 | −2.082 (fixed −1.845) |
| Adam | cosine | 0.03 | 0.5 | 0.9 | 0 | −2.673 (fixed −2.281) |
| AdamW | cosine | 0.03 | 0.5 | 0.9 | 0 | −2.673 (fixed −2.281) |
| RMSprop | cosine | 0.01 | 0.5 | 0.9 | 0 | −2.756 (fixed −2.599) |

With WD = 0, Adam and AdamW are the same update, so their schedule rows coincide.

## Commands

```bash
export MNIST_DATA_DIR=/path/containing/MNIST OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup python -u experiments/prospective_schedules/run.py pipeline --workers 4 \
  > results/prospective_schedules/pipeline.log 2>&1 < /dev/null &
python experiments/prospective_schedules/summarize.py   # also writes results/reactive_rule tables
python -m unittest experiments/reactive_rule/test_controls.py
```

Individual jobs: `run.py fit --family F --seed S`, `run.py search --family F --kind cosine`,
`run.py select --family F`, `run.py evaluate --suite regular --family F --method {dev_mean --seed S|cosine|step|warmup_cosine|fixed_engine}`.
Completed jobs are skipped on re-run. `--output DIR` redirects everything (for smoke tests).

Layout: `dev_mean/{F}_{S}/` (development schedule, development rows/traces, fit
metadata with checkpoint SHA-256), `search/` (every candidate's dev score),
`selected/{F}.json`, `{regular,extreme}/{F}_{method}[_{seed}]/` (evaluation,
compressed traces, replayed schedule, timing), `summary.csv`, `comparisons.csv`
(bootstrap for the biased, mixed and all groups), `tables.md`, `logs/pipeline.jsonl`.

## Results

Differences are A − B in percentage points with paired hierarchical bootstrap 95%
CIs (`scripts/paired_bootstrap.py` unchanged: 10,000 resamples, seed 238, policy
seeds then episode seeds; unseeded rows broadcast). "Seeds beating" counts policy
seeds whose biased-mean difference is positive. ± is the policy-seed SD. Failed
episodes are retained.

### Regular suite, biased mean (nine orderings)

| | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| Fixed (saved) | 88.36 | 91.50 | 91.50 | 92.37 |
| Policy (saved, 5 seeds) | 89.61 ± 0.51 | 90.31 ± 0.66 | 90.34 ± 0.80 | 92.57 ± 0.07 |
| Dev-mean schedule (5 seeds) | 88.71 ± 0.33 | 90.14 ± 0.62 | 90.08 ± 0.91 | 92.49 ± 0.08 |
| Tuned schedule (cosine) | 88.93 | **93.21** | **93.21** | **93.28** |
| **Policy − dev-mean schedule** | **+0.90 [+0.39, +1.55]; 5/5** | +0.17 [−0.18, +0.67]; 2/5 | +0.26 [−0.13, +0.71]; 3/5 | +0.08 [−0.01, +0.17]; 3/5 |
| **Policy − tuned schedule** | **+0.68 [+0.13, +1.21]; 5/5** | −2.91 [−3.52, −2.34]; 0/5 | −2.87 [−3.57, −2.18]; 0/5 | −0.71 [−0.86, −0.57]; 0/5 |
| Tuned schedule − fixed (saved) | +0.56 [+0.08, +1.08] | +1.71 [+1.56, +1.86] | +1.71 [+1.56, +1.86] | +0.91 [+0.79, +1.02] |
| Dev-mean schedule − fixed (saved) | +0.34 [−0.11, +0.71]; 4/5 | −1.36 [−1.93, −0.80]; 0/5 | −1.42 [−2.14, −0.62]; 1/5 | +0.12 [+0.01, +0.23]; 5/5 |

### Extreme suite, biased mean (seven orderings; policies and schedules selected on the regular training mixture)

| | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| Fixed (saved) | 47.55 | 63.36 | 63.36 | 67.00 |
| Policy (saved, 5 seeds) | 51.66 ± 1.23 | 46.05 ± 10.02 | 49.04 ± 10.77 | 67.56 ± 0.92 |
| Dev-mean schedule (5 seeds) | 48.11 ± 2.85 (38 failed) | 49.90 ± 10.43 | 51.53 ± 10.93 | 66.40 ± 1.71 |
| Tuned schedule (cosine) | 47.66 (15 failed) | 65.82 | 65.82 | 68.41 |
| **Policy − dev-mean schedule** | **+3.55 [+1.55, +6.41]; 5/5** | −3.85 [−7.04, +0.48]; 1/5 | −2.49 [−5.26, +1.53]; 1/5 | **+1.16 [+0.41, +1.95]; 5/5** |
| **Policy − tuned schedule** | **+4.00 [+2.49, +5.39]; 5/5** | −19.76 [−27.83, −12.25]; 0/5 | −16.78 [−25.80, −9.03]; 0/5 | −0.84 [−1.98, +0.17]; 0/5 |
| Tuned schedule − fixed (saved) | +0.10 [−1.19, +1.54] | +2.45 [+1.73, +3.19] | +2.45 [+1.73, +3.19] | +1.41 [+0.68, +2.18] |

The full tables, including each shape, shuffled accuracy, failures and the
same-engine fixed control, are in [`tables.md`](tables.md).

## Interpretation

- **SGD:** the frozen policy beats both prospective open-loop controls on both
  suites, with all five seeds and CIs excluding zero. This is consistent with a
  benefit from feedback for SGD that its own development schedule and a tuned
  cosine/step/warmup schedule do not reproduce. The tuned SGD schedule beats saved
  fixed on regular (+0.56 pp) but not on extreme.
- **Adam/AdamW:** a development-tuned cosine LR schedule beats the frozen policies
  by about 2.9 pp (regular) and 17–20 pp (extreme), and beats fixed by 1.7–2.5 pp.
  Policy minus its own development schedule is small and not established (CIs
  cross zero). These results give no evidence that feedback causes a gain for the
  original adaptive-family policies.
- **RMSprop:** the policy edges out its own development schedule (+0.08 pp regular,
  CI crosses zero; +1.16 pp extreme, 5/5). The tuned cosine schedule, however, beats
  the policy on regular (−0.71 pp, 0/5 seeds) and is nominally ahead on extreme
  (CI crosses zero).
- Compared with the retrospective control (two seeds, +0.43/+0.42/+0.39/+0.04 pp
  all-order), the prospective development-mean gaps have the same sign for SGD and
  RMSprop. For Adam/AdamW they are not established.

## Caveats

- **SGD engine difference.** The saved SGD fixed rows use `torch.optim.SGD`; replaying
  the identical configuration through the batched engine gives −0.42 pp regular
  [−1.26, +0.02] and **−2.38 pp extreme** [−3.21, −1.60], with 21 vs 0 failed
  episodes on extreme (LR 0.3 is close to instability). Adaptive families agree
  within 0.02 pp. Policies and schedules both use the batched engine, so
  policy-vs-schedule comparisons share an implementation. Comparisons against
  *saved* SGD fixed mix implementations. Both are reported (`fixed (same engine)` rows).
- The development-best LR is at the grid's upper edge for SGD (0.3) and Adam/AdamW
  (0.03). A wider LR axis might make the schedules stronger. The axes were kept as
  specified.
- The adaptive search is a stratified random 160-of-288 subset per shape, not the
  full grid. Its configuration (fixed seed) was set before any evaluation.
- The test data are not new: these seeds and MNIST test images are the ones used
  throughout the project, and the authors have seen earlier results on them. No
  test metric chose any schedule or shape.
- The bootstrap CIs are marginal, not adjusted for the many comparisons, and
  conditional on saved checkpoints. Unseeded rows (schedules, fixed) carry
  episode-level uncertainty only. Extreme results are a transfer test; nothing was
  tuned on extreme orderings.
- Runtime: the pipeline took 5,869 s wall on 4 one-thread workers on a shared
  32-vCPU VM (load ≈ 20 from other sessions). Python 3.12, torch 2.5.1+cpu. Saved
  policy rows come from earlier runs; replay drift between environments is
  documented in the main README (≲0.05 pp on group means).
