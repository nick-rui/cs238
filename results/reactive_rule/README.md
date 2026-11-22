# Hand-written reactive rule vs reinforcement learning

Does a simple feedback rule, written before seeing test results and tuned only on
development data, match the REINFORCE policies? The rule uses one coordinate of
the policy's own observation vector.

## Pre-registered design

The rule, grid, selection protocol and decision criterion were committed in
`design.json` and `experiments/reactive_rule/run.py` (commit `4449c8b`) **before
any test evaluation of the rule**. The rule starts from each family's tuned fixed
configuration and reads `cos_t = cosine(upcoming minibatch gradient, optimizer
memory)`, which is the policy's observation index 7 (SGD momentum buffer) or 8
(the Adam/AdamW first moment and the RMSprop momentum buffer):

```
s_0 = 1                                   (memory is empty at t = 0)
s_t = kappa * s_{t-1}                  if cos_t < tau     (cut memory trust)
s_t = s_{t-1} + rho * (1 - s_{t-1})    otherwise          (restore toward 1)
momentum/beta1_t = m_fixed * s_t
lr_t             = lr_fixed * s_t ** gamma                (gamma = 0: LR untouched)
beta2/alpha and weight decay stay at the fixed configuration.
```

The grid has τ ∈ {−0.2, −0.1, 0, 0.1, 0.2, 0.3, 0.5}, κ ∈ {0.25, 0.5, 0.75},
ρ ∈ {0.25, 0.5, 1}, γ ∈ {0, 0.5, 1}, plus one **never-fires** candidate (τ = −1),
which is exactly the fixed configuration. That makes 190 candidates × 8
development episodes = **1,520 episodes**, under the compute-matched 3,840/4,032.
Selection is the fixed-baseline protocol: the minimum mean development score on
seeds 1810000–1810007, two per training regime. The pre-registered decision: RL
beats the rule on a suite/family if the policy − rule biased-mean 95% paired
bootstrap CI is entirely positive, the rule beats RL if it is entirely negative,
and otherwise neither is established.

Evaluation covers the regular (10 orders) and extreme (8 orders) suites with 20
paired seeds each, in batches of 4, through the same engine as the policies (a
`controller=` hook added to both rollouts; it defaults to `None`, so existing
paths are unchanged). The never-fires rule reproduces the same-engine fixed run
bit for bit (unit test).

## Development selection outcome

| Family | Selected (τ, κ, ρ, γ) | Dev score | Never-fires (fixed) score |
|---|---|---:|---:|
| SGD | never fires | −1.8449 | −1.8449 |
| Adam | never fires | −2.2805 | −2.2805 |
| AdamW | never fires | −2.2805 | −2.2805 |
| RMSprop | (−0.2, 0.75, 1.0, 0) | −2.6037 | −2.5993 |

For SGD, Adam and AdamW, every candidate that fires scored worse on development
than doing nothing, so **the pre-registered rule equals the fixed configuration**
for those families. RMSprop's selected rule fires on only 0.3% (regular) and 1.2%
(extreme) of test control steps, and its test accuracy matches fixed to within
0.01 pp.

**Secondary, not pre-registered as a headline (`secondary_firing/`).** For SGD,
Adam and AdamW we also evaluated the best development candidate that can fire
(τ > −1, chosen from the same tuning CSV, so still development-only). For SGD
that is (−0.2, 0.75, 0.25, 0); for Adam/AdamW it is (−0.2, 0.75, 0.5, 0). They
cut memory on about 88–95% of control steps.

## Results (biased mean, pp [95% CI]; seeds beating out of 5)

| Regular suite | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| Fixed (saved) | 88.36 | 91.50 | 91.50 | 92.37 |
| Reactive rule (pre-registered) | 87.95 (= same-engine fixed) | 91.50 (= fixed) | 91.50 (= fixed) | 92.37 |
| Policy (saved, 5 seeds) | 89.61 ± 0.51 | 90.31 ± 0.66 | 90.34 ± 0.80 | 92.57 ± 0.07 |
| **Policy − rule** | **+1.66 [+0.84, +2.81]; 5/5** | **−1.20 [−1.81, −0.62]; 0/5** | **−1.16 [−1.85, −0.47]; 0/5** | **+0.20 [+0.10, +0.30]; 5/5** |
| Rule − fixed (same engine) | 0 | 0 | 0 | +0.00 [−0.00, +0.01] |
| Warm-start policy − rule | +2.91 [+2.06, +4.12]; 5/5 | +1.19 [+0.94, +1.44]; 5/5 | +1.16 [+0.90, +1.44]; 5/5 | +0.26 [+0.18, +0.34]; 5/5 |
| Secondary firing rule − fixed (same engine) | +0.64 [−0.08, +1.70] | −0.01 [−0.12, +0.08] | −0.01 [−0.12, +0.08] | – |
| Policy − secondary firing rule | +1.02 [+0.57, +1.50]; 5/5 | −1.19 [−1.81, −0.63]; 0/5 | −1.15 [−1.83, −0.46]; 0/5 | – |

| Extreme suite | SGD | Adam | AdamW | RMSprop |
|---|---:|---:|---:|---:|
| Fixed (saved) | 47.55 | 63.36 | 63.36 | 67.00 |
| Reactive rule (pre-registered) | 45.18 (21 failed; = same-engine fixed) | 63.35 | 63.35 | 67.01 |
| Policy (saved, 5 seeds) | 51.66 ± 1.23 | 46.05 ± 10.02 | 49.04 ± 10.77 | 67.56 ± 0.92 |
| **Policy − rule** | **+6.48 [+5.10, +7.75]; 5/5** | **−17.29 [−25.45, −9.79]; 0/5** | **−14.31 [−23.37, −6.61]; 0/5** | +0.55 [−0.33, +1.23]; 4/5 |
| Warm-start policy − rule | +4.91 [+1.48, +7.71]; 4/5 | +3.06 [+0.86, +5.38]; 5/5 | +3.63 [+1.96, +5.38]; 5/5 | −0.07 [−2.07, +1.61]; 3/5 |
| Secondary firing rule − fixed (same engine) | +3.26 [+2.52, +3.98] | +0.74 [+0.30, +1.18] | +0.74 [+0.30, +1.18] | – |
| Policy − secondary firing rule | +3.22 [+1.82, +4.53]; 5/5 | −18.03 [−26.15, −10.54]; 0/5 | −15.05 [−24.09, −7.34]; 0/5 | – |

Warm-start rows are the fresh-environment five-seed evaluations in
`results/warm_start/`. They start from the same tuned fixed settings as the rule,
so they are the initialization-matched RL comparison. Full tables (shuffled,
failures, the comparison with the tuned schedule) are in [`tables.md`](tables.md).

## Does RL beat the rule?

By the pre-registered criterion, for the original frozen policies:

- **SGD: yes, on both suites** (+1.66 pp regular, +6.48 pp extreme, 5/5 seeds). It
  still holds against the best rule that fires (+1.02 / +3.22 pp, 5/5).
- **RMSprop: yes on regular** (+0.20 pp, 5/5). **Not established on extreme**
  (CI crosses zero, 4/5).
- **Adam and AdamW: no.** The rule beats RL on both suites, because the rule is
  the tuned fixed configuration and the original policies lose to fixed.
- **Initialization-matched (warm-start) policies beat the rule** on regular for
  all four families (5/5 seeds) and on extreme for SGD, Adam and AdamW. RMSprop on
  extreme is not established.

For SGD, Adam and AdamW, development data preferred no reaction at all within this
rule family. This test therefore mostly compares RL against fixed for those
families. It does not show that *no* hand-written reactive rule could match RL.
The secondary firing rules improve over same-engine fixed on extreme (SGD +3.26 pp,
Adam/AdamW +0.74 pp) even though development scored them worse. Development
selection on mild regimes did not anticipate extreme-shift behaviour. Also, the
dev-tuned cosine LR schedule (`results/prospective_schedules/`) beats the rule
for every family on both suites, and beats the original Adam/AdamW/RMSprop
policies.

## Commands

```bash
export MNIST_DATA_DIR=/path/containing/MNIST OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiments/reactive_rule/run.py design        # writes design.json
nohup python -u experiments/reactive_rule/run.py pipeline --workers 2 \
  > results/reactive_rule/pipeline.log 2>&1 < /dev/null &
# secondary: best firing candidate, SGD/Adam/AdamW, both suites
python experiments/reactive_rule/firing.py --suite regular --family SGD
python experiments/prospective_schedules/summarize.py   # needs prospective_schedules fixed_engine rows
python -m unittest experiments/reactive_rule/test_controls.py
```

## Caveats

- SGD saved fixed rows come from `torch.optim.SGD`, while the rule uses the batched
  engine. The identical configuration differs by −0.42 pp (regular) and −2.38 pp
  (extreme, 21 vs 0 failures). Rule-vs-fixed is therefore reported against the
  same-engine fixed run. Policy-vs-rule shares the engine.
- The rule family is small and one-signal (memory cosine only). The policy also
  sees the loss trend, gradient norm and previous-gradient cosine.
- The test seeds and MNIST test images are not new to the project. No test metric
  chose any rule constant. CIs are marginal, unadjusted, and conditional on saved
  checkpoints. Rule rows carry episode-level uncertainty only.
- Runtime: the pipeline took 2,823 s wall on 2 one-thread workers (shared VM), and
  the secondary evaluations about 13 minutes on 2 workers.
