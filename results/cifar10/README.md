# CIFAR-10 regular suite (flattened MLP)

This run repeats the Fashion-MNIST regular-suite run on CIFAR-10. Everything except the input data matches it: same policy-vs-fixed comparison, regular orderings, policy seeds 338/339, tuning grids, development-only selection, and test protocol. Outputs are isolated under `results/cifar10/`. MNIST and Fashion-MNIST defaults and outputs are unchanged.

**Headline:** on CIFAR-10, the policy-controlled optimizer is worse than its tuned fixed configuration for all four optimizers, on shuffled and biased streams alike. This holds for both policy seeds (0/2 wins in every row) and on every one of the ten orderings. All bootstrap intervals exclude zero.

## Design

- **Data:** `CS238_DATASET=cifar10` loads torchvision CIFAR-10 from `MNIST_DATA_DIR` (default `data/`, which is ignored). Images are flattened in HWC order and scaled by /255, with no normalization or augmentation.
- **Split:** the 50k training images are permuted with the same seed-238 generator and split 40k train / 2k control / 2k development, the split used by `scripts/time_cifar10_episode.py`. The test set is the 10k CIFAR-10 test images.
- **Classifier:** a 3072→64→10 ReLU MLP, the MNIST classifier with a wider input, not a CNN. Initialization seeds are unchanged.
- **Unchanged from Fashion-MNIST:**
  - Streams, policies, rewards, observation scaling and action ranges.
  - 500 outer REINFORCE updates × 8 episodes for each policy seed (338 and 339).
  - Checkpoint selection on 16 development episodes at updates {0, 25, 50, 100, 200, 300, 400, 500}.
  - Fixed tuning on the same 8 development episodes, with grids of 144 configurations for SGD and 288 for each other optimizer.
  - Test: 20 paired episode seeds × 10 regular orderings × (fixed, policy 338, policy 339) = 600 records per optimizer.
- **Selection:** no test feedback is used anywhere. Policy checkpoints and fixed configurations are chosen on development data only.
- **Failed episodes:** retained (accuracy 0 / loss 100 by the existing safeguards), never dropped.
- **Threads:** every process ran one thread (`OMP_NUM_THREADS=MKL_NUM_THREADS=1`, `--threads 1`), including SGD training and evaluation. The Fashion-MNIST run used two threads for those. This changes only floating-point summation order, not the protocol.

## Commands

```bash
# Dataset isolation, split, test-set shape and a one-episode smoke test per optimizer.
MNIST_DATA_DIR=/workspace/cs238/data .venv/bin/python scripts/check_dataset_selection.py
CS238_DATASET=cifar10 MNIST_DATA_DIR=/workspace/cs238/data .venv/bin/python experiments/recovery_study/study.py check
# Full run: at most eight one-thread processes; resumes completed work; logs to results/cifar10/logs/.
PYTHON=.venv/bin/python MNIST_DATA_DIR=/workspace/cs238/data nohup scripts/run_cifar10.sh > results/cifar10/logs/driver.log 2>&1 &
# Seed-level tables (this README's numbers).
.venv/bin/python scripts/summarize_policy_vs_fixed.py --dataset cifar10 --write
```

`run_cifar10.sh` runs the README's Fashion-MNIST sequence with `CS238_DATASET=cifar10`, in this order:

1. Eight policy trainings at once.
2. After the two SGD trainings, the four fixed-baseline tunings.
3. The four evaluations.
4. `run.py import`, then `scripts/analyze_regular.py`.

The SGD and AdamW evaluations were started by hand (the same commands) as soon as their inputs existed, and the driver skipped them. Before launching, a smoke run of 1-update training, a 2-candidate tune and a reduced evaluation was done in a scratch directory.

`summarize_policy_vs_fixed.py` asserts that its means match `analyze_regular.py`'s `aggregates.csv`. On the committed Fashion-MNIST results it also reproduces the 23 failed training episodes reported in the top-level README.

**Wall clock:** 5 h 09 min from launch to analysis on a 32-vCPU VM shared with other experiment sessions (load ≈ 26–32). Policy training took 101–102 min per SGD seed and 149–154 min per Adam/AdamW/RMSprop seed. The RMSprop tune, which ran after the AdamW tune on the same slot, was the critical path. This is within the 5–7 h allowance in `results/cifar10_timing.json`.

## Results

Final test accuracy (%) on the 10k test set. "Shuffled" is the `mixed` ordering. "Biased mean" weights the nine biased orderings equally. Policy − fixed is paired by episode seed.

The 95% interval is the repository's hierarchical paired bootstrap (`scripts/paired_bootstrap.py`: 10,000 resamples, seed 238, resampling policy seeds and then episode seeds). **With only two policy seeds it cannot characterize training-seed uncertainty**, so read it alongside the seed SD and the per-seed values. Episode win rate is the fraction of policy/fixed episode pairs where the policy is higher.

| Optimizer | Condition | Fixed % | Policy % (338 / 339) | Policy − fixed pp (338 / 339) | Seed SD pp | Seed wins | 95% bootstrap CI pp | Episode win rate | Failed test (fixed / policy) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SGD | shuffled | 29.65 | 27.37 (26.88 / 27.86) | -2.28 (-2.77 / -1.79) | 0.69 | 0/2 | [-3.36, -1.23] | 0.14 | 0 / 0 |
| SGD | biased mean | 25.49 | 22.42 (22.03 / 22.82) | -3.07 (-3.47 / -2.67) | 0.56 | 0/2 | [-3.87, -2.21] | 0.17 | 0 / 0 |
| Adam | shuffled | 29.65 | 24.08 (24.33 / 23.82) | -5.57 (-5.32 / -5.83) | 0.36 | 0/2 | [-6.67, -4.45] | 0.00 | 0 / 0 |
| Adam | biased mean | 26.10 | 20.54 (20.83 / 20.26) | -5.56 (-5.28 / -5.84) | 0.40 | 0/2 | [-6.45, -4.71] | 0.06 | 0 / 0 |
| AdamW | shuffled | 29.69 | 24.30 (24.32 / 24.27) | -5.40 (-5.37 / -5.42) | 0.03 | 0/2 | [-6.75, -3.97] | 0.00 | 0 / 0 |
| AdamW | biased mean | 26.08 | 21.37 (21.14 / 21.60) | -4.71 (-4.94 / -4.49) | 0.32 | 0/2 | [-5.48, -3.96] | 0.07 | 0 / 0 |
| RMSprop | shuffled | 29.98 | 28.21 (28.54 / 27.87) | -1.77 (-1.44 / -2.11) | 0.47 | 0/2 | [-2.58, -0.91] | 0.15 | 0 / 0 |
| RMSprop | biased mean | 27.30 | 26.32 (26.62 / 26.02) | -0.98 (-0.67 / -1.28) | 0.43 | 0/2 | [-1.54, -0.42] | 0.29 | 0 / 0 |

| Optimizer | Tuned fixed configuration | Selected checkpoint (338 / 339) | Failed training episodes (338 / 339, of 4000 each) | Failed dev episodes | Train minutes (338 / 339) |
|---|---|---:|---:|---:|---:|
| SGD | lr=0.03, momentum=0.25, weight_decay=0.0001 | 100 / 300 | 7 / 6 | 0 / 0 | 102 / 101 |
| Adam | lr=0.001, weight_decay=0.0, beta1=0.5, beta2=0.9 | 400 / 500 | 4 / 3 | 0 / 0 | 152 / 154 |
| AdamW | lr=0.001, weight_decay=0.01, beta1=0.5, beta2=0.9 | 400 / 500 | 1 / 0 | 0 / 0 | 149 / 149 |
| RMSprop | lr=0.0003, weight_decay=0.0, momentum=0.5, alpha=0.9 | 400 / 500 | 26 / 34 | 0 / 0 | 151 / 151 |

All-order means (fixed / policy, pp difference): SGD 25.91 / 22.92 (−2.99), Adam 26.46 / 20.90 (−5.56), AdamW 26.44 / 21.66 (−4.78), RMSprop 27.56 / 26.51 (−1.06).

Per-ordering differences are in `optimizer_families/summary.csv` and the heatmap `optimizer_families/comparison.png`. Every one of the 40 family × ordering cells is negative. The largest losses are SGD `single_early` (−8.35 pp) and Adam `single_late` (−9.36 pp). The smallest is RMSprop `single_early` (−0.12 pp).

**Failures:**
- Test: no episodes failed (0 of 2,400 records).
- Training: 81 of 32,000 episodes failed and were retained. RMSprop accounts for 60 of them (26 + 34), compared with 4 on Fashion-MNIST.
- Development: no checkpoint-evaluation episodes failed.

## Caveats and observations (descriptive only)

- **Low accuracy is expected.** A 64-unit MLP on raw CIFAR-10 pixels with 200 updates of batch 64 (12,800 examples, about a third of an epoch) reaches roughly 25–30% accuracy. The question here is policy relative to fixed, not absolute accuracy.
- **Development data already favored fixed.** On the same 8 development episodes used for tuning, every selected policy checkpoint scores worse than the tuned fixed configuration. Score is log final loss + mean log path loss; lower is better.

  | Optimizer | Selected policy checkpoint (338 / 339) | Tuned fixed |
  |---|---:|---:|
  | SGD | 1.498 / 1.482 | 1.474 |
  | Adam | 1.539 / 1.553 | 1.441 |
  | AdamW | 1.549 / 1.560 | 1.440 |
  | RMSprop | 1.488 / 1.489 | 1.444 |

  The test result is therefore not an artifact of checkpoint selection.
- **Policies chose much smaller learning rates than the tuned fixed configurations.** On test traces, Adam and AdamW policies sit at the action space's 1e-4 learning-rate floor in 74% and 72% of decisions. Their median is 1e-4, against the fixed configuration's 1e-3. SGD's median policy LR is 0.007 (fixed 0.03), and RMSprop's is 1.8e-4 (fixed 3e-4). The policy's training reward is improvement in control-probe loss. We have not tested why training converged to these settings, so no causal claim is made.
- **Scope:**
  - Two policy seeds only, as in Fashion-MNIST.
  - One architecture.
  - Unchanged MNIST-derived hyperparameter ranges, initial actions, observation scaling, and 200-step episode length, none retuned for CIFAR-10.
  - A fixed baseline tuned on 288 (144 for SGD) candidates, against a policy trained with 4,000 episodes.

  These results describe this transfer of the unchanged protocol, not policy control on CIFAR-10 in general.
- **Threads:** see Design. The Fashion-MNIST run used two threads for SGD, this one used one.

## Files

- `recovery_study/`: SGD policies (`gradient_338/`, `gradient_339/`), SGD tuning (`baseline_tuning.csv`, `baseline_configs.json`), SGD evaluation/traces, `test_streams.json`, `checks.json`.
- `optimizer_families/`: Adam/AdamW/RMSprop policies, tuning and evaluation. Also the imported SGD records, `summary.csv`, `aggregates.csv` and `comparison.png`.
- `policy_vs_fixed.{csv,md}`: the seed-level tables above (CSV also has `all`-order rows).
