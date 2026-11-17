# Policy-Controlled Optimizers

CS238 experiments on learning optimizer hyperparameters from feedback. A small policy chooses learning rate, weight decay, and optimizer-specific memory coefficients every five classifier updates. Gradient features are included by default.

The repository contains the implementation used for the original MNIST ordering study and its stronger ordering stress tests, selected policy checkpoints, tuned fixed configurations, evaluation records, and plotting scripts. The experiment code has been organized into portable source and results directories; the numerical update rules and saved policy weights are preserved.

## Setup

Tested with Python 3.12 on CPU. No GPU or external training service is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MNIST downloads automatically to `data/` on first use of an experiment module. To use an existing torchvision dataset cache, set `MNIST_DATA_DIR` to its parent directory containing `MNIST/`. Dataset files are not included in Git.

## Inspect the saved results

These commands use the included evaluations and compressed traces; no training is needed:

```bash
python scripts/analyze_regular.py
python scripts/plot_momentum.py
python experiments/extreme_orderings/analyze.py
```

Generated figures and aggregate CSVs are placed under `results/`. Trace readers use an uncompressed CSV if present, otherwise its committed `.csv.gz` counterpart. `gradient` in the original file names and record fields identifies the Policy-Controlled Optimizer.

## Validate

```bash
python experiments/recovery_study/study.py check
python experiments/optimizer_families/validate.py
python experiments/extreme_orderings/run.py check
python scripts/check_reproduction.py --suite regular
python scripts/check_reproduction.py --suite extreme
```

Checks cover exact example-multiset preservation and class balance, block membership, differentiable policy rollouts, native PyTorch optimizer equivalence, independent time-varying moment calculations, and replay of saved evaluations. The reproduction check evaluates each family's fixed configuration and both policy checkpoints on a representative condition and compares against the committed records.

## Re-evaluate the saved policies

These commands replace the corresponding evaluation CSVs and create fresh trace CSVs. They do not train policies or tune baselines.

```bash
# Original SGD suite, then import its records into the family comparison.
python experiments/recovery_study/study.py evaluate
python experiments/optimizer_families/run.py import

# Original Adam, AdamW and RMSprop suites.
python experiments/optimizer_families/run.py evaluate --force

# Stronger ordering suite: repeat for SGD, Adam, AdamW and RMSprop.
python experiments/extreme_orderings/run.py evaluate --family SGD --force
```

Regenerate summaries and figures using the analysis commands above. Omitting `--force` preserves completed family evaluations. The SGD recovery runner always regenerates its evaluation.

## Train and tune from scratch

Each policy trains for 500 outer updates with eight episodes per update. The included seeds are 338 and 339. Completed training runs are reused unless `--force` is supplied.

```bash
python experiments/recovery_study/study.py train --force
python experiments/recovery_study/study.py tune
python experiments/optimizer_families/run.py train --force
python experiments/optimizer_families/run.py tune --force
```

Then run the evaluation commands above. To run a single policy:

```bash
python experiments/recovery_study/study.py train --job gradient_338 --force
python experiments/optimizer_families/run.py train --job Adam_gradient_338 --threads 1 --force
```

`--family Adam` also restricts family training, tuning or evaluation. Run at most a few jobs concurrently on a laptop; these commands are sequential by default. Full retraining is substantially more expensive than evaluation.

## Policy-control ablations

`experiments/ablations/run.py` runs SGD and Adam with seeds 338/339 under the
unchanged 500-update, eight-episode training and development-selection protocol.
`lr_only` controls only LR, fixing memory and weight decay to the tuned baseline;
`fixed_memory` controls LR and weight decay; `no_feedback` keeps all actions but
removes the three gradient-feedback inputs. Held knobs remain observed. Policy
layers match the retained inputs/actions and retain the original initialization.
For SGD LR-only, the baseline's zero weight decay makes its log observation
negative infinity; this constant is mapped to -20 (the family engine's existing
nonfinite-observation convention). Other SGD observations and the full policy
path are unchanged.

```bash
python experiments/ablations/run.py smoke --threads 1
python experiments/ablations/validate.py
python experiments/ablations/run.py train --job SGD_lr_only_338 --threads 1
python experiments/ablations/run.py train --threads 1
python experiments/ablations/run.py evaluate --threads 1
python experiments/ablations/analyze.py
```

Run individual training jobs concurrently as appropriate (six one-thread jobs on
an eight-core machine). Completed jobs reuse `done.json`; smoke runs use a separate
`results/ablations/smoke/` tree with one update, never production checkpoints.
All new records live under `results/ablations/`; existing result files are read
only. Evaluation reuses each per-job CSV and combines complete families. The
summary compares all ten orderings against the committed full-policy and fixed
baseline rows, with paired variant-minus-full differences. Traces and training
records are gzip-compressed; intermediate checkpoints are ignored as elsewhere.

## Method

- **Classifier:** 784 → 64 → ReLU → 10 multilayer perceptron, MNIST, 200 updates of batch size 64.
- **Policy:** one 16-unit tanh hidden layer; Gaussian actions with standard deviation 0.5 during training, deterministic means at evaluation. SGD has 9 inputs and 3 outputs; the other families have 10 inputs and 4 outputs.
- **Observations:** current settings, elapsed time, relative control loss, recent loss improvement, gradient norm, cosine alignment with optimizer memory, and cosine alignment with the previous gradient. No class IDs, block boundaries, or future minibatches are supplied. The inspected minibatch gradient is reused for the next update.
- **Actions:** SGD `(learning rate, momentum, weight decay)`; Adam/AdamW `(learning rate, beta1, beta2, weight decay)`; RMSprop `(learning rate, momentum, alpha, weight decay)`. Adam uses coupled L2 regularization; AdamW uses decoupled decay.
- **Training:** REINFORCE with a leave-one-out return baseline within each stream regime; outer Adam at 0.003 and gradient clipping at 1. Each batch contains two episodes of each of four regimes: random, single-class burst, two-class burst, and two different-class bursts.
- **Reward:** `log(previous control loss) - log(current control loss) - log(current control loss)/40`. No reward explicitly requests a particular action pattern.
- **Adam normalization:** cumulative products of the historical beta coefficients, updated every classifier step. Using the current beta raised to the current step would not account for changing coefficients.

The 60,000-image training split is permuted with seed 238: 50,000 classifier-training examples, 2,000 control examples, and 2,000 development examples; the remaining 6,000 are unused. Each episode uses 512 control examples. Development selects policy checkpoints and fixed configurations using final log loss plus mean trajectory log loss. Final evaluation uses all 10,000 official test images; diagnostic trajectories use the first 1,000. Test metrics never provide policy feedback.

## Data orderings

Every episode contains exactly 1,280 draws per class (with replacement), 12,800 examples total. All paired methods and orderings share the exact example multiset and classifier initialization. Digits are permuted per episode.

The original suite includes random order, early/middle/late single-class bursts, a two-class burst, repeated bursts, gradual bias, and rapid bursts. See `experiments/recovery_study/study.py` for precise construction.

The stronger suite is defined in `experiments/extreme_orderings/run.py`:

| Condition | Biased segment | Shuffled recovery batches |
|---|---|---:|
| Maximum single class | One digit, batches 80–99 | 100 |
| Late single class | One digit, batches 175–194 | 5 |
| Maximum pair | Two digits, batches 140–179 | 20 |
| Half-class block | Five digits, batches 60–159 | 40 |
| Class staircase | Ten 16-batch single-class blocks, batches 20–179 | 20 |
| Fully sorted | Ten 20-batch single-class blocks | 0 |
| Alternating halves | Eight 25-batch blocks alternating disjoint five-class sets | 0 |

Batch indices are zero-based. Shuffled segments use the remaining bag, which may omit classes exhausted by a long block. A recovery phase therefore does not necessarily restore uniform class frequencies. These are structured shifts, not optimized adversarial attacks.

## Results and interpretation

Mean final accuracy (%) on the seven stronger biased orderings, equally weighted:

| Optimizer | Tuned fixed | Policy-controlled |
|---|---:|---:|
| SGD | 47.55 | 51.63 |
| Adam | 63.36 | 54.15 |
| AdamW | 63.36 | 55.74 |
| RMSprop | 67.00 | 68.14 |

Both policy seeds are included, with 20 paired episode seeds per condition. These stress tests freeze policies and baselines selected on the earlier mixture. They are not comparisons against baselines retuned for the stronger distribution. Individual conditions differ sharply: alternating halves helps policy-controlled SGD, Adam and AdamW, while the late single-class block severely hurts policy-controlled Adam and AdamW.

Other limitations: two policy-training seeds, one small network, short episodes, unequal total tuning/meta-training compute, different initial hyperparameters, and different retained failure handling for SGD. Policy action changes demonstrate reactivity but do not establish its causal benefit. Means and interquartile bands are descriptive, not significance tests. The test images were used in earlier exploratory experiments; new episode seeds do not make them a new test dataset.

## Layout

- `experiments/recovery_study/`: SGD rollout engine, balanced streams, policy training, fixed-SGD search and evaluation.
- `experiments/optimizer_families/`: Adam/AdamW/RMSprop implementations, policy training, baseline tuning and numerical checks.
- `experiments/extreme_orderings/`: stronger stream generator, frozen evaluation and analysis.
- `scripts/`: original-suite summaries, momentum plot, reproduction checks.
- `results/`: selected policy weights, development curves, tuned configurations, evaluation CSVs, compressed traces and verification records.

Small `best.pt` files contain policy weights and checkpoint metadata, not trained MNIST classifiers. Their hashes are recorded in `results/checkpoint_sha256.json`.
