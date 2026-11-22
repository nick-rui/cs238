# Policy-Controlled Optimizers

CS238 experiments on learning optimizer hyperparameters from feedback. A small policy chooses learning rate, weight decay, and optimizer-specific memory coefficients every five classifier updates. Gradient features are included by default.

The repository contains the implementation used for the original MNIST ordering study and its stronger ordering stress tests, selected policy checkpoints, tuned fixed configurations, evaluation records, and plotting scripts. The experiment code has been organized into portable source and results directories; the numerical update rules and saved policy weights are preserved.

## Final report and experiment index

The six-page final report is [`reports/final_report/final_report.pdf`](reports/final_report/final_report.pdf) (source: `main.tex`). Every table cell is traced to a results file in [`table_number_sources.md`](reports/final_report/table_number_sources.md). All experiments below are merged into `main`.

| Experiment | Runner | Results |
|---|---|---|
| Original SGD suite | `experiments/recovery_study/study.py` | `results/recovery_study/` |
| Original Adam/AdamW/RMSprop suites | `experiments/optimizer_families/run.py` | `results/optimizer_families/` |
| Frozen extreme orderings | `experiments/extreme_orderings/run.py` | `results/extreme_orderings/` |
| Policy seeds 340–342 | `scripts/additional_seeds.py` | [`results/additional_seeds/`](results/additional_seeds/README.md) |
| Compute-matched fixed search | `--grid wide` on both tuners | [`results/compute_matched_baseline/`](results/compute_matched_baseline/README.md) |
| Open-loop mean schedules | `experiments/open_loop/run.py` | [`results/open_loop/`](experiments/open_loop/README.md) |
| Prospective (dev-fitted) schedules | `experiments/prospective_schedules/run.py` | [`results/prospective_schedules/`](results/prospective_schedules/README.md) |
| Hand-written reactive rule | `experiments/reactive_rule/run.py` | [`results/reactive_rule/`](results/reactive_rule/README.md) |
| Control ablations (tuned-held) | `experiments/ablations/run.py` | `results/ablations/` |
| Control ablations (start-held) | `experiments/ablations/run.py --held-source start` | [`results/ablations_start_values/`](results/ablations_start_values/summary.md) |
| Warm-start policies | `experiments/warm_start/run.py` | [`results/warm_start/`](results/warm_start/README.md) |
| Extreme-ordering retraining (2 seeds) | `experiments/extreme_orderings/retrain.py` | `results/extreme_retrained/` |
| Extreme-ordering retraining (5 seeds) | `experiments/extreme_orderings/launch_retrain.py --five-seeds` | [`results/extreme_retrained_five_seeds/`](results/extreme_retrained_five_seeds/RESULTS.md) |
| Held-out ordering types | `experiments/extreme_orderings/launch_held_out.py` | [`results/extreme_held_out/`](results/extreme_held_out/README.md) |
| Warm starts on extreme orderings | `experiments/extreme_warm_start/experiment.py` | [`results/extreme_warm_start/`](results/extreme_warm_start/README.md) |
| Paired bootstrap intervals | `scripts/paired_bootstrap.py` | [`results/paired_bootstrap/`](results/paired_bootstrap/README.md) |
| Fashion-MNIST and CIFAR-10 timing | `CS238_DATASET=fashion_mnist`, `scripts/time_cifar10_episode.py` | `results/fashion_mnist/`, `results/cifar10_timing.json` |
| Clean held-out test (6,000 unused training-split images) | `experiments/clean_test/run.py` | [`results/clean_test/`](results/clean_test/README.md) |
| Scrambled-feedback control | `experiments/scrambled_feedback/run.py` | [`results/scrambled_feedback/`](results/scrambled_feedback/README.md) |
| Bootstrap CIs for later experiments | `scripts/paired_bootstrap_extended.py` | [`results/paired_bootstrap_extended/`](results/paired_bootstrap_extended/README.md) |

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

On the CPU experiment VM, the extreme replay has a pre-existing mismatch on the original RMSprop fixed configuration (`single_late`, episode 1920000, 0.61 percentage points absolute accuracy difference). The strict check is unchanged; the held-out study runs all comparators fresh in the same environment instead of mixing fresh and committed evaluation rows.

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

## Replay saved mean schedules without feedback

For the feedback-removal control (no retraining, original results preserved),
see [open-loop schedule replay](experiments/open_loop/README.md). It freezes one
pooled mean schedule per saved policy and evaluates all eight on the regular suite:

```bash
python experiments/open_loop/run.py evaluate
python experiments/open_loop/run.py summarize
python experiments/open_loop/check.py
```

## Train and tune from scratch

Each policy trains for 500 outer updates with eight episodes per update. The original seeds are 338 and 339; seeds 340–342 are trained by `scripts/additional_seeds.py` (see [additional seeds](results/additional_seeds/README.md)). Completed training runs are reused unless `--force` is supplied.

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

## Fashion-MNIST and CIFAR-10 CPU timing

Set `CS238_DATASET=fashion_mnist` to use Fashion-MNIST with the same classifier,
split sizes, regular orderings, and policy-training seeds (338/339). The default
is `mnist`; existing MNIST artifacts are unchanged. Fashion-MNIST outputs are
isolated under `results/fashion_mnist/{recovery_study,optimizer_families}/`.
Using the repository virtual environment, a complete sequential run is:

```bash
export CS238_DATASET=fashion_mnist CUDA_VISIBLE_DEVICES=''
# Download once before launching any jobs in parallel.
.venv/bin/python -c 'import sys; sys.path.insert(0,"experiments/recovery_study"); import engine as e; e.MNIST(str(e.DATA_DIR),train=False,download=True)'
.venv/bin/python experiments/recovery_study/study.py tune
.venv/bin/python experiments/optimizer_families/run.py tune --threads 1
.venv/bin/python experiments/recovery_study/study.py train
.venv/bin/python experiments/optimizer_families/run.py train --threads 1
.venv/bin/python experiments/recovery_study/study.py evaluate
.venv/bin/python experiments/optimizer_families/run.py evaluate --threads 1
.venv/bin/python experiments/optimizer_families/run.py import
.venv/bin/python scripts/analyze_regular.py
```

Training does not depend on tuning: independent jobs may overlap within an
eight-thread CPU budget (SGD train/evaluate use two threads; family jobs and
SGD tuning use one). Each evaluation needs its own tuned baseline and both
trained policies; import and analysis follow all evaluations. Do not launch
duplicate writers for the same job. The completed run includes eight policies,
four tuned baselines, and 600 test records per optimizer (20 paired episode
seeds × 10 orderings × fixed/two policy seeds). All test episodes succeeded;
23 of 32,000 training episodes were marked failed by the existing safeguards
and retained, not dropped.

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/time_cifar10_episode.py`
downloads CIFAR-10 into ignored `data/` and times the existing batched engines
on MNIST, Fashion-MNIST, and CIFAR-10. CIFAR-10 uses a flattened 3072→64→10 MLP,
not a CNN, with a 40k/2k/2k train/control/dev split. One eight-classifier training
rollout measured 5.93 s for SGD (two threads) and 17.18 s for Adam (one thread),
versus MNIST's 4.34/10.45 s. The saved `results/cifar10_timing.json` projects
about 4.33 hours for four optimizers on eight cores: 87 min tuning, 148 min
training, 25 min evaluation. This single-sample estimate proxies AdamW/RMSprop
with Adam and excludes contention and I/O; allow roughly 5–7 hours. The script
refuses to overwrite its saved JSON. Current grids have 144 SGD and 288
candidates per other optimizer; older timing notes used different grid counts.

Run reproduction checks with `CS238_DATASET` unset. The regular suite passes.
The extreme suite has a documented pre-existing RMSprop fixed-baseline mismatch
on main: accuracy absolute error about 0.00610006 (0.61 percentage points), loss
error 0.02667534, and path-log-loss error 0.01941165. The assertion and tolerances
are unchanged; `--suite extreme` is expected to fail at that record.

### Generalization to held-out ordering types

For each of the eight extreme orderings, train on the other seven and select checkpoints and a retuned fixed configuration on those seven types only. Test the two retrained policies, two original frozen policies, original fixed configuration and fold-retuned fixed configuration on identical held-out episodes. The all-order and biased-only aggregates weight ordering types equally. See [the held-out results](results/extreme_held_out/README.md).

```bash
# Quick pipeline check, isolated from the full output.
python experiments/extreme_orderings/launch_held_out.py --smoke --workers 2
# Full CPU run (64 policy trainings, several hours); resumes completed jobs.
python experiments/extreme_orderings/launch_held_out.py --workers 8
# Validate saved folds and regenerate their summaries without retraining.
python experiments/extreme_orderings/held_out.py analyze
```

The study writes only to `results/extreme_held_out/`. To rerun from scratch, start with an empty output directory instead of the committed results. A single fold can be run with `held_out.py train --family SGD --held-out class_sorted --seed 338`; `tune --family SGD` tunes all folds for that optimizer, and `evaluate --family SGD --held-out class_sorted` needs both trained seeds and the tuned configuration.

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

Use `--held-source start` to rerun `lr_only` and `fixed_memory` with the
full rollout's step-0 settings **before the first policy action**: SGD momentum
0.475, Adam betas (0.9, 0.999), and weight decay 0.0001 for both LR-only policies.
In `fixed_memory`, weight decay is still learned. This runs both families with
policy seeds 338/339 and writes only to `results/ablations_start_values/`.

```bash
python experiments/ablations/run.py train --held-source start --threads 1
python experiments/ablations/run.py evaluate --held-source start --threads 1
python experiments/ablations/analyze.py --held-source start
```

The [start-held summary](results/ablations_start_values/summary.md) pairs the
new regular-suite results against both the full policy and the corresponding
tuned-held ablation by episode seed, ordering, and policy seed.

The regular reproduction check passes. The extreme check still fails on the
fixed RMSprop replay (absolute accuracy error 0.00610006), with the same
mismatch on unchanged `main`; the check and its thresholds are unchanged.

### Compute-matched fixed-baseline search

Both tuners accept `--grid wide --output-dir results/compute_matched_baseline`;
the default grids are unchanged. The wider grids evaluate 504 SGD configurations
(4,032 episodes) and 480 configurations per other optimizer (3,840 episodes),
always on the original eight development episodes. Isolated evaluation requires
`--fixed-only --config-path PATH --output-dir DIR` (and `--family` for family/stress runners).
See [commands, replay caveats and results](results/compute_matched_baseline/README.md).
Old and new fixed configurations are replayed with one Torch thread on both suites;
policy numbers reuse the saved records, not newly evaluated policies. Old SGD
`single_middle` replay drifts by about -0.02 percentage points in mean accuracy:
seed 1900006 matches saved accuracy with two threads but not one. Therefore the
comparison includes both saved and replayed old baselines.

### Warm-start policies

`experiments/warm_start/run.py` reuses the original training loops with the tuned
fixed configuration as the controller's initial state. All new outputs live in
`results/warm_start/`; existing evaluations and checkpoints are read-only inputs.
The optional `warm` engine argument leaves the original path unchanged when omitted.
Observation normalizers are unchanged. The relative learning-rate action starts
with bias zero; other output biases invert the action transform, clipping the
inverse-tanh logits to [-3, 3]. Thus boundary targets (including zero weight decay)
are approximated rather than represented exactly. Initial states are clamped to
the action ranges (SGD's zero weight decay becomes 1e-6).

```bash
.venv/bin/python experiments/warm_start/run.py sanity --threads 1
.venv/bin/python experiments/warm_start/run.py train --family Adam --seed 338 --threads 1
.venv/bin/python experiments/warm_start/run.py evaluate --family Adam --suite regular --threads 1
.venv/bin/python experiments/warm_start/run.py evaluate --family Adam --suite extreme --threads 1
.venv/bin/python experiments/warm_start/run.py analyze
```

Train each of SGD, Adam, AdamW, RMSprop with seeds 338–342 (one torch thread per
job, at most eight concurrent jobs, after downloading MNIST once). The original
500 updates, eight episodes per update, episode seeds, development checkpoints,
and best-by-development-score selection are retained. Evaluation pairs all five
warm policies with freshly evaluated fixed configurations and original full
policies 338/339, using the original episode batches and thread counts (two for
regular SGD, one otherwise; omit `--threads` to select these defaults). Drift CSVs
compare these controls to saved results. Summary CSVs report accuracies, paired
mean differences, and counts of policy seeds whose paired mean difference is
positive (five seeds against fixed; two matched seeds against full).
The extreme summary includes all eight extreme-suite orders, including its mixed
reference. Warm-minus-full uses only matched policy seeds 338/339, so it need not
equal the difference of the displayed five-seed warm and two-seed full means.

The extreme reproduction check has an existing RMSprop fixed-baseline mismatch
on unchanged main: accuracy differs by approximately 0.0061. The check is not
relaxed; see `results/warm_start/reproduction_extreme_main.log` for the replay.
Full-suite re-evaluation also exposes larger pre-existing saved-result drift,
including SGD extreme accuracy differences up to 0.098. Selected worst cases
replayed on unchanged main reproduce the new numbers exactly at one thread;
using two threads does not restore the saved numbers. See
`results/warm_start/drift_probe_main.json`. All comparisons in the new summary
use freshly paired evaluations, not the drifting saved reference numbers.

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

Mean final accuracy (%) on the seven stronger biased orderings, equally weighted, for the original two policy seeds. The final report's Table I gives the five-seed numbers, and later sections cover retuned baselines, warm starts and retraining:

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
- `experiments/extreme_orderings/`: stronger stream generator, frozen evaluation, retraining and held-out evaluations.
- `experiments/open_loop/`, `experiments/ablations/`, `experiments/warm_start/`: feedback-removal, restricted-control and warm-start studies.
- `scripts/`: original-suite summaries, momentum plot, reproduction checks, additional seeds, compute-matched analysis, paired bootstrap, CIFAR-10 timing.
- `reports/`: project update and final report.
- `results/`: selected policy weights, development curves, tuned configurations, evaluation CSVs, compressed traces and verification records.

Small `best.pt` files contain policy weights and checkpoint metadata, not trained MNIST classifiers. Their hashes are recorded in `results/checkpoint_sha256.json`.

## Fifth experiment: extreme-ordering retraining

`experiments/extreme_orderings/retrain.py` trains on the eight extreme ordering
types with new episode seeds, selects on extreme development episodes, and
evaluates on the saved extreme suite's official 10k test set. This is
in-distribution for ordering types, not a held-out-ordering generalization test.
The regular-tuned saved fixed baseline is disadvantaged; a separately retuned
fixed row provides a fairer comparison. Saved frozen evaluations are read only.

```bash
# Repeat train for seeds 338/339 and all four families; tune once per family.
.venv/bin/python experiments/extreme_orderings/retrain.py train --family SGD --seed 338
.venv/bin/python experiments/extreme_orderings/retrain.py tune --family SGD
.venv/bin/python experiments/extreme_orderings/retrain.py evaluate --family SGD
.venv/bin/python experiments/extreme_orderings/retrain.py analyze
# Or run the complete train/tune/evaluate/analyze pipeline (8 workers maximum):
mkdir -p results/extreme_retrained/logs
nohup .venv/bin/python experiments/extreme_orderings/launch_retrain.py > results/extreme_retrained/logs/launcher.log 2>&1 &
```

All outputs go to `results/extreme_retrained/`. Add `--smoke` for two
training updates, one tuning candidate, and one test episode per ordering
(isolated under `smoke/`; train both policy seeds before smoke evaluation).
Each process uses one torch CPU thread. Summary CSVs separate policy seeds;
paired differences use episode-matched saved fixed results and report standard
errors and wins out of 20 (averaging orderings within seed for aggregate rows).

### Five-seed retraining extension

The extension trains seeds 340–342 with the same 500-update schedule and reuses
the saved retrain runs for 338–339 and both fixed baselines. Frozen controllers
are matched by policy seed, including the evaluations from
[the additional-seed experiment](results/additional_seeds/README.md).
The original results are read-only.

```bash
# Rerun the analysis from the committed evaluations and frozen-control rows.
.venv/bin/python experiments/extreme_orderings/summarize_retrain.py

# For a fresh run without the cached frozen-control rows, point at the merged
# additional-seed evaluations.
.venv/bin/python experiments/extreme_orderings/launch_retrain.py --five-seeds \
  --frozen-results results/additional_seeds/extreme
```

The launcher skips completed work. It uses at most eight one-thread workers
(`--workers` can lower this). Add `--smoke` to test the pipeline separately
with two new training updates and one test episode per ordering. It also takes
the archived frozen-control directory.

See [the five-seed results](results/extreme_retrained_five_seeds/RESULTS.md).
The summary reports sample SD and range across policy seeds, and comparisons
count wins by policy seed after averaging matched orderings and test seeds.
