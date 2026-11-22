# Open-loop schedule control (regular suite)

This control asks whether each saved feedback policy beats its own **average
time-varying hyperparameter schedule**. It does not retrain any policy, tune
another baseline, or alter existing results.

## Definition

For each optimizer (SGD, Adam, AdamW, RMSprop) and policy seed (338, 339), take
the arithmetic mean of each actual hyperparameter setting at each of the 40
control steps in its saved regular-suite traces. Each mean pools all 10
conditions and all 20 episode seeds (200 trajectories), including failures if
present. The mean is of the settings, not raw actions, log learning rates, or
transformed moment coefficients.

Replay the resulting **single 40-step schedule per policy** unchanged on every
regular condition and episode seed. There is no condition-specific schedule,
policy invocation, or hyperparameter adjustment from control loss, gradient
features, or test metrics. SGD uses its existing schedule path; the other
families use a new optional `schedule` argument in `family_engine.rollout`.
Optimizer memory, historical beta products, classifier initialization, stream
construction, control interval, failure handling, and evaluation are unchanged.
Losses/gradients are still computed for optimization, diagnostics, and failure
handling, but never choose replay settings.

This is a **retrospective** control: its schedule averages the same saved
evaluation trajectories used for comparison. No test metric selects a schedule,
but this is not held-out schedule fitting or an optimized open-loop baseline.
If closed-loop wins, the result supports trajectory-specific adaptation over
this particular pooled mean; averaging nonlinear hyperparameters also changes
their distribution, so it is not a universal proof that feedback is necessary.
If replay matches or wins, these evaluations do not establish a feedback benefit.

## Run

From the repository root, with dependencies installed and MNIST available:

```bash
python experiments/open_loop/run.py evaluate
python experiments/open_loop/run.py summarize
python experiments/open_loop/check.py
```

`evaluate` runs all eight policies, 200 episodes each. It never loads a policy
for inference. Use `--family Adam --policy-seed 338` for one job, or run different
families concurrently. Default CPU threading is one per process (`--threads`).
Each job refuses to overwrite an existing directory, including a partial job;
use `--output results/open_loop/another_run` to repeat. Run `summarize` only after
the desired jobs finish; it also supports partial runs and reports policy counts.

All new artifacts are under `results/open_loop/regular/`:

- Per-policy `schedule.csv`, `evaluation.csv`, compressed `traces.csv.gz`, and
  `provenance.json` with input/checkpoint SHA-256 hashes and runtime metadata.
- `paired_differences.csv`: each replay matched to the same closed-loop policy
  seed, episode seed, and condition, with the existing fixed baseline alongside.
- `summary.csv`: equal-episode-weight means for all conditions, random order,
  and the nine biased conditions; `policy_seed_summary.csv` and
  `condition_summary.csv` preserve finer detail.
- `checks.json`: independently checked means and all replay settings; exact
  historical per-episode schedule replay for each of the eight policies;
  forbidden-policy checks proving replay does not call the policy.

Accuracy fields use fractions (multiply by 100 for percent/percentage points).
Differences are **closed-loop minus open-loop**; positive accuracy favors
closed-loop, while negative loss favors closed-loop. Two policy seeds are
descriptive replicates, not 400 independent policy-training runs.
