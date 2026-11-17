# Additional policy seeds

This isolated extension adds policy seeds 340, 341 and 342 to all four optimizer
families. Original results are inputs only and are not overwritten. Existing CLI
defaults remain unchanged.

Run from the repository root with its CPU Python environment:

```bash
nohup .venv/bin/python -u scripts/additional_seeds.py pipeline \
  --seeds 340 341 342 --workers 4 \
  > results/additional_seeds/pipeline.log 2>&1 < /dev/null &
```

The pipeline trains 12 policies with four concurrent one-thread processes, then
evaluates them on both suites, replays representative evaluations, summarizes,
and validates. Training is unchanged: 500 outer updates, eight episodes per
update, and development selection at 0/25/50/100/200/300/400/500. It calls the
original training functions. Completed jobs are reused; interrupted jobs restart
from their beginning. Re-run the same command to resume. Do not run two copies
against the same output directory.

Regular evaluations use episode seeds 1900000–1900019 and all ten original
conditions. Extreme evaluations use seeds 1920000–1920019 and all eight original
conditions, including `mixed`. Evaluation files contain new policy rows only;
summary tables reuse original fixed rows without retuning or re-evaluating them.
The test set, initial classifier, balanced example bags, feedback and evaluation
metrics are the original study's. New episode seeds are not a new test dataset.

Layout:

- `training/{family}_gradient_{seed}/`: selected policy, development curve,
  completion metadata; intermediate checkpoints and histories remain ignored.
- `{regular,extreme}/{family}_{seed}/`: evaluations, compressed action traces,
  timings, and representative numerical/action-trace replay checks.
- `summary.csv`: each condition's policy accuracy mean and sample standard
  deviation across five policy seeds (338–342), fixed accuracy, difference in
  percentage points, and number of policy seeds beating fixed.
- `policy_seed_summary.csv`: episode-averaged accuracy and failures by policy
  seed. `biased_mean` equally weights all non-mixed conditions, since each has
  the same number of episodes. It is not a significance test.
- `original_sha256.json`: original tracked result hashes; `checks.json` verifies
  those hashes and every new evaluation/trace grid.
- `design.json`, `pipeline_done.json`: protocol and wall-clock timings.

Independent commands:

```bash
.venv/bin/python scripts/additional_seeds.py train --family SGD --seed 340
.venv/bin/python scripts/additional_seeds.py evaluate --family SGD --seed 340 --suite regular
.venv/bin/python scripts/additional_seeds.py replay --family SGD --seed 340 --suite regular
.venv/bin/python scripts/additional_seeds.py summarize
.venv/bin/python scripts/additional_seeds.py validate
.venv/bin/python -m unittest discover -s scripts -p 'test_additional_seeds.py'
```

For smoke tests only, train with `--updates 1 --out-dir /path/outside/results`
and evaluate with `--smoke` in that same directory. Never reuse smoke artifacts
for a production run. Production validation requires all 500 updates.
