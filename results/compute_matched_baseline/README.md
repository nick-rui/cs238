# Wider fixed-configuration baseline search

The selection protocol is unchanged: seeds 1810000–1810007, two episodes each of
mixed/single_train/pair_train/double_train, and mean final log development loss
plus mean trajectory log loss. Policies are frozen. All new searches and fixed
evaluations use one Torch thread. Wider grids are strict supersets of the old grids:

- SGD: LR {0.0003,0.001,0.002,0.003,0.006,0.01,0.02,0.03,0.06,0.1,0.2,0.3}
  × momentum {0,0.25,0.5,0.75,0.9,0.95,0.99}
  × decay {0,1e-6,1e-5,1e-4,1e-3,0.01}: 504 configurations, 4,032 episodes.
- Adam/AdamW/RMSprop: LR {1e-4,2e-4,3e-4,6e-4,1e-3,2e-3,3e-3,6e-3,0.01,0.03}
  × beta1/momentum {0,0.5,0.9,0.95} × beta2/alpha {0.9,0.99,0.999,0.9999}
  × decay {0,1e-4,0.01}: 480 configurations each, 3,840 episodes each.

## Reproduce

From the repository root, with the CPU requirements installed:

```bash
source .venv/bin/activate
B=results/compute_matched_baseline
python experiments/recovery_study/study.py tune --grid wide --threads 1 --output-dir "$B" &
for F in Adam AdamW RMSprop; do
  python experiments/optimizer_families/run.py tune --family "$F" --grid wide --threads 1 --output-dir "$B" &
done
wait

# Four independent fixed-only evaluation jobs for each pass; no policy evaluation.
for PASS in new old_replayed; do
  CONFIG="$B"; DEST="$B"
  if [ "$PASS" = old_replayed ]; then
    CONFIG=results/optimizer_families; DEST="$B/old_replayed"
  fi
  for F in SGD Adam AdamW RMSprop; do
    (
      if [ "$F" = SGD ]; then
        python experiments/recovery_study/study.py evaluate --threads 1 --fixed-only --config-path "$CONFIG/${F}_baseline.json" --output-dir "$DEST/regular"
      else
        python experiments/optimizer_families/run.py evaluate --family "$F" --threads 1 --fixed-only --config-path "$CONFIG/${F}_baseline.json" --output-dir "$DEST/regular"
      fi
      python experiments/extreme_orderings/run.py evaluate --family "$F" --threads 1 --fixed-only --config-path "$CONFIG/${F}_baseline.json" --output-dir "$DEST/extreme"
    ) &
  done
  wait
done
python scripts/analyze_compute_matched.py
python scripts/test_compute_matched.py
```

Family tuning/evaluation preserves completed outputs unless `--force` is supplied;
the SGD runner regenerates its outputs. Use a new output directory for a fresh run.
Trace CSVs are compressed after evaluation to match the repository convention.

## Records and interpretation

`*_baseline.json` contains the selected configuration, score, budget, development
seeds/orders and search time. Tuning CSVs contain all candidate/episode records;
candidate IDs follow the Cartesian-product order in the tuner. `regular/` and
`extreme/` contain new fixed evaluations; `old_replayed/` contains fresh evaluations
of the original fixed configs using exactly the same evaluation settings.
Original seeds are 1900000–1900019; stress seeds are 1920000–1920019.
Raw evaluation CSVs include final loss, accuracy and trajectory log loss. Existing
headline aggregates contain accuracy only, so the comparison retains that format.

`aggregates.csv` reports old fixed (saved), old fixed (replayed), new fixed, and
policy (saved). `fixed_improvement_pp` compares new against replayed old;
`difference_pp` is saved policy minus new fixed. `config_comparison.json` records
old/new configs and timing. Saved policies include both seeds 338 and 339;
the analyzer checks that every condition has the same paired episode seeds.
All group means weight conditions equally.

### Replay caveat and cheap cause check

On old SGD `single_middle`, saved mean accuracy is 89.1190001%, versus 89.0990001%
with one thread (about -0.02pp). Seed 1900006 gives 90.04% with one thread but the
saved 90.44% with two or four threads; deterministic-algorithm toggling does not
change this. Losses still need not agree bitwise. The current environment is
Python 3.12.16, torch 2.5.1+cpu, MKL 2024.2 and MKL-DNN 3.5.3, AVX2.
Saved metadata records one thread for family policy training but no comparable
evaluation torch/MKL provenance. Details are in `replay_investigation.txt`.

The existing regular reproduction check passes. The extreme check fails for
fixed RMSprop at seed 1920000, single_late, with accuracy error 0.0061000586,
loss error 0.0266753435 and trajectory-log-loss error 0.0194116533. Running the
unchanged base commit in a separate checkout produces the identical failure:
this is preexisting replay drift, not introduced by the runner changes. It was
not fixed or hidden by loosening checks. All old/new fixed comparisons therefore
include same-environment replays. Policy comparisons still use saved policy
records and are not strictly same-environment replays.

## Outcome

All four wide searches select the **same configurations as the old searches**.
Every new fixed evaluation record is identical to its fresh old-config replay.
Increasing configuration coverage did not improve development selection or
same-environment test accuracy on these grids. No policy was retrained;
only the explicitly requested reproduction checks replay representative policies.

| Optimizer | Selected config (unchanged) | Search seconds | Old replay seconds | New evaluation seconds |
|---|---|---:|---:|---:|
| SGD | lr=0.3, momentum=0.75, wd=0 | 998.011 | 134.831 | 119.693 |
| Adam | lr=0.01, beta1=0.5, beta2=0.99, wd=0 | 1866.669 | 230.141 | 204.491 |
| AdamW | lr=0.01, beta1=0.5, beta2=0.99, wd=0 | 1870.127 | 225.550 | 201.645 |
| RMSprop | lr=0.001, momentum=0.5, alpha=0.999, wd=0 | 2059.665 | 221.784 | 194.051 |

Searches ran concurrently on the 8-core CPU VM. Evaluation times sum both suites
per optimizer; old replays overlapped the searches. Total search budget is 15,552
episodes; old/new fixed evaluations add 2,880 episodes. Tuning CSV integrity checks
verify all candidate/seed/regime combinations and independently recover each argmin.

Mean final accuracy (%); original biased averages nine conditions, stress averages seven:

| Optimizer | Group | Old fixed saved | Old fixed replayed | New fixed | Policy saved |
|---|---|---:|---:|---:|---:|
| SGD | Original random | 92.2665 | 92.2610 | 92.2610 | 92.2473 |
| SGD | Original biased | 88.3638 | 88.3599 | 88.3599 | 89.1264 |
| SGD | Stress biased | 47.5545 | 47.5931 | 47.5931 | 51.6280 |
| Adam | Original random | 92.6330 | 92.6315 | 92.6315 | 92.5158 |
| Adam | Original biased | 91.5040 | 91.5064 | 91.5064 | 90.6890 |
| Adam | Stress biased | 63.3636 | 63.3640 | 63.3640 | 54.1522 |
| AdamW | Original random | 92.6330 | 92.6315 | 92.6315 | 92.3982 |
| AdamW | Original biased | 91.5040 | 91.5064 | 91.5064 | 90.5145 |
| AdamW | Stress biased | 63.3636 | 63.3640 | 63.3640 | 55.7352 |
| RMSprop | Original random | 93.1910 | 93.1930 | 93.1930 | 93.5553 |
| RMSprop | Original biased | 92.3703 | 92.3676 | 92.3676 | 92.5664 |
| RMSprop | Stress biased | 66.9984 | 67.0026 | 67.0026 | 68.1446 |

Validation outputs are saved as `focused_tests.txt`, `recovery_check.txt`,
`family_check.txt`, `extreme_check.txt`, `reproduction_regular.txt`, and
`reproduction_extreme{,_base}.txt`. All pass except the documented preexisting
extreme reproduction failure. The baseline source commit was
`1fa9c02fc39a331fa5b9ca84d1f3765db65e8366`.
