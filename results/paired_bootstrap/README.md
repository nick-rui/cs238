# Saved-result paired bootstrap

Run `.venv/bin/python scripts/paired_bootstrap.py` from any directory. CPU NumPy only;
no training, evaluation, or modification of existing results. Inputs are read from the
working tree when present, otherwise from the specified PR refs via git show.
inputs.json records path, source_ref, commit_sha, and content SHA256; for working-tree
inputs commit_sha is HEAD context, while the content hash identifies the actual bytes.

Each difference is accuracy(A) minus accuracy(B), in percentage points. Pair strictly
by condition and episode seed, and by policy seed when both methods are seeded.
Missing/duplicate/nonfinite pairs raise errors. Fixed baselines broadcast across policy
seeds. Failed episodes retain their stored accuracy; failed_a/b count original failed
rows (not broadcast copies). There is no filtering or imputation.

10,000 percentile bootstrap resamples use NumPy default_rng seed 238 for each row.
Resample policy seeds with replacement, then episode seeds with replacement, sharing
the episode draw across selected policy seeds. Aggregate biased_mean excludes mixed;
all includes mixed. Conditions have equal weight; seed draws are common across conditions
and episode draws are shared for identical episode-seed grids, independent otherwise.
n_episodes counts condition/episode pairs, not policy-seed repetitions. Win rate is the
fraction of paired differences >0, with ties half; aggregate rates equally weight
conditions. seeds_beating counts strictly positive per-policy-seed condition-mean differences.
p_le_0 is the fraction of bootstrap means <=0, not a calibrated hypothesis-test p-value.

Regular and frozen extreme policy comparisons use five seeds (338–342). Open-loop and
ablations use paired seeds 338/339, as do extreme-retrained comparisons. **Two-seed
bootstrap distributions have only two independent policy seeds and cannot robustly
characterize training-seed uncertainty.** Fixed-vs-fixed rows have one deterministic
seed slot (n_policy_seeds=1), with uncertainty from episodes only. CIs are marginal,
not adjusted for multiple comparisons, and conditional on saved checkpoints and tuning.
Compute-matched per-episode evaluations ARE saved for both suites, so both requested
comparisons are included, against original saved fixed results (not old_replayed).
No requested saved-data comparison is skipped; ablations only exist for SGD/Adam,
open-loop only for regular, and retrained policies only have two seeds.

The extreme-suite RMSprop reproduction check already fails on unchanged main
(accuracy difference ~0.0061, `scripts/check_reproduction.py --suite extreme`).
The check is not relaxed; this analysis uses saved numbers and is unaffected.
PR4 summary.csv five-seed policy means are checked automatically to tolerance 1e-10 pp.

## Key results

Positive differences favor the first method; full win rates, failure counts, seed counts,
and bootstrap tail fractions are in bootstrap.csv.

| Comparison | Suite | Family | Biased mean pp [95% CI] | Mixed pp [95% CI] |
|---|---|---|---:|---:|
| policy_vs_fixed | regular | SGD | +1.246 [+0.770, +1.752] | +0.214 [-0.224, +0.671] |
| policy_vs_compute_fixed | regular | SGD | +1.250 [+0.774, +1.752] | +0.219 [-0.217, +0.676] |
| compute_fixed_vs_fixed | regular | SGD | -0.004 [-0.043, +0.035] | -0.006 [-0.021, +0.004] |
| policy_vs_fixed | regular | Adam | -1.199 [-1.810, -0.625] | +0.088 [-0.320, +0.520] |
| policy_vs_compute_fixed | regular | Adam | -1.201 [-1.811, -0.630] | +0.089 [-0.318, +0.521] |
| compute_fixed_vs_fixed | regular | Adam | +0.002 [-0.006, +0.012] | -0.001 [-0.004, +0.000] |
| policy_vs_fixed | regular | AdamW | -1.161 [-1.847, -0.471] | +0.039 [-0.426, +0.534] |
| policy_vs_compute_fixed | regular | AdamW | -1.163 [-1.846, -0.475] | +0.041 [-0.424, +0.536] |
| compute_fixed_vs_fixed | regular | AdamW | +0.002 [-0.006, +0.012] | -0.001 [-0.004, +0.000] |
| policy_vs_fixed | regular | RMSprop | +0.202 [+0.096, +0.306] | +0.367 [+0.200, +0.523] |
| policy_vs_compute_fixed | regular | RMSprop | +0.204 [+0.099, +0.309] | +0.365 [+0.197, +0.521] |
| compute_fixed_vs_fixed | regular | RMSprop | -0.003 [-0.006, -0.000] | +0.002 [+0.000, +0.006] |
| policy_vs_fixed | extreme | SGD | +4.106 [+2.679, +5.421] | +0.519 [-0.013, +1.067] |
| policy_vs_compute_fixed | extreme | SGD | +4.068 [+2.605, +5.429] | +0.530 [-0.003, +1.075] |
| compute_fixed_vs_fixed | extreme | SGD | +0.039 [-0.149, +0.235] | -0.010 [-0.031, +0.000] |
| policy_vs_fixed | extreme | Adam | -17.309 [-25.454, -9.810] | +0.548 [+0.162, +0.945] |
| policy_vs_compute_fixed | extreme | Adam | -17.309 [-25.455, -9.811] | +0.556 [+0.168, +0.956] |
| compute_fixed_vs_fixed | extreme | Adam | +0.000 [+0.000, +0.001] | -0.008 [-0.019, +0.000] |
| policy_vs_fixed | extreme | AdamW | -14.323 [-23.379, -6.619] | +0.491 [+0.061, +0.949] |
| policy_vs_compute_fixed | extreme | AdamW | -14.323 [-23.380, -6.619] | +0.499 [+0.068, +0.961] |
| compute_fixed_vs_fixed | extreme | AdamW | +0.000 [+0.000, +0.001] | -0.008 [-0.019, +0.000] |
| policy_vs_fixed | extreme | RMSprop | +0.563 [-0.320, +1.242] | +0.272 [+0.109, +0.421] |
| policy_vs_compute_fixed | extreme | RMSprop | +0.559 [-0.325, +1.238] | +0.273 [+0.111, +0.422] |
| compute_fixed_vs_fixed | extreme | RMSprop | +0.004 [-0.001, +0.012] | -0.001 [-0.012, +0.009] |
| policy_vs_open_loop | regular | SGD | +0.465 [+0.291, +0.646] | +0.084 [-0.026, +0.169] |
| open_loop_vs_fixed | regular | SGD | +0.297 [-0.028, +0.629] | -0.103 [-0.614, +0.406] |
| lr_only_vs_fixed | regular | SGD | +0.124 [-1.220, +1.125] | -0.136 [-0.425, +0.167] |
| policy_vs_lr_only | regular | SGD | +0.639 [-0.185, +1.836] | +0.117 [-0.351, +0.539] |
| fixed_memory_vs_fixed | regular | SGD | -0.139 [-1.421, +1.206] | +0.470 [-0.550, +1.460] |
| policy_vs_fixed_memory | regular | SGD | +0.901 [-0.656, +2.369] | -0.489 [-1.694, +0.763] |
| no_feedback_vs_fixed | regular | SGD | +0.788 [+0.340, +1.266] | +0.342 [-0.486, +1.180] |
| policy_vs_no_feedback | regular | SGD | -0.026 [-0.283, +0.155] | -0.362 [-0.860, +0.001] |
| retrained_vs_retuned_fixed | extreme_retrained | SGD | +5.994 [+3.791, +8.251] | +3.360 [+2.368, +4.391] |
| policy_vs_open_loop | regular | Adam | +0.467 [-0.236, +1.222] | +0.025 [-0.023, +0.073] |
| open_loop_vs_fixed | regular | Adam | -1.282 [-1.875, -0.750] | -0.142 [-0.565, +0.293] |
| lr_only_vs_fixed | regular | Adam | +1.112 [+0.809, +1.410] | +1.421 [+1.085, +1.774] |
| policy_vs_lr_only | regular | Adam | -1.927 [-2.167, -1.681] | -1.538 [-1.818, -1.233] |
| fixed_memory_vs_fixed | regular | Adam | +1.051 [+0.797, +1.330] | +1.161 [+0.690, +1.651] |
| policy_vs_fixed_memory | regular | Adam | -1.866 [-2.120, -1.614] | -1.278 [-1.525, -1.016] |
| no_feedback_vs_fixed | regular | Adam | -2.092 [-2.657, -1.547] | -0.344 [-0.835, +0.202] |
| policy_vs_no_feedback | regular | Adam | +1.277 [+0.526, +2.008] | +0.227 [-0.261, +0.676] |
| retrained_vs_retuned_fixed | extreme_retrained | Adam | +4.879 [+1.332, +7.981] | +1.479 [+1.293, +1.639] |
| policy_vs_open_loop | regular | AdamW | +0.431 [-0.231, +1.149] | +0.033 [-0.008, +0.075] |
| open_loop_vs_fixed | regular | AdamW | -1.420 [-1.920, -0.968] | -0.268 [-0.710, +0.167] |
| retrained_vs_retuned_fixed | extreme_retrained | AdamW | +5.417 [+1.785, +8.721] | +1.385 [+1.204, +1.528] |
| policy_vs_open_loop | regular | RMSprop | +0.034 [-0.043, +0.103] | +0.127 [-0.061, +0.319] |
| open_loop_vs_fixed | regular | RMSprop | +0.162 [+0.048, +0.267] | +0.237 [+0.053, +0.407] |
| retrained_vs_retuned_fixed | extreme_retrained | RMSprop | +3.247 [+1.567, +4.999] | +0.912 [+0.578, +1.237] |
