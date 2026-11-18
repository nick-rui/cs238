"""Paired hierarchical bootstrap using saved CSVs only (CPU NumPy)."""

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paired_bootstrap"
FAMILIES = ("SGD", "Adam", "AdamW", "RMSprop")
REFS = {
    "main": "origin/main",
    "additional": "origin/devin/1794943419-additional-policy-seeds",
    "open": "origin/devin/1794944000-open-loop-controls",
    "compute": "origin/devin/1794943599-compute-matched-baseline",
    "ablation": "origin/devin/1794943493-policy-ablations",
    "retrained": "origin/devin/1794955081-extreme-retrain",
}
N_RESAMPLES = 10000
RNG_SEED = 238


class Loader:
    def __init__(self):
        self.inputs = []

    def read(self, path, source):
        ref = REFS[source]
        local = ROOT / path
        sha = subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT).decode().strip()
        if local.exists():
            content = local.read_bytes()
            source_ref = "working_tree"
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        else:
            content = subprocess.check_output(["git", "show", f"{sha}:{path}"], cwd=ROOT)
            source_ref = ref
        self.inputs.append(dict(path=path, source_ref=source_ref, commit_sha=sha,
                                sha256=hashlib.sha256(content).hexdigest()))
        return list(csv.DictReader(io.StringIO(content.decode())))


def select(rows, seeded):
    return [r for r in rows if (int(r["policy_seed"]) >= 0) == seeded]


def paired(a, b):
    """Require complete rectangular pairing; never silently drop episodes/seeds."""
    def index(rows):
        result = {}
        for r in rows:
            key = (int(r["policy_seed"]), r["order"], int(r["seed"]))
            if key in result:
                raise ValueError(f"Duplicate episode: {key}")
            value = float(r["accuracy"]) * 100
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite accuracy: {key}")
            result[key] = value
        if not result:
            raise ValueError("Empty comparison")
        return result
    ai, bi = index(a), index(b)
    sa, sb = sorted({k[0] for k in ai}), sorted({k[0] for k in bi})
    if sa != [-1] and sb != [-1] and sa != sb:
        raise ValueError("Policy seed mismatch")
    seeds = sb if sa == [-1] else sa
    keys_a = {(o, e) for _, o, e in ai}
    if keys_a != {(o, e) for _, o, e in bi}:
        raise ValueError("Episode mismatch")
    matrices = {}
    try:
        for order in sorted({o for o, _ in keys_a}):
            episodes = tuple(sorted(e for o, e in keys_a if o == order))
            d = np.array([[ai[(-1 if sa == [-1] else s, order, e)] -
                           bi[(-1 if sb == [-1] else s, order, e)]
                           for e in episodes] for s in seeds])
            matrices[order] = (episodes, d)
    except KeyError as exc:
        raise ValueError("Incomplete episode rectangle") from exc
    return matrices


def bootstrap(matrices, n_resamples=N_RESAMPLES, rng_seed=RNG_SEED):
    """Equal-weight conditions; shared episode draws for identical seed grids."""
    if not matrices:
        raise ValueError("Empty conditions")
    rng = np.random.default_rng(rng_seed)
    n = next(iter(matrices.values()))[1].shape[0]
    seed_draw = rng.integers(n, size=(n_resamples, n))
    draws = {}
    samples = np.zeros(n_resamples)
    seed_means = np.zeros(n)
    wins = []
    means = []
    for episodes, d in matrices.values():
        if d.shape != (n, len(episodes)) or not len(episodes):
            raise ValueError("Invalid difference matrix")
        if episodes not in draws:
            draws[episodes] = rng.integers(len(episodes), size=(n_resamples, len(episodes)))
        samples += d[seed_draw[:, :, None], draws[episodes][:, None, :]].mean(axis=(1, 2))
        seed_means += d.mean(axis=1)
        means.append(d.mean())
        wins.append(np.mean((d > 0) + 0.5 * (d == 0)))
    samples /= len(matrices)
    low, high = np.percentile(samples, [2.5, 97.5])
    return dict(n_policy_seeds=n, n_episodes=sum(len(e) for e, _ in matrices.values()),
                mean_diff_pp=float(np.mean(means)), ci_low_pp=float(low), ci_high_pp=float(high),
                p_le_0=float(np.mean(samples <= 0)), episode_win_rate=float(np.mean(wins)),
                seeds_beating=int(np.sum(seed_means > 0)))


def main():
    loader = Loader()
    rows = []
    policies, fixed = {}, {}
    mean_checks = {}

    def compare(name, family, suite, a, b):
        matrices = paired(a, b)
        groups = {k: {k: v} for k, v in matrices.items()}
        groups["biased_mean"] = {k: v for k, v in matrices.items() if k != "mixed"}
        groups["all"] = matrices
        for condition, group in groups.items():
            orders = set(group)
            failures = lambda rr: sum(str(r.get("failed", "False")).lower() == "true"
                                      for r in rr if r["order"] in orders)
            rows.append(dict(comparison=name, family=family, suite=suite, condition=condition,
                             **bootstrap(group), failed_a=failures(a), failed_b=failures(b)))

    for suite in ("regular", "extreme"):
        for family in FAMILIES:
            path = ("results/recovery_study/evaluation.csv" if family == "SGD" and suite == "regular"
                    else f"results/{'optimizer_families' if suite == 'regular' else 'extreme_orderings'}/{family}_evaluation.csv")
            base = loader.read(path, "main")
            p = select(base, True)
            f = select(base, False)
            for seed in (340, 341, 342):
                p += loader.read(f"results/additional_seeds/{suite}/{family}_{seed}/evaluation.csv", "additional")
            policies[suite, family], fixed[suite, family] = p, f
            compare("policy_vs_fixed", family, suite, p, f)
            for condition in sorted({r["order"] for r in p}) + ["biased_mean", "all"]:
                rr = [r for r in p if r["order"] == condition or condition == "all" or
                      (condition == "biased_mean" and r["order"] != "mixed")]
                mean_checks[suite, family, condition] = np.mean([float(r["accuracy"]) * 100 for r in rr])
            compute = loader.read(f"results/compute_matched_baseline/{suite}/{family}_evaluation.csv", "compute")
            compare("policy_vs_compute_fixed", family, suite, p, compute)
            compare("compute_fixed_vs_fixed", family, suite, compute, f)

    for family in FAMILIES:
        p = [r for r in policies["regular", family] if int(r["policy_seed"]) in (338, 339)]
        f = fixed["regular", family]
        ol = []
        for seed in (338, 339):
            ol += loader.read(f"results/open_loop/regular/{family}_gradient_{seed}/evaluation.csv", "open")
        compare("policy_vs_open_loop", family, "regular", p, ol)
        compare("open_loop_vs_fixed", family, "regular", ol, f)
        if family in ("SGD", "Adam"):
            ablations = loader.read(f"results/ablations/{family}_evaluation.csv", "ablation")
            for variant in ("lr_only", "fixed_memory", "no_feedback"):
                a = [r for r in ablations if r["variant"] == variant]
                compare(f"{variant}_vs_fixed", family, "regular", a, f)
                compare(f"policy_vs_{variant}", family, "regular", p, a)
        retrained = loader.read(f"results/extreme_retrained/{family}_evaluation.csv", "retrained")
        compare("retrained_vs_retuned_fixed", family, "extreme_retrained",
                select(retrained, True), select(retrained, False))

    summary = loader.read("results/additional_seeds/summary.csv", "additional")
    max_diff = max(abs(mean_checks[r["suite"], r["family"], r["condition"]] -
                       float(r["policy_mean_pct"])) for r in summary)
    if max_diff > 1e-10:
        raise ValueError(f"PR4 policy mean mismatch: {max_diff}")
    OUT.mkdir(exist_ok=True)
    with (OUT / "bootstrap.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "inputs.json").write_text(json.dumps(loader.inputs, indent=2) + "\n")
    table = ["| Comparison | Suite | Family | Biased mean pp [95% CI] | Mixed pp [95% CI] |",
             "|---|---|---|---:|---:|"]
    for r in rows:
        if r["condition"] != "biased_mean":
            continue
        mixed = next(x for x in rows if all(x[k] == r[k] for k in ("comparison", "suite", "family")) and x["condition"] == "mixed")
        fmt = lambda x: f'{x["mean_diff_pp"]:+.3f} [{x["ci_low_pp"]:+.3f}, {x["ci_high_pp"]:+.3f}]'
        table.append(f'| {r["comparison"]} | {r["suite"]} | {r["family"]} | {fmt(r)} | {fmt(mixed)} |')
    (OUT / "README.md").write_text(README + "\n" + "\n".join(table) + "\n")
    print(f"Wrote {len(rows)} bootstrap rows; {len(loader.inputs)} input records")
    print(f"PR4 5-seed policy means: max abs diff = {max_diff:.12g} pp ({len(summary)} rows)")


README = """# Saved-result paired bootstrap

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
"""


if __name__ == "__main__":
    main()
