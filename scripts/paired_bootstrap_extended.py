"""Paired hierarchical bootstrap CIs for the later experiments (saved CSVs only).

Same method as scripts/paired_bootstrap.py, whose pairing and resampling
functions are imported unchanged: pair by ordering and episode seed (and policy
seed when both arms are seeded), resample policy seeds and then episode seeds,
10,000 percentile resamples, NumPy default_rng(238) for every row, orderings
weighted equally, fixed arms broadcast across policy seeds.

Groups (``--groups``; default all whose inputs exist):
  ablations_start  start-held lr_only / fixed_memory vs full policy, fixed,
                   and the tuned-held ablation (SGD, Adam; seeds 338-339)
  retrained_five   five-seed extreme retraining vs extreme-retuned fixed,
                   saved fixed, and seed-matched frozen policies
  warm             warm-start vs freshly paired fixed (5 seeds) and vs full
                   policy (seeds 338-339), regular and extreme suites
  held_out         leave-one-type-out retrained policies vs fold-retuned
                   fixed, frozen policy and frozen fixed (seeds 338-339)
  clean_test       results/clean_test: headline comparisons on the fresh
                   official-test replay and on the 6,000 clean images
  scrambled        results/scrambled_feedback: real vs scrambled feedback,
                   scrambled vs fixed and open loop, both test sets

Writes results/paired_bootstrap_extended/{bootstrap.csv,inputs.json,summary_table.md}.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paired_bootstrap import N_RESAMPLES, RNG_SEED, bootstrap, paired  # noqa: E402

OUT = ROOT / "results/paired_bootstrap_extended"
FAMILIES = ("SGD", "Adam", "AdamW", "RMSprop")
FOLDS = ("mixed", "single_max", "single_late", "pair_max", "half_block", "staircase",
         "class_sorted", "alternating_halves")
GROUPS = ("ablations_start", "retrained_five", "warm", "held_out", "clean_test", "scrambled")


class Inputs:
    def __init__(self):
        self.records = []
        self.head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    def read(self, path, metric="accuracy", **filters):
        """Rows of a CSV as dicts with ``accuracy`` set from ``metric``."""
        local = ROOT / path
        content = local.read_bytes()
        text = gzip.decompress(content).decode() if local.suffix == ".gz" else content.decode()
        if path not in {r["path"] for r in self.records}:
            self.records.append(dict(path=path, head=self.head, sha256=hashlib.sha256(content).hexdigest()))
        rows = []
        for r in csv.DictReader(io.StringIO(text)):
            if all(r.get(k) == str(v) for k, v in filters.items()):
                rows.append(dict(r, accuracy=r[metric]))
        if not rows:
            raise ValueError(f"No rows in {path} for {filters}")
        return rows


def seeds(rows, wanted):
    return [r for r in rows if int(r["policy_seed"]) in wanted]


def keep_orders(rows, orders):
    return [r for r in rows if r["order"] in orders]


class Runner:
    def __init__(self):
        self.rows = []

    def compare(self, group, name, family, suite, a, b, headline, test="official"):
        matrices = paired(a, b)
        conditions = {k: {k: v} for k, v in matrices.items()}
        conditions["all"] = matrices
        if "mixed" in matrices and len(matrices) > 1:
            conditions["biased_mean"] = {k: v for k, v in matrices.items() if k != "mixed"}
        for condition, group_matrices in conditions.items():
            orders = set(group_matrices)
            failed = lambda rr: sum(str(r.get("failed", "False")).lower() == "true" for r in rr if r["order"] in orders)
            self.rows.append(dict(group=group, comparison=name, family=family, suite=suite, test=test,
                                  condition=condition, headline=condition == headline,
                                  **bootstrap(group_matrices), failed_a=failed(a), failed_b=failed(b)))


def ablations_start(inp, run):
    for family in ("SGD", "Adam"):
        start = inp.read(f"results/ablations_start_values/{family}_evaluation.csv")
        tuned = inp.read(f"results/ablations/{family}_evaluation.csv")
        base = inp.read(f"results/optimizer_families/{family}_evaluation.csv")
        full = [r for r in base if r["variant"] == "gradient"]
        fixed = [r for r in base if r["variant"] == "fixed"]
        for variant in ("lr_only", "fixed_memory"):
            a = [r for r in start if r["variant"] == variant]
            run.compare("ablations_start", f"start_{variant}_vs_full_policy", family, "regular", a, full, "all")
            run.compare("ablations_start", f"start_{variant}_vs_fixed", family, "regular", a, fixed, "all")
            run.compare("ablations_start", f"start_{variant}_vs_tuned_held_{variant}", family, "regular", a,
                        [r for r in tuned if r["variant"] == variant], "all")


def retrained_five(inp, run):
    for family in FAMILIES:
        old = inp.read(f"results/extreme_retrained/{family}_evaluation.csv")
        new = inp.read(f"results/extreme_retrained_five_seeds/{family}_evaluation.csv", variant="policy_retrained")
        retrained = [r for r in old if r["variant"] == "policy_retrained"] + new
        retuned = [r for r in old if r["variant"] == "fixed_retuned"]
        frozen_base = inp.read(f"results/extreme_orderings/{family}_evaluation.csv")
        saved_fixed = [r for r in frozen_base if r["variant"] == "fixed"]
        frozen = seeds([r for r in frozen_base if r["variant"] == "policy"], (338, 339))
        for s in (340, 341, 342):
            frozen += inp.read(f"results/additional_seeds/extreme/{family}_{s}/evaluation.csv")
        run.compare("retrained_five", "retrained_vs_retuned_fixed", family, "extreme", retrained, retuned, "biased_mean")
        run.compare("retrained_five", "retrained_vs_saved_fixed", family, "extreme", retrained, saved_fixed, "biased_mean")
        run.compare("retrained_five", "retrained_vs_frozen", family, "extreme", retrained, frozen, "biased_mean")


def warm(inp, run):
    for family in FAMILIES:
        for suite, headline in (("regular", "biased_mean"), ("extreme", "all")):
            rows = inp.read(f"results/warm_start/{family}_{suite}_evaluation.csv")
            w = [r for r in rows if r["variant"] == "warm"]
            fixed = [r for r in rows if r["variant"] == "fixed"]
            full = [r for r in rows if r["variant"] == "full"]
            run.compare("warm", "warm_vs_fixed", family, suite, w, fixed, headline)
            run.compare("warm", "warm_vs_full", family, suite, seeds(w, (338, 339)), full, headline)


def held_out(inp, run):
    for family in FAMILIES:
        rows = []
        for fold in FOLDS:
            rows += keep_orders(inp.read(f"results/extreme_held_out/{family}/{fold}/evaluation.csv"), {fold})
        by = lambda v: [r for r in rows if r["variant"] == v]
        retrained = by("policy_retrained")
        run.compare("held_out", "retrained_vs_fold_retuned_fixed", family, "extreme_held_out", retrained,
                    by("fixed_retuned"), "all")
        run.compare("held_out", "retrained_vs_frozen_policy", family, "extreme_held_out", retrained,
                    by("policy_frozen"), "all")
        run.compare("held_out", "retrained_vs_frozen_fixed", family, "extreme_held_out", retrained,
                    by("fixed_frozen"), "all")


CLEAN_COMPARISONS = [
    # name, suite, method, comparator, policy seeds, headline condition
    ("policy_vs_fixed", "regular", "policy", "fixed", (338, 339, 340, 341, 342), "biased_mean"),
    ("policy_vs_fixed", "extreme", "policy", "fixed", (338, 339, 340, 341, 342), "biased_mean"),
    ("retrained_vs_retuned_fixed", "extreme", "retrained", "fixed_retuned", (338, 339, 340, 341, 342), "biased_mean"),
    ("retrained_vs_frozen", "extreme", "retrained", "policy", (338, 339, 340, 341, 342), "biased_mean"),
    ("warm_vs_fixed", "regular", "warm", "fixed", (338, 339, 340, 341, 342), "biased_mean"),
    ("warm_vs_fixed", "extreme", "warm", "fixed", (338, 339, 340, 341, 342), "all"),
    ("policy_vs_open_loop", "regular", "policy", "open_loop", (338, 339), "biased_mean"),
]


def clean_rows(inp, suite, method, family, policy_seeds, metric):
    base = f"results/clean_test/units/{suite}/{method}"
    if method.startswith("fixed"):
        return inp.read(f"{base}/{family}/evaluation.csv", metric)
    return [r for s in policy_seeds for r in inp.read(f"{base}/{family}_{s}/evaluation.csv", metric)]


def clean_test(inp, run):
    for test in ("official", "clean"):
        metric = f"{test}_accuracy"
        for name, suite, method, comparator, ps, headline in CLEAN_COMPARISONS:
            for family in FAMILIES:
                a = clean_rows(inp, suite, method, family, ps, metric)
                b = clean_rows(inp, suite, comparator, family, ps, metric)
                run.compare("clean_test", name, family, suite, a, b, headline, test)


SCRAMBLED_COMPARISONS = [
    ("real_vs_scrambled_cross", "real", "scrambled_cross", (338, 339, 340, 341, 342)),
    ("real_vs_scrambled_same_order", "real", "scrambled_same_order", (338, 339, 340, 341, 342)),
    ("scrambled_cross_vs_fixed", "scrambled_cross", "fixed", (338, 339, 340, 341, 342)),
    ("scrambled_same_order_vs_fixed", "scrambled_same_order", "fixed", (338, 339, 340, 341, 342)),
    ("scrambled_cross_vs_open_loop", "scrambled_cross", "open_loop", (338, 339)),
    ("real_vs_open_loop", "real", "open_loop", (338, 339)),
]


def scrambled_rows(inp, arm, family, policy_seeds, metric):
    if arm == "real":
        return clean_rows(inp, "regular", "policy", family, policy_seeds, metric)
    if arm in ("fixed", "open_loop"):
        return clean_rows(inp, "regular", arm, family, policy_seeds, metric)
    donor = arm.replace("scrambled_", "")
    return [r for s in policy_seeds
            for r in inp.read(f"results/scrambled_feedback/units/{donor}/{family}_{s}/evaluation.csv", metric)]


def scrambled(inp, run):
    for test in ("official", "clean"):
        metric = f"{test}_accuracy"
        for name, a_arm, b_arm, ps in SCRAMBLED_COMPARISONS:
            for family in FAMILIES:
                run.compare("scrambled", name, family, "regular", scrambled_rows(inp, a_arm, family, ps, metric),
                            scrambled_rows(inp, b_arm, family, ps, metric), "biased_mean", test)


def available(group):
    marker = {"clean_test": "results/clean_test/summary.csv",
              "scrambled": "results/scrambled_feedback/arms.csv"}.get(group)
    return marker is None or (ROOT / marker).exists()


def table(rows):
    fmt = lambda r: f'{r["mean_diff_pp"]:+.3f} [{r["ci_low_pp"]:+.3f}, {r["ci_high_pp"]:+.3f}]'
    lines = ["| Group | Comparison | Suite | Test | Optimizer | Condition | Seeds | Difference pp [95% CI] | Seeds beating | Episode win rate |",
             "|---|---|---|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["headline"]:
            lines.append(f'| {r["group"]} | {r["comparison"]} | {r["suite"]} | {r["test"]} | {r["family"]} | '
                         f'{r["condition"]} | {r["n_policy_seeds"]} | {fmt(r)} | {r["seeds_beating"]}/{r["n_policy_seeds"]} | '
                         f'{r["episode_win_rate"]:.3f} |')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+", choices=GROUPS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    groups = args.groups or [g for g in GROUPS if available(g)]
    inp, run = Inputs(), Runner()
    for group in groups:
        globals()[group](inp, run)
        print(f"{group}: {sum(r['group'] == group for r in run.rows)} rows", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "bootstrap.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run.rows[0]))
        writer.writeheader()
        writer.writerows(run.rows)
    (args.out / "inputs.json").write_text(json.dumps(dict(
        groups=groups, n_resamples=N_RESAMPLES, rng_seed=RNG_SEED, inputs=inp.records), indent=2) + "\n")
    (args.out / "summary_table.md").write_text(table(run.rows))
    print(f"Wrote {len(run.rows)} rows for {groups}", flush=True)


if __name__ == "__main__":
    main()
