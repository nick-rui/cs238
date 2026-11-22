"""Seed-level policy-vs-fixed summary for one CS238_DATASET regular-suite run.

Reads results/<dataset>/optimizer_families/{family}_evaluation.csv (SGD imported
from recovery_study), the policy training/development records and the tuned
baselines. No training or evaluation. Prints Markdown tables; --write saves
policy_vs_fixed.{csv,md} under results/<dataset>/ (refused for existing files).
"""

import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paired_bootstrap import bootstrap, paired  # noqa: E402

FAMILIES = ["SGD", "Adam", "AdamW", "RMSprop"]
SEEDS = [338, 339]
GROUPS = {"shuffled": ["mixed"], "biased_mean": None, "all": None}


def is_true(value):
    return str(value).lower() == "true"


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--dataset", default="cifar10", choices=["mnist", "fashion_mnist", "cifar10"])
    arg.add_argument("--write", action="store_true")
    a = arg.parse_args()
    root = ROOT / "results" / ("" if a.dataset == "mnist" else a.dataset)
    fam_dir, sgd_dir = root / "optimizer_families", root / "recovery_study"
    rows, md = [], []
    for family in FAMILIES:
        records = list(csv.DictReader((fam_dir / f"{family}_evaluation.csv").open()))
        fixed = [r for r in records if r["variant"] == "fixed"]
        policy = [r for r in records if r["variant"] == "gradient"]
        orders = sorted({r["order"] for r in records})
        assert len(fixed) == 200 and len(policy) == 400 and len(orders) == 10 and "mixed" in orders
        assert sorted({int(r["policy_seed"]) for r in policy}) == SEEDS
        matrices = paired(policy, fixed)  # rows: policy seed, cols: episode seed
        baseline = json.loads((fam_dir / f"{family}_baseline.json").read_text())["config"]
        for group, members in GROUPS.items():
            members = members or ([o for o in orders if o != "mixed"] if group == "biased_mean" else orders)
            sub = {o: matrices[o] for o in members}
            # Equal weight per ordering, as analyze_regular.py's aggregates.
            acc = lambda rr: float(np.mean([np.mean([float(r["accuracy"]) * 100 for r in rr if r["order"] == o])
                                            for o in members]))
            per_seed = {s: float(np.mean([d[i].mean() for _, d in sub.values()])) for i, s in enumerate(SEEDS)}
            boot = bootstrap(sub)
            row = dict(
                family=family, condition=group, n_orderings=len(members),
                fixed_pct=acc(fixed), policy_pct=acc(policy),
                **{f"policy_{s}_pct": acc([r for r in policy if int(r["policy_seed"]) == s]) for s in SEEDS},
                diff_pp=boot["mean_diff_pp"],
                **{f"diff_{s}_pp": per_seed[s] for s in SEEDS},
                seed_sd_pp=float(np.std(list(per_seed.values()), ddof=1)),
                seed_wins=sum(v > 0 for v in per_seed.values()), n_seeds=len(SEEDS),
                ci_low_pp=boot["ci_low_pp"], ci_high_pp=boot["ci_high_pp"],
                episode_win_rate=boot["episode_win_rate"],
                failed_test_fixed=sum(is_true(r["failed"]) for r in fixed if r["order"] in members),
                failed_test_policy=sum(is_true(r["failed"]) for r in policy if r["order"] in members),
            )
            rows.append(row)
        # Check against analyze_regular.py's aggregates when present.
        agg_path = fam_dir / "aggregates.csv"
        if agg_path.exists():
            for agg in csv.DictReader(agg_path.open()):
                if agg["family"] != family:
                    continue
                mine = next(r for r in rows if r["family"] == family and
                            r["condition"] == {"Random": "shuffled", "Biased": "biased_mean"}[agg["ordering"]])
                for k, m in [("fixed_accuracy", "fixed_pct"), ("policy_accuracy", "policy_pct"), ("difference_pp", "diff_pp")]:
                    assert abs(float(agg[k]) - mine[m]) < 1e-6, (family, agg, mine)
        jobs = [(sgd_dir / f"gradient_{s}") if family == "SGD" else (fam_dir / f"{family}_gradient_{s}") for s in SEEDS]
        training = []
        for s, job in zip(SEEDS, jobs):
            done = json.loads((job / "done.json").read_text())
            dev = list(csv.DictReader((job / "development.csv").open()))
            checkpoint = sorted({int(r["checkpoint"]) for r in policy if int(r["policy_seed"]) == s} or {-1})
            training.append(dict(seed=s, failed=done["failed"], episodes=done["episodes"],
                                 minutes=done["seconds"] / 60, checkpoint=checkpoint,
                                 dev_failed=sum(is_true(r["failed"]) for r in dev)))
        md.append((family, baseline, training))
    table = ["| Optimizer | Condition | Fixed % | Policy % (338 / 339) | Policy − fixed pp (338 / 339) | Seed SD pp | Seed wins | 95% bootstrap CI pp | Episode win rate | Failed test (fixed / policy) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["condition"] == "all":
            continue
        table.append(
            f"| {r['family']} | {r['condition'].replace('_', ' ')} | {r['fixed_pct']:.2f} | "
            f"{r['policy_pct']:.2f} ({r['policy_338_pct']:.2f} / {r['policy_339_pct']:.2f}) | "
            f"{r['diff_pp']:+.2f} ({r['diff_338_pp']:+.2f} / {r['diff_339_pp']:+.2f}) | {r['seed_sd_pp']:.2f} | "
            f"{r['seed_wins']}/{r['n_seeds']} | [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] | "
            f"{r['episode_win_rate']:.2f} | {r['failed_test_fixed']} / {r['failed_test_policy']} |")
    table += ["", "| Optimizer | Tuned fixed configuration | Selected checkpoint (338 / 339) | Failed training episodes (338 / 339, of 4000 each) | Failed dev episodes | Train minutes (338 / 339) |",
              "|---|---|---:|---:|---:|---:|"]
    for family, c, t in md:
        conf = ", ".join(f"{k}={v}" for k, v in c.items() if k != "family")
        table.append(f"| {family} | {conf} | {t[0]['checkpoint'][0]} / {t[1]['checkpoint'][0]} | "
                     f"{t[0]['failed']} / {t[1]['failed']} | {t[0]['dev_failed']} / {t[1]['dev_failed']} | "
                     f"{t[0]['minutes']:.0f} / {t[1]['minutes']:.0f} |")
    text = "\n".join(table) + "\n"
    print(text)
    if a.write:
        for name in ["policy_vs_fixed.csv", "policy_vs_fixed.md"]:
            assert not (root / name).exists(), f"Refusing to overwrite {root / name}"
        with (root / "policy_vs_fixed.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (root / "policy_vs_fixed.md").write_text(text)


if __name__ == "__main__":
    main()
