"""Warm-start controllers, paired frozen evaluation, and seed-level comparisons."""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/optimizer_families"))
import family_engine as f

b, e = f.base, f.e
OUT = ROOT / "results/warm_start"
FAMILIES = ["SGD", *f.FAMILIES]
SEEDS = list(range(338, 343))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(family):
    if family == "SGD":
        return json.loads((ROOT / "results/recovery_study/baseline_configs.json").read_text())[family]
    return json.loads((ROOT / f"results/optimizer_families/{family}_baseline.json").read_text())["config"]


def policy(family, seed, warm=None):
    return e.policy("gradient", seed, warm=warm) if family == "SGD" else f.policy(family, "gradient", seed, warm=warm)


def rollout(family, seeds, orders, pol=None, warm=None, fixed=None, evaluation=None):
    if family == "SGD":
        if fixed is not None:
            return [b.baseline(fixed, s, o, evaluation)[0] for s, o in zip(seeds, orders)]
        return e.rollout(pol, "gradient", seeds, orders, warm=warm, evaluation=evaluation)[0]
    return f.rollout(family, "gradient", seeds, orders, pol, warm=warm, fixed=fixed, evaluation=evaluation)[0]


def sanity(family):
    c = config(family)
    pol = policy(family, 338, c)
    scores = {}
    for method in ["warm", "fixed"]:
        rows = []
        for start in range(0, 16, 8):
            rows += rollout(family, b.DEVSEEDS[start:start + 8], b.DEVORDERS[start:start + 8],
                            pol, warm=c if method == "warm" else None,
                            fixed=c if method == "fixed" else None, evaluation=(e.xd, e.yd))
        scores[method] = float(np.mean([b.score(r) for r in rows]))
    scores.update(family=family, difference=scores["warm"] - scores["fixed"])
    (OUT / f"{family}_sanity.json").write_text(json.dumps(scores, indent=2))
    print("SANITY", json.dumps(scores), flush=True)


def train(family, seed):
    c = config(family)
    if family == "SGD":
        folder = OUT / "SGD"
        folder.mkdir(exist_ok=True)
        original_policy, original_rollout = e.policy, e.rollout
        with patch.object(b, "OUT", folder), patch.object(e, "policy", lambda *a, **kw: original_policy(*a, warm=c, **kw)), patch.object(e, "rollout", lambda *a, **kw: original_rollout(*a, warm=c, **kw)):
            b.train("gradient", seed)
    else:
        runner = load_module("family_runner", "experiments/optimizer_families/run.py")
        original_policy, original_rollout = f.policy, f.rollout
        with patch.object(runner, "OUT", OUT), patch.object(f, "policy", lambda *a, **kw: original_policy(*a, warm=c, **kw)), patch.object(f, "rollout", lambda *a, **kw: original_rollout(*a, warm=c, **kw)):
            runner.train(family, "gradient", seed)


def checkpoint(family, seed, warm):
    if warm:
        folder = OUT / (f"SGD/gradient_{seed}" if family == "SGD" else f"{family}_gradient_{seed}")
    else:
        folder = ROOT / "results" / (f"recovery_study/gradient_{seed}" if family == "SGD" else f"optimizer_families/{family}_gradient_{seed}")
    return torch.load(folder / "best.pt", weights_only=True)


def evaluate(family, suite, threads):
    orders, seeds = b.ORDERS, b.TESTSEEDS
    if suite == "extreme":
        stress = load_module("stress_streams", "experiments/extreme_orderings/run.py")
        orders, seeds = stress.ORDERS, stress.SEEDS
    torch.set_num_threads(threads)
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    testset = (test.data.reshape(-1, 784).float() / 255, test.targets)
    rows = []
    for variant, ps in [("fixed", -1), *[("full", s) for s in [338, 339]], *[("warm", s) for s in SEEDS]]:
        pol, update = None, -1
        if variant != "fixed":
            saved = checkpoint(family, ps, variant == "warm")
            pol = policy(family, ps)
            pol.load_state_dict(saved["policy"])
            pol.eval()
            update = saved["update"]
        for order in orders:
            for start in range(0, len(seeds), 4):
                ss = seeds[start:start + 4]
                rr = rollout(family, ss, [order] * len(ss), pol,
                             warm=config(family) if variant == "warm" else None,
                             fixed=config(family) if variant == "fixed" else None, evaluation=testset)
                for row in rr:
                    row.update(family=family, suite=suite, variant=variant, policy_seed=ps, checkpoint=update, threads=threads)
                rows += rr
            print("EVALUATED", family, suite, variant, ps, order, flush=True)
    e.writecsv(OUT / f"{family}_{suite}_evaluation.csv", rows)
    with (ROOT / "results" / ("extreme_orderings" if suite == "extreme" else "optimizer_families") / f"{family}_evaluation.csv").open() as file:
        reference = list(csv.DictReader(file))
    lookup = {(int(r["seed"]), r["order"], "full" if r["variant"] in ["gradient", "policy"] else "fixed", int(r["policy_seed"])): r for r in reference}
    drift = []
    for variant, ps in [("fixed", -1), ("full", 338), ("full", 339)]:
        group = [r for r in rows if r["variant"] == variant and r["policy_seed"] == ps]
        assert len(group) == len(seeds) * len(orders)
        drift.append(dict(family=family, suite=suite, variant=variant, policy_seed=int(ps), threads=threads,
                          **{f"max_abs_{k}": max(abs(float(r[k]) - float(lookup[(r["seed"], r["order"], variant, ps)][k])) for r in group) for k in ["accuracy", "eval_loss", "path_log_loss"]}))
    e.writecsv(OUT / f"{family}_{suite}_drift.csv", drift)
    print("DRIFT", json.dumps(drift), flush=True)


def analyze():
    summaries, differences = [], []
    for path in sorted(OUT.glob("*_evaluation.csv")):
        with path.open() as file:
            records = list(csv.DictReader(file))
        groups = {}
        for r in records:
            group = "extreme" if r["suite"] == "extreme" else "shuffled" if r["order"] == "mixed" else "biased"
            groups.setdefault((r["family"], r["suite"], group), []).append(r)
        for (family, suite, group), rows in groups.items():
            means = {v: float(np.mean([float(r["accuracy"]) for r in rows if r["variant"] == v])) for v in ["fixed", "full", "warm"]}
            summaries.append(dict(family=family, suite=suite, group=group, **means))
            for other in ["fixed", "full"]:
                deltas = []
                for ps in SEEDS if other == "fixed" else [338, 339]:
                    warm = [r for r in rows if r["variant"] == "warm" and int(r["policy_seed"]) == ps]
                    ref = {(r["seed"], r["order"]): r for r in rows if r["variant"] == other and int(r["policy_seed"]) == (-1 if other == "fixed" else ps)}
                    assert len(ref) == len(warm) > 0
                    delta = float(np.mean([float(r["accuracy"]) - float(ref[(r["seed"], r["order"])]["accuracy"]) for r in warm]))
                    deltas.append(delta)
                    differences.append(dict(family=family, suite=suite, group=group, comparison="warm-" + other, policy_seed=ps, difference=delta))
                summaries[-1].update({f"warm_minus_{other}": float(np.mean(deltas)), f"wins_vs_{other}": sum(d > 0 for d in deltas), f"seeds_vs_{other}": len(deltas)})
    e.writecsv(OUT / "summary.csv", summaries)
    e.writecsv(OUT / "seed_differences.csv", differences)
    print((OUT / "summary.csv").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["train", "evaluate", "analyze", "sanity"])
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--threads", type=int, help="Default: 1, except regular SGD evaluation uses the original 2")
    parser.add_argument("--suite", choices=["regular", "extreme"], default="regular")
    args = parser.parse_args()
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads or 1)
    OUT.mkdir(exist_ok=True)
    if args.mode == "analyze":
        analyze()
    else:
        for family in [args.family] if args.family else FAMILIES:
            if args.mode == "sanity":
                sanity(family)
            elif args.mode == "train":
                for seed in [args.seed] if args.seed else SEEDS:
                    train(family, seed)
            else:
                threads = args.threads or (2 if family == "SGD" and args.suite == "regular" else 1)
                evaluate(family, args.suite, threads)
