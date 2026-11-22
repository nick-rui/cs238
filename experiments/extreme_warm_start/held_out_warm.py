"""Held-out-type protocol with warm-started controllers (stretch, two seeds).

``held_out.train`` is reused unchanged (fold training orders, 7000000-based
training seeds, development seeds 1940000-1940015 without the held-out type),
but each policy starts from its fold's retuned fixed config, which was
selected without the held-out type (results/extreme_held_out/<family>/<held>/
fixed_retuned.json). Test: the held-out type only, seeds 1920000-1920019,
paired against fresh replays of the fold-retuned fixed config and the
cold-start held-out policies.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/extreme_orderings"))
import held_out as h  # noqa: E402
import torch  # noqa: E402

e = h.e
OUT = ROOT / "results/extreme_warm_start/held_out"
COLD = h.OUT
FAMILIES, ORDERS, SEEDS = h.FAMILIES, h.ORDERS, h.POLICY_SEEDS
VARIANTS = {"fixed_retuned": (-1,), "policy_retrained": SEEDS, "warm_policy": SEEDS}


def fold_config(family, held_out):
    return json.loads((h.fold_dir(COLD, family, held_out) / "fixed_retuned.json").read_text())["config"]


def train(root, family, held_out, seed, smoke):
    c = fold_config(family, held_out)
    make_policy, rollout = h.policy, h.rollout

    def warm_policy(fam, s):
        return e.policy(h.KIND, s, warm=c) if fam == "SGD" else h.f.policy(fam, h.KIND, s, warm=c)

    def warm_rollout(fam, seeds, orders, pol=None, *, fixed=None, **kwargs):
        assert fixed is None and "warm" not in kwargs
        return rollout(fam, seeds, orders, pol, warm=c, **kwargs)

    h.policy, h.rollout = warm_policy, warm_rollout
    try:
        h.train(root, family, held_out, seed, smoke)
    finally:
        h.policy, h.rollout = make_policy, rollout
    path = h.fold_dir(root, family, held_out) / str(seed) / "training_summary.json"
    record = json.loads(path.read_text())
    record.setdefault("warm_config", c)
    h.write_json(path, record)


def evaluate(root, family, held_out, smoke):
    folder = h.fold_dir(root, family, held_out)
    path = folder / "evaluation.csv"
    if path.exists():
        return
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    testset = (test.data.reshape(-1, 784).float() / 255, test.targets)
    seeds = list(h.test_seeds(smoke))
    config = fold_config(family, held_out)
    rows = []
    for variant, policy_seeds in VARIANTS.items():
        for ps in policy_seeds:
            pol, fixed, warm, update = None, None, None, -1
            if variant == "fixed_retuned":
                fixed = config
            else:
                base = COLD if variant == "policy_retrained" else root
                pol, update = h.load_policy(h.fold_dir(base, family, held_out) / str(ps) / "best.pt",
                                            family, ps, held_out)
                warm = config if variant == "warm_policy" else None
            for st in range(0, len(seeds), 4):
                ss = seeds[st:st + 4]
                kwargs = dict(evaluation=testset)
                if warm is not None:
                    kwargs["warm"] = warm
                rr = h.rollout(family, ss, [held_out] * len(ss), pol, fixed=fixed, **kwargs)[0]
                rows += [dict(row, family=family, variant=variant, policy_seed=ps, checkpoint=update) for row in rr]
    tmp = path.with_suffix(".tmp")
    e.writecsv(tmp, rows)
    tmp.replace(path)
    print(f"{family} {held_out}: {len(rows)} episodes", flush=True)


def analyze(root, smoke):
    seeds = list(h.test_seeds(smoke))
    acc, drift = {}, []
    for family in FAMILIES:
        for held_out in ORDERS:
            fresh = h.read_rows(h.fold_dir(root, family, held_out) / "evaluation.csv")
            saved = {(r["variant"], int(r["policy_seed"]), int(r["seed"])): float(r["accuracy"])
                     for r in h.read_rows(h.fold_dir(COLD, family, held_out) / "evaluation.csv")}
            for r in fresh:
                key = (r["variant"], int(r["policy_seed"]), int(r["seed"]))
                acc[(family, held_out) + key] = float(r["accuracy"])
                if key[0] != "warm_policy":
                    drift.append(abs(float(r["accuracy"]) - saved[key]))
            assert len(fresh) == len(seeds) * 5
    scopes = {"all": ORDERS, "biased": tuple(o for o in ORDERS if o != "mixed")}
    scopes.update({o: (o,) for o in ORDERS})
    summary, comparisons = [], []
    for family in FAMILIES:
        for scope, orders in scopes.items():
            values = {}
            for variant, policy_seeds in VARIANTS.items():
                for ps in policy_seeds:
                    values[variant, ps] = mean(acc[family, o, variant, ps, s] for o in orders for s in seeds)
                    summary.append(dict(family=family, scope=scope, variant=variant, policy_seed=ps,
                                        accuracy=values[variant, ps]))
            for control in ("fixed_retuned", "policy_retrained"):
                deltas = [values["warm_policy", ps] - values[control, -1 if control == "fixed_retuned" else ps]
                          for ps in SEEDS]
                comparisons.append(dict(family=family, scope=scope, comparison=f"warm_policy-{control}",
                                        mean_difference=mean(deltas), wins=sum(d > 0 for d in deltas),
                                        seeds=len(deltas), **{f"seed_{ps}": d for ps, d in zip(SEEDS, deltas)}))
    e.writecsv(root / "summary.csv", summary)
    e.writecsv(root / "comparisons.csv", comparisons)
    h.write_json(root / "drift.json", dict(max_abs_accuracy=max(drift), exact=sum(d == 0 for d in drift),
                                           episodes=len(drift), reference=str(COLD.relative_to(ROOT))))
    for row in comparisons:
        if row["scope"] in scopes and row["scope"] in ("all", "biased"):
            print(json.dumps(row))


def launch(root, workers, smoke, families):
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    extra = ["--smoke"] if smoke else []

    def job(args):
        name = "_".join(args)
        with (logs / f"{name}.log").open("a") as log:
            subprocess.run([sys.executable, __file__, *args, *extra], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True, env=env)
        print("COMPLETED", name, time.strftime("%H:%M:%S"), flush=True)

    folds = [(fam, o) for fam in families for o in ORDERS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(job, [["train", "--family", fam, "--held-out", o, "--seed", str(s)]
                            for fam, o in folds for s in SEEDS]))
        list(pool.map(job, [["evaluate", "--family", fam, "--held-out", o] for fam, o in folds]))
    if set(families) == set(FAMILIES):
        subprocess.run([sys.executable, __file__, "analyze", *extra], cwd=ROOT, check=True, env=env)
    print("HELD-OUT COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate", "analyze", "launch"))
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--held-out", choices=ORDERS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--workers", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    h.check_design()
    root = OUT / "smoke" if args.smoke else OUT
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        train(root, args.family, args.held_out, args.seed, args.smoke)
    elif args.mode == "evaluate":
        evaluate(root, args.family, args.held_out, args.smoke)
    elif args.mode == "analyze":
        analyze(root, args.smoke)
    else:
        launch(root, args.workers, args.smoke, args.families)
