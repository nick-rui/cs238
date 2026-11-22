"""Warm-started controllers trained on the extreme-ordering distribution.

Training reuses ``retrain.train`` unchanged (500 updates x 8 episodes, the
alternating extreme training halves, the 6000000-based training seeds and the
extreme development seeds 1930000-1930015), but every policy starts from the
extreme-retuned fixed configuration via the existing ``warm=`` initialization
and rollout path. Evaluation re-runs every comparator fresh in this
environment on the eight extreme orderings and test seeds 1920000-1920019.
"""

import argparse
import concurrent.futures
import csv
import gzip
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/extreme_orderings"))
import retrain as r  # noqa: E402
import torch  # noqa: E402

e, f, b = r.e, r.f, r.b
OUT = ROOT / "results/extreme_warm_start"
FAMILIES = tuple(r.FAMILIES)
SEEDS = tuple(range(338, 343))
ORDERS = tuple(r.ORDERS)
TEST_SEEDS = tuple(r.SEEDS)
NEW = "warm_extreme"
VARIANTS = ("fixed_retuned", "policy_retrained", "warm_regular", NEW)
JOB_ORDER = ("Adam", "AdamW", "SGD", "RMSprop")  # slowest first


def extreme_config(family):
    return json.loads((ROOT / f"results/extreme_retrained/{family}_baseline.json").read_text())["config"]


def regular_config(family):
    if family == "SGD":
        return json.loads((ROOT / "results/recovery_study/baseline_configs.json").read_text())[family]
    return json.loads((ROOT / f"results/optimizer_families/{family}_baseline.json").read_text())["config"]


def checkpoint_path(family, variant, seed, out=OUT):
    if variant == NEW:
        return out / f"{family}_gradient_{seed}" / "best.pt"
    if variant == "warm_regular":
        sub = f"SGD/gradient_{seed}" if family == "SGD" else f"{family}_gradient_{seed}"
        return ROOT / "results/warm_start" / sub / "best.pt"
    if variant == "policy_retrained":
        root = r.OUT if seed in r.ORIGINAL_POLICY_SEEDS else r.EXTENDED_OUT
        return root / f"{family}_gradient_{seed}" / "best.pt"
    raise ValueError(variant)


def warm_for(family, variant):
    return {NEW: extreme_config(family), "warm_regular": regular_config(family)}.get(variant)


def train(family, seed, out):
    """Run retrain.train with warm-started policy construction and rollouts."""
    c = extreme_config(family)
    make_policy, rollout, saved_out = r.make_policy, r.rollout, r.OUT

    def warm_policy(fam, s):
        return e.policy("gradient", s, warm=c) if fam == "SGD" else f.policy(fam, "gradient", s, warm=c)

    def warm_rollout(*args, **kwargs):
        assert "warm" not in kwargs and "fixed" not in kwargs
        return rollout(*args, warm=c, **kwargs)

    r.OUT, r.make_policy, r.rollout = out, warm_policy, warm_rollout
    try:
        r.train(family, seed)
    finally:
        r.OUT, r.make_policy, r.rollout = saved_out, make_policy, rollout
    done = out / f"{family}_gradient_{seed}" / "done.json"
    record = json.loads(done.read_text())
    if "warm_config" not in record:
        record.update(warm_config=c, warm_source=f"results/extreme_retrained/{family}_baseline.json")
        done.write_text(json.dumps(record))


def evaluate(family, variant, seed, out):
    """Evaluate one (variant, policy seed) on all eight orderings x 20 seeds."""
    folder = out / "evaluation"
    folder.mkdir(exist_ok=True)
    path = folder / f"{family}_{variant}_{seed}.csv"
    if path.exists():
        return
    torch.set_num_threads(1)
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    testset = (test.data.reshape(-1, 784).float() / 255, test.targets)
    pol, update, fixed = None, -1, None
    if variant == "fixed_retuned":
        fixed = extreme_config(family)
    else:
        saved = torch.load(checkpoint_path(family, variant, seed, out), weights_only=True)
        update = saved["update"]
        pol = r.make_policy(family, seed)
        pol.load_state_dict(saved["policy"])
        pol.eval()
    warm = warm_for(family, variant)
    rows, traces = [], []
    start = time.time()
    for order in ORDERS:
        for st in range(0, len(TEST_SEEDS), 4):
            ss = list(TEST_SEEDS[st:st + 4])
            if family == "SGD" and fixed is not None:
                pairs = [b.baseline(fixed, s, order, testset, True) for s in ss]
                rr, tr = [p[0] for p in pairs], [t for p in pairs for t in p[1]]
            elif family == "SGD":
                rr, tr, *_ = e.rollout(pol, "gradient", ss, [order] * 4, evaluation=testset, trace=True, warm=warm)
            else:
                rr, tr, *_ = f.rollout(family, "gradient", ss, [order] * 4, pol=pol, fixed=fixed,
                                       evaluation=testset, trace=True, warm=warm)
            for row in rr:
                row.update(family=family, variant=variant, policy_seed=seed, checkpoint=update, threads=1)
            for row in tr:
                row.update(family=family, variant=variant, policy_seed=seed)
            rows += rr
            traces += tr
    assert len(rows) == len(ORDERS) * len(TEST_SEEDS)
    if variant == NEW:
        r.writegzip(folder / f"{family}_{variant}_{seed}_traces.csv.gz", traces)
    tmp = path.with_suffix(".tmp")
    e.writecsv(tmp, rows)
    tmp.replace(path)
    print(json.dumps(dict(family=family, variant=variant, policy_seed=seed, seconds=round(time.time() - start, 1))), flush=True)


def read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load(out):
    data = {}
    for family in FAMILIES:
        for variant in VARIANTS:
            for seed in [-1] if variant == "fixed_retuned" else SEEDS:
                rows = read(out / "evaluation" / f"{family}_{variant}_{seed}.csv")
                index = {(row["order"], int(row["seed"])): row for row in rows}
                assert len(index) == len(rows) == len(ORDERS) * len(TEST_SEEDS)
                assert set(index) == {(o, s) for o in ORDERS for s in TEST_SEEDS}
                data[family, variant, seed] = index
    return data


def drift(data, out):
    """Compare fresh comparator rows with the committed evaluation records."""
    sources = {
        "fixed_retuned": lambda fam, s: (r.OUT / f"{fam}_evaluation.csv", "fixed_retuned"),
        "policy_retrained": lambda fam, s: ((r.OUT if s in r.ORIGINAL_POLICY_SEEDS else r.EXTENDED_OUT) / f"{fam}_evaluation.csv", "policy_retrained"),
        "warm_regular": lambda fam, s: (ROOT / f"results/warm_start/{fam}_extreme_evaluation.csv", "warm"),
    }
    records = []
    for (family, variant, seed), fresh in sorted(data.items()):
        if variant == NEW:
            continue
        path, name = sources[variant](family, seed)
        saved = {(row["order"], int(row["seed"])): row for row in read(path)
                 if row["variant"] == name and int(row["policy_seed"]) == seed}
        assert set(fresh) <= set(saved), (family, variant, seed)
        diffs = [abs(float(fresh[k]["accuracy"]) - float(saved[k]["accuracy"])) for k in fresh]
        records.append(dict(family=family, variant=variant, policy_seed=seed, source=str(path.relative_to(ROOT)),
                            max_abs_accuracy=max(diffs), mean_abs_accuracy=mean(diffs),
                            exact_episodes=sum(d == 0 for d in diffs), episodes=len(diffs)))
    e.writecsv(out / "drift.csv", records)
    return records


def acc(index, orders):
    return mean(float(index[o, s]["accuracy"]) for o in orders for s in TEST_SEEDS)


def analyze(out):
    data = load(out)
    drift(data, out)
    scopes = {"biased": [o for o in ORDERS if o != "mixed"], "all": list(ORDERS)}
    scopes.update({o: [o] for o in ORDERS})
    scores, summary, comparisons = [], [], []
    for family in FAMILIES:
        for scope, orders in scopes.items():
            seed_acc = {}
            for variant in VARIANTS:
                seeds = [-1] if variant == "fixed_retuned" else SEEDS
                values = {s: acc(data[family, variant, s], orders) for s in seeds}
                failures = sum(row["failed"] == "True" for s in seeds for o in orders for row in
                               [data[family, variant, s][o, t] for t in TEST_SEEDS])
                seed_acc[variant] = values
                for s, v in values.items():
                    scores.append(dict(family=family, scope=scope, variant=variant, policy_seed=s, accuracy=v))
                vs = list(values.values())
                summary.append(dict(family=family, scope=scope, variant=variant, mean=mean(vs),
                                    sd=stdev(vs) if len(vs) > 1 else 0.0, min=min(vs), max=max(vs),
                                    policy_seeds=len(vs), failures=failures, episodes=len(vs) * len(orders) * len(TEST_SEEDS)))
            for control in VARIANTS[:-1]:
                deltas = [seed_acc[NEW][s] - seed_acc[control][-1 if control == "fixed_retuned" else s] for s in SEEDS]
                comparisons.append(dict(family=family, scope=scope, comparison=f"{NEW}-{control}",
                                        mean_difference=mean(deltas), sd=stdev(deltas), min=min(deltas), max=max(deltas),
                                        wins=sum(d > 0 for d in deltas), seeds=len(deltas),
                                        **{f"seed_{s}": d for s, d in zip(SEEDS, deltas)}))
    e.writecsv(out / "policy_seed_scores.csv", scores)
    e.writecsv(out / "summary.csv", summary)
    e.writecsv(out / "comparisons.csv", comparisons)
    training = []
    for family in FAMILIES:
        for s in SEEDS:
            folder = out / f"{family}_gradient_{s}"
            done = json.loads((folder / "done.json").read_text())
            best = torch.load(folder / "best.pt", weights_only=True)["update"]
            training.append(dict(family=family, policy_seed=s, best_update=best, seconds=done["seconds"],
                                 training_failures=done["failed"], threads=done["threads"]))
    e.writecsv(out / "training_summary.csv", training)
    for row in comparisons:
        if row["scope"] in ("biased", "all"):
            print(json.dumps({k: row[k] for k in ("family", "scope", "comparison", "mean_difference", "sd", "wins")}))


def design(out, smoke):
    r.assert_disjoint()
    (out / "design.json").write_text(json.dumps(dict(
        smoke=smoke, families=FAMILIES, policy_seeds=SEEDS, orders=ORDERS, test_seeds=TEST_SEEDS,
        initialization="Policy output bias set by the existing warm= path from the extreme-retuned fixed config "
                       "(results/extreme_retrained/<family>_baseline.json); rollouts start optimizer state at that "
                       "config (warm_settings clamps).",
        warm_configs={fam: extreme_config(fam) for fam in FAMILIES},
        training="retrain.train unchanged: 500 updates x 8 episodes, Adam lr 0.003, grad clip 1, paired i/i+4 baseline",
        training_seed_formula="6000000 + policy_seed*10000 + update*8 + episode (same as cold-start retraining)",
        training_orders="Alternate first/last four extreme ORDERS, repeated twice",
        development_seeds=r.DEVSEEDS, development_orders=r.DEVORDERS,
        selection="Minimum mean b.score on development images at updates 0,25,50,100,...,500; no test feedback",
        comparators={
            "fixed_retuned": "extreme-retuned fixed config (results/extreme_retrained/<family>_baseline.json)",
            "policy_retrained": "cold-start extreme retrained, seeds 338-339 results/extreme_retrained, 340-342 results/extreme_retrained_five_seeds",
            "warm_regular": "regular-distribution warm-start policies (results/warm_start), warm from regular-tuned config",
        },
        evaluation="All comparators re-evaluated fresh here, 1 torch thread, batches of 4 test seeds; failed episodes kept",
    ), indent=2))


def launch(out, workers, smoke):
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    extra = ["--smoke"] if smoke else []

    def job(args):
        name = "_".join(args)
        with (logs / f"{name}.log").open("a") as log:
            subprocess.run([sys.executable, __file__, *args, *extra], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True, env=env)
        print("COMPLETED", name, time.strftime("%H:%M:%S"), flush=True)

    trains = [["train", "--family", fam, "--seed", str(s)] for fam in JOB_ORDER for s in SEEDS]
    evals = [["evaluate", "--family", fam, "--variant", v, "--seed", str(s)]
             for fam in JOB_ORDER for v in VARIANTS for s in ([-1] if v == "fixed_retuned" else SEEDS)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        # Comparator evaluations do not depend on training; new ones wait for it.
        comparator = [a for a in evals if a[4] != NEW]
        futures = [pool.submit(job, a) for a in trains + comparator]
        for fut in futures:
            fut.result()
        list(pool.map(job, [a for a in evals if a[4] == NEW]))
    subprocess.run([sys.executable, __file__, "analyze", *extra], cwd=ROOT, check=True, env=env)
    print("EXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate", "analyze", "launch"])
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--seed", type=int, choices=(-1, *SEEDS))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--workers", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--smoke", action="store_true", help="2 updates, one test seed, separate smoke/ directory")
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = OUT / "smoke" if args.smoke else OUT
    if args.smoke:
        r.UPDATES = 2
        TEST_SEEDS = TEST_SEEDS[:4]
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        if args.family is None or args.seed not in SEEDS:
            parser.error("train needs --family and --seed in 338-342")
        train(args.family, args.seed, out)
    elif args.mode == "evaluate":
        if args.family is None or args.variant is None or args.seed is None:
            parser.error("evaluate needs --family, --variant and --seed")
        if (args.seed == -1) != (args.variant == "fixed_retuned"):
            parser.error("fixed_retuned uses --seed -1; policies use 338-342")
        evaluate(args.family, args.variant, args.seed, out)
    elif args.mode == "analyze":
        analyze(out)
    else:
        design(out, args.smoke)
        launch(out, args.workers, args.smoke)
