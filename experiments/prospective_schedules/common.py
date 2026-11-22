"""Shared helpers for the prospective-schedule and reactive-rule controls.

Everything here is fitted or selected on development episodes only. Test-suite
episodes are used only by `evaluate_rows`, after a method is fully specified.
Import with the suite already chosen: `setup("extreme")` patches the stream
constructors exactly as experiments/extreme_orderings/run.py does, so regular
and extreme work must run in separate processes.
"""

import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/optimizer_families"))
sys.path.insert(0, str(ROOT / "scripts"))
import family_engine as f  # noqa: E402

b, e = f.base, f.e
FAMILIES = ["SGD", "Adam", "AdamW", "RMSprop"]
POLICY_SEEDS = [338, 339, 340, 341, 342]
SUITES = ["regular", "extreme"]
_suite = {"name": None, "orders": None, "seeds": None}


def setup(suite, threads=1):
    torch.set_num_threads(threads)
    if _suite["name"] not in (None, suite):
        raise RuntimeError("One suite per process")
    if suite == "extreme":
        spec = importlib.util.spec_from_file_location(
            "stress_streams", ROOT / "experiments/extreme_orderings/run.py")
        stress = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stress)  # patches b.stream / b.batches / e.batches
        torch.set_num_threads(threads)
        _suite.update(name=suite, orders=list(stress.ORDERS), seeds=list(stress.SEEDS))
    elif suite == "regular":
        _suite.update(name=suite, orders=list(b.ORDERS), seeds=list(b.TESTSEEDS))
    else:
        raise ValueError(suite)
    return _suite["orders"], _suite["seeds"]


def testset():
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    return test.data.reshape(-1, 784).float() / 255, test.targets


def columns(family):
    return ["lr", "momentum"] + ([] if family == "SGD" else ["second_decay"]) + ["weight_decay"]


def fixed_config(family):
    return json.loads((ROOT / "results/optimizer_families" / f"{family}_baseline.json").read_text())["config"]


def config_values(family, config):
    """Fixed configuration as a (lr, memory, [second memory], wd) tuple."""
    if family == "SGD":
        return (config["lr"], config["momentum"], config["weight_decay"])
    second = config.get("beta2", config.get("alpha"))
    return (config["lr"], config.get("beta1", config.get("momentum")), second, config["weight_decay"])


def checkpoint(family, seed):
    if seed in (338, 339):
        return (ROOT / "results" / ("recovery_study" if family == "SGD" else "optimizer_families")
                / (f"gradient_{seed}" if family == "SGD" else f"{family}_gradient_{seed}") / "best.pt")
    return ROOT / "results/additional_seeds/training" / f"{family}_gradient_{seed}" / "best.pt"


def load_policy(family, seed):
    saved = torch.load(checkpoint(family, seed), weights_only=True)
    pol = e.policy("gradient", seed) if family == "SGD" else f.policy(family, "gradient", seed)
    pol.load_state_dict(saved["policy"])
    pol.eval()
    return pol, saved["update"]


def rollout(family, seeds, orders, evaluation, *, pol=None, schedule=None, controller=None, trace=False):
    """One batched rollout; exactly one of pol / schedule / controller is used."""
    if sum(x is not None for x in (pol, schedule, controller)) != 1:
        raise ValueError("Choose one of policy, schedule, controller")
    if family == "SGD":
        schedules = None if schedule is None else {o: schedule for o in set(orders)}
        rr, tr, *_ = e.rollout(pol, "gradient", seeds, orders, evaluation=evaluation,
                               schedules=schedules, controller=controller, trace=trace)
        if trace:
            tr = b.annotate(tr)
    else:
        rr, tr, *_ = f.rollout(family, "gradient", seeds, orders, pol=pol, evaluation=evaluation,
                               schedule=schedule, controller=controller, trace=trace)
    return rr, tr


def dev_score(family, evaluation=None, n=8, **method):
    """Mean development score on the fixed-tuning episodes (two per regime)."""
    evaluation = evaluation or (e.xd, e.yd)
    rows = []
    for st in range(0, n, 8):
        rr, _ = rollout(family, b.DEVSEEDS[st:st + 8], b.DEVORDERS[st:st + 8], evaluation, **method)
        rows += rr
    return float(np.mean([b.score(r) for r in rows])), rows


def evaluate_rows(family, data, make_method, batch=4):
    """Evaluate on the current suite in batches of four seeds per condition.

    make_method() returns fresh keyword arguments for each batch (controllers
    carry per-episode state, so each batch gets its own instance).
    """
    records, traces = [], []
    orders, seeds = _suite["orders"], _suite["seeds"]
    for order in orders:
        for st in range(0, len(seeds), batch):
            ss = seeds[st:st + batch]
            rr, tr = rollout(family, ss, [order] * len(ss), data, trace=True, **make_method())
            records += rr
            traces += tr
    return records, traces


def readcsv(path):
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as file:
        return list(csv.DictReader(file))


def write_gz(path, rows):
    buffer = io.StringIO(newline="")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    Path(path).write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- saved comparators

def saved_paths(suite, family):
    base = ("results/recovery_study/evaluation.csv" if family == "SGD" and suite == "regular"
            else f"results/{'optimizer_families' if suite == 'regular' else 'extreme_orderings'}/{family}_evaluation.csv")
    extra = [f"results/additional_seeds/{suite}/{family}_{s}/evaluation.csv" for s in (340, 341, 342)]
    return base, extra


def saved_rows(suite, family):
    """Saved fixed rows and five-seed frozen-policy rows (same as paired_bootstrap.py)."""
    base, extra = saved_paths(suite, family)
    rows = readcsv(ROOT / base)
    fixed = [r for r in rows if int(r["policy_seed"]) < 0]
    policy = [r for r in rows if int(r["policy_seed"]) >= 0]
    for path in extra:
        policy += readcsv(ROOT / path)
    if sorted({int(r["policy_seed"]) for r in policy}) != POLICY_SEEDS:
        raise ValueError("Expected five saved policy seeds")
    return fixed, policy


def warm_rows(suite, family):
    """Fresh-environment warm-start policies (five seeds) and their paired fixed rows."""
    rows = readcsv(ROOT / f"results/warm_start/{family}_{suite}_evaluation.csv")
    return ([r for r in rows if r["variant"] == "fixed"], [r for r in rows if r["variant"] == "warm"])


def compare(a, b_rows, orders):
    """Paired hierarchical bootstrap of accuracy(a) - accuracy(b), in pp.

    Uses scripts/paired_bootstrap.py unchanged (10,000 resamples, seed 238,
    policy seeds then episode seeds; unseeded rows broadcast).
    """
    import paired_bootstrap as pb
    matrices = pb.paired(a, b_rows)
    if set(matrices) != set(orders):
        raise ValueError("Comparison does not cover every condition")
    groups = {"biased": {k: v for k, v in matrices.items() if k != "mixed"},
              "mixed": {"mixed": matrices["mixed"]}, "all": matrices}
    return {name: pb.bootstrap(group) for name, group in groups.items()}


def condition_means(rows, orders):
    """Equal-condition mean accuracy (%) and failures for an unseeded or seeded method."""
    by = {o: [float(r["accuracy"]) * 100 for r in rows if r["order"] == o] for o in orders}
    if any(not v for v in by.values()):
        raise ValueError("Missing condition")
    biased = [o for o in orders if o != "mixed"]
    out = dict(mixed=float(np.mean(by["mixed"])), biased=float(np.mean([np.mean(by[o]) for o in biased])),
               all=float(np.mean([np.mean(by[o]) for o in orders])),
               failures=sum(str(r["failed"]) == "True" for r in rows))
    seeds = sorted({int(r["policy_seed"]) for r in rows})
    if seeds != [-1]:
        per = []
        for s in seeds:
            rr = [r for r in rows if int(r["policy_seed"]) == s]
            per.append(np.mean([np.mean([float(r["accuracy"]) * 100 for r in rr if r["order"] == o])
                                for o in biased]))
        out["biased_seed_sd"] = float(np.std(per, ddof=1)) if len(per) > 1 else float("nan")
    return out


def fmt(x):
    return f'{x["mean_diff_pp"]:+.2f} [{x["ci_low_pp"]:+.2f}, {x["ci_high_pp"]:+.2f}]'


def check_finite(rows):
    for r in rows:
        if not math.isfinite(float(r["accuracy"])):
            raise ValueError("Nonfinite accuracy")
