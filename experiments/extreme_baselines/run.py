"""Extreme-ordering schedule and random baselines.

(1) PR #14's schedule search (cosine / step / warmup-cosine, same axes, same
    candidates) selected on the extreme development episodes used by the
    extreme-retrained policies (seeds 1930000-1930015 x ORDERS*2, mean b.score).
(2) Random baselines: random schedule (uniform post-tanh actions through the
    policy's own action mapping), untrained (update-0) policies, random fixed
    grid configurations.
(3) Fresh frozen regular policies and the extreme-retuned fixed config through
    the batched engine. Other comparators are PR #15's fresh rows.
Definitions were committed in results/extreme_baselines/README.md before any
test evaluation. Test episodes are read only by `evaluate`.
"""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/prospective_schedules"))
import common as c  # noqa: E402

c.setup("extreme")
sys.path.insert(0, str(ROOT / "experiments/extreme_orderings"))
import retrain as r  # noqa: E402

assert Path(r.extreme.__file__).resolve() == ROOT / "experiments/extreme_orderings/run.py"
_spec = importlib.util.spec_from_file_location("prospective_run", ROOT / "experiments/prospective_schedules/run.py")
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)  # candidates(), make_schedule(), KINDS; its OUT is never written here
torch.set_num_threads(1)

e, f, b = c.e, c.f, c.b
OUT = ROOT / "results/extreme_baselines"
FAMILIES = c.FAMILIES
ORDERS = list(r.ORDERS)
TEST_SEEDS = list(r.SEEDS)
POLICY_SEEDS = list(c.POLICY_SEEDS)
RANDOM_SEEDS = [0, 1, 2, 3, 4]
KINDS = ps.KINDS
H = ps.H
# method -> seeds it runs with (-1 = unseeded)
METHODS = {
    **{f"{k}_extreme": [-1] for k in KINDS},
    "fixed_engine": [-1],
    "random_schedule_cold": RANDOM_SEEDS,
    "random_schedule_warm": RANDOM_SEEDS,
    "untrained_cold": POLICY_SEEDS,
    "untrained_warm": POLICY_SEEDS,
    "random_fixed": RANDOM_SEEDS,
    "frozen_regular": POLICY_SEEDS,
}
TRACED = {f"{k}_extreme" for k in KINDS} | {"fixed_engine"}


def extreme_config(family):
    return json.loads((ROOT / f"results/extreme_retrained/{family}_baseline.json").read_text())["config"]


def as_values(family, config):
    return dict(zip(c.columns(family), c.config_values(family, config)))


def n_actions(family):
    return 3 if family == "SGD" else 4


class RandomActions(nn.Module):
    """Stand-in policy: post-tanh actions i.i.d. U[-1, 1], one row per control step.

    Returns mu = atanh(a); the engine's deterministic path uses raw = mu and
    a = tanh(raw), then applies the policy's own action mapping.
    """

    def __init__(self, family, seeds, orders, random_seed):
        super().__init__()
        draws = [np.random.default_rng([238, 7, random_seed, int(s), ORDERS.index(o)])
                 .uniform(-1.0, 1.0, size=(H, n_actions(family))) for s, o in zip(seeds, orders)]
        a = torch.tensor(np.stack(draws, 1), dtype=torch.float32)  # (H, n, k)
        self.register_buffer("a", a.clamp(-1 + 1e-6, 1 - 1e-6))
        self.t = 0

    def forward(self, obs):
        if obs.shape[0] != self.a.shape[1]:
            raise ValueError("Batch does not match the drawn episodes")
        out = torch.atanh(self.a[self.t])
        self.t += 1
        return out


def rollout(family, seeds, orders, evaluation, *, pol=None, schedule=None, warm=None, trace=False):
    """Same engine calls as PR #15 (policies, warm=) and PR #14 (schedule=)."""
    if (pol is None) == (schedule is None) or (schedule is not None and warm is not None):
        raise ValueError("Choose a policy (optionally warm) or a schedule")
    if family == "SGD":
        schedules = None if schedule is None else {o: schedule for o in set(orders)}
        rr, tr, *_ = e.rollout(pol, "gradient", list(seeds), list(orders), evaluation=evaluation,
                               schedules=schedules, trace=trace, warm=warm)
        if trace:
            tr = b.annotate(tr)
    else:
        rr, tr, *_ = f.rollout(family, "gradient", list(seeds), list(orders), pol=pol,
                               evaluation=evaluation, schedule=schedule, trace=trace, warm=warm)
    return rr, tr


def dev_score(family, **method):
    """Extreme development score: mean b.score over 16 episodes, two batches of eight (as retrain.train)."""
    rows = []
    for st in (0, 8):
        rr, _ = rollout(family, r.DEVSEEDS[st:st + 8], r.DEVORDERS[st:st + 8], (e.xd, e.yd), **method)
        rows += rr
    return float(np.mean([b.score(x) for x in rows])), rows


# ---------------------------------------------------------------- random fixed / untrained

def random_fixed_config(family, k):
    grid = r.configs(family)
    index = int(np.random.default_rng([238, 11, FAMILIES.index(family), k]).integers(len(grid)))
    return index, grid[index]


def untrained_policy(family, seed, warm):
    if family == "SGD":
        return e.policy("gradient", seed, warm=warm)
    return f.policy(family, "gradient", seed, warm=warm)


def check_untrained(out):
    """Rebuilt update-0 policies reproduce the update-0 development rows of each training run."""
    records = []
    for family in FAMILIES:
        for seed in POLICY_SEEDS:
            for start in ("cold", "warm"):
                warm = extreme_config(family) if start == "warm" else None
                if start == "warm":
                    path = ROOT / f"results/extreme_warm_start/{family}_gradient_{seed}/development.csv"
                else:
                    root = r.OUT if seed in r.ORIGINAL_POLICY_SEEDS else r.EXTENDED_OUT
                    path = root / f"{family}_gradient_{seed}/development.csv"
                saved = {(int(x["seed"]), x["order"]): x for x in c.readcsv(path) if int(x["update"]) == 0}
                pol = untrained_policy(family, seed, warm)
                pol.eval()
                _, rows = dev_score(family, pol=pol, warm=warm)
                diffs = [abs(float(x["accuracy"]) - float(saved[int(x["seed"]), x["order"]]["accuracy"])) for x in rows]
                ldiffs = [abs(float(x["eval_loss"]) - float(saved[int(x["seed"]), x["order"]]["eval_loss"])) for x in rows]
                records.append(dict(family=family, policy_seed=seed, start=start, source=str(path.relative_to(ROOT)),
                                    episodes=len(rows), max_abs_accuracy=max(diffs), max_abs_eval_loss=max(ldiffs)))
                print(json.dumps(records[-1]), flush=True)
    e.writecsv(out / "untrained_check.csv", records)
    bad = [x for x in records if x["max_abs_accuracy"] > 1e-6]
    if bad:
        raise SystemExit(f"{len(bad)} untrained rebuilds do not match their update-0 development rows")


# ---------------------------------------------------------------- schedule search

def search(family, kind, out, limit=None):
    path = out / "search" / f"{family}_{kind}.csv"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    records = []
    cands = ps.candidates(family, kind)[:limit]
    for cand in cands:
        score, rows = dev_score(family, schedule=ps.make_schedule(family, kind, cand))
        records.append(dict(family=family, **cand, score=score,
                            dev_accuracy=float(np.mean([x["accuracy"] for x in rows])),
                            failures=sum(x["failed"] for x in rows)))
        if cand["config_id"] % 24 == 0:
            print("SEARCH", family, kind, cand["config_id"], round(time.perf_counter() - start), flush=True)
    tmp = path.with_suffix(".tmp")
    e.writecsv(tmp, records)
    tmp.rename(path)
    path.with_suffix(".json").write_text(json.dumps(dict(
        candidates=len(records), episodes=16 * len(records), dev_seeds=r.DEVSEEDS,
        dev_orders=r.DEVORDERS, test_data_used=False, seconds=time.perf_counter() - start), indent=2) + "\n")


def select(family, out):
    path = out / "selected" / f"{family}.json"
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = {}
    for kind in KINDS:
        rows = c.readcsv(out / "search" / f"{family}_{kind}.csv")
        best = min(rows, key=lambda x: float(x["score"]))  # first minimum on ties
        chosen[kind] = dict(score=float(best["score"]), config_id=int(best["config_id"]),
                            config={k: float(best[k]) for k in c.columns(family)},
                            dev_accuracy=float(best["dev_accuracy"]), failures=int(best["failures"]))
    overall = min(KINDS, key=lambda k: chosen[k]["score"])
    fixed = as_values(family, extreme_config(family))
    fixed_score, rows = dev_score(family, schedule=ps.make_schedule(family, "constant", fixed))
    regular = json.loads((ROOT / f"results/prospective_schedules/selected/{family}.json").read_text())
    regular_score, _ = dev_score(family, schedule=ps.make_schedule(family, "cosine", regular["kinds"]["cosine"]["config"]))
    result = dict(family=family, kinds=chosen, overall=overall, fixed_config=fixed,
                  fixed_dev_score_same_engine_16=fixed_score,
                  regular_cosine_config=regular["kinds"]["cosine"]["config"],
                  regular_cosine_extreme_dev_score_16=regular_score,
                  search_episodes=sum(16 * len(ps.candidates(family, k)) for k in KINDS),
                  selection="min mean b.score, extreme dev seeds 1930000-1930015 x ORDERS*2; no test data")
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


# ---------------------------------------------------------------- test evaluation

def method_spec(family, method, seed, out):
    """Returns (make_kwargs(seeds, orders), metadata)."""
    if method.endswith("_extreme"):
        kind = method[:-len("_extreme")]
        sel = json.loads((out / "selected" / f"{family}.json").read_text())
        sched = ps.make_schedule(family, kind, sel["kinds"][kind]["config"])
        return (lambda ss, oo: dict(schedule=sched)), dict(schedule=sched, config=sel["kinds"][kind]["config"])
    if method == "fixed_engine":
        sched = ps.make_schedule(family, "constant", as_values(family, extreme_config(family)))
        return (lambda ss, oo: dict(schedule=sched)), dict(schedule=sched, config=extreme_config(family))
    if method == "random_fixed":
        index, config = random_fixed_config(family, seed)
        sched = ps.make_schedule(family, "constant", as_values(family, config))
        return (lambda ss, oo: dict(schedule=sched)), dict(schedule=sched, grid_index=index, config=config)
    if method.startswith("random_schedule"):
        warm = extreme_config(family) if method.endswith("warm") else None
        return (lambda ss, oo: dict(pol=RandomActions(family, ss, oo, seed), warm=warm)), dict(warm=warm)
    if method.startswith("untrained"):
        warm = extreme_config(family) if method.endswith("warm") else None
        pol = untrained_policy(family, seed, warm)
        pol.eval()
        return (lambda ss, oo: dict(pol=pol, warm=warm)), dict(warm=warm, checkpoint_update=0)
    if method == "frozen_regular":
        pol, update = c.load_policy(family, seed)
        return (lambda ss, oo: dict(pol=pol)), dict(checkpoint=str(c.checkpoint(family, seed).relative_to(ROOT)),
                                                     checkpoint_update=update,
                                                     checkpoint_sha256=c.sha256(c.checkpoint(family, seed)))
    raise ValueError(method)


def evaluate(family, method, seed, out, test_seeds=TEST_SEEDS):
    folder = out / "evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{family}_{method}_{seed}"
    if (folder / f"{name}.json").exists():
        return
    data = c.testset()
    make, meta = method_spec(family, method, seed, out)
    start = time.perf_counter()
    records, traces = [], []
    for order in ORDERS:
        for st in range(0, len(test_seeds), 4):
            ss = test_seeds[st:st + 4]
            kwargs = make(ss, [order] * len(ss))
            rr, tr = rollout(family, ss, [order] * len(ss), data, trace=True, **kwargs)
            if isinstance(kwargs.get("pol"), RandomActions) and kwargs["pol"].t != H:
                raise AssertionError("Random policy not called once per control step")
            if "schedule" in kwargs:
                for x in tr:
                    want = kwargs["schedule"][int(x["step"]) // 5 - 1]
                    got = [float(x[k]) for k in c.columns(family)]
                    if not np.array_equal(np.float32(got), np.float32(want)):
                        raise AssertionError(("schedule changed", name, x["step"]))
            records += rr
            traces += tr
    assert len(records) == len(ORDERS) * len(test_seeds)
    c.check_finite(records)
    for x in records + traces:
        x.update(family=family, variant=method, policy_seed=seed)
    if method in TRACED or (method == "random_schedule_cold" and seed == 0):
        c.write_gz(folder / f"{name}_traces.csv.gz", traces)
    if method.startswith(("random_schedule", "untrained")):
        # Compact per-step settings, averaged over episodes, for inspection.
        steps = []
        for step in range(5, 201, 5):
            group = [x for x in traces if int(x["step"]) == step]
            steps.append(dict(step=step, **{k: float(np.mean([float(x[k]) for x in group])) for k in c.columns(family)},
                              **{f"{k}_sd": float(np.std([float(x[k]) for x in group])) for k in c.columns(family)}))
        e.writecsv(folder / f"{name}_settings.csv", steps)
    e.writecsv(folder / f"{name}.csv", records)
    meta = {k: v for k, v in meta.items() if k != "schedule"}
    (folder / f"{name}.json").write_text(json.dumps(dict(
        family=family, method=method, seed=seed, episodes=len(records),
        failures=sum(x["failed"] for x in records), threads=torch.get_num_threads(),
        torch_version=torch.__version__, seconds=time.perf_counter() - start, **meta), indent=2, default=str) + "\n")
    print("EVALUATED", name, round(time.perf_counter() - start), flush=True)


def replay_check(family, out, seed=338):
    """This rollout path reproduces PR #15's fresh cold-policy rows on four test seeds."""
    root = r.OUT if seed in r.ORIGINAL_POLICY_SEEDS else r.EXTENDED_OUT
    saved = torch.load(root / f"{family}_gradient_{seed}/best.pt", weights_only=True)
    pol = r.make_policy(family, seed)
    pol.load_state_dict(saved["policy"])
    pol.eval()
    ref = {(x["order"], int(x["seed"])): x for x in
           c.readcsv(ROOT / f"results/extreme_warm_start/evaluation/{family}_policy_retrained_{seed}.csv")}
    data = c.testset()
    diffs = []
    for order in ORDERS:
        rr, _ = rollout(family, TEST_SEEDS[:4], [order] * 4, data, pol=pol, trace=True)
        diffs += [abs(float(x["accuracy"]) - float(ref[order, int(x["seed"])]["accuracy"])) for x in rr]
    record = dict(family=family, policy_seed=seed, episodes=len(diffs), max_abs_accuracy=max(diffs),
                  exact=sum(d == 0 for d in diffs))
    print("REPLAY", json.dumps(record), flush=True)
    (out / f"replay_check_{family}.json").write_text(json.dumps(record, indent=2) + "\n")


# ---------------------------------------------------------------- pipeline

def run_jobs(jobs, workers, out, log):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def one(args):
        with (out / "logs" / ("_".join(args).replace("--", "") + ".log")).open("a") as handle:
            code = subprocess.call([sys.executable, "-u", __file__, *args, "--output", str(out)],
                                   stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        print(json.dumps(dict(job=args, code=code, time=time.strftime("%H:%M:%S"))), file=log, flush=True)
        return code

    with ThreadPoolExecutor(workers) as pool:
        codes = list(pool.map(one, jobs))
    if any(codes):
        raise SystemExit(f"{sum(1 for x in codes if x)} jobs failed")


def eval_jobs(methods):
    order = ["Adam", "AdamW", "SGD", "RMSprop"]  # slowest first
    return [["evaluate", "--family", fam, "--method", m, "--seed", str(s)]
            for fam in order for m in methods for s in METHODS[m]]


def pipeline(workers, out):
    (out / "logs").mkdir(parents=True, exist_ok=True)
    start = time.time()
    with (out / "logs" / "pipeline.jsonl").open("a") as log:
        stage1 = [["search", "--family", fam, "--kind", k] for fam in FAMILIES for k in KINDS]
        stage1 += [["check-untrained"]]
        stage1 += eval_jobs([m for m in METHODS if not m.endswith("_extreme")])
        run_jobs(stage1, workers, out, log)
        run_jobs([["select", "--family", fam] for fam in FAMILIES], workers, out, log)
        run_jobs(eval_jobs([m for m in METHODS if m.endswith("_extreme")]), workers, out, log)
    (out / "pipeline_done.json").write_text(json.dumps(dict(
        wall_seconds=time.time() - start, workers=workers), indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["search", "select", "evaluate", "check-untrained", "pipeline", "smoke"])
    p.add_argument("--family", choices=FAMILIES)
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--method", choices=list(METHODS))
    p.add_argument("--seed", type=int)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--output", type=Path, help="Alternative output root (smoke tests)")
    a = p.parse_args()
    out = a.output or OUT
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "search":
        search(a.family, a.kind, out)
    elif a.mode == "select":
        print(json.dumps(select(a.family, out), indent=2))
    elif a.mode == "evaluate":
        if a.seed not in METHODS[a.method]:
            p.error(f"{a.method} runs with seeds {METHODS[a.method]}")
        evaluate(a.family, a.method, a.seed, out)
    elif a.mode == "check-untrained":
        check_untrained(out)
    elif a.mode == "pipeline":
        if a.workers > 10:
            p.error("At most ten concurrent workers on the shared machine")
        pipeline(a.workers, out)
    else:  # smoke: tiny search and one test seed batch per method, in a separate directory
        for fam in [a.family] if a.family else FAMILIES:
            replay_check(fam, out)
            for k in KINDS:
                search(fam, k, out, limit=1)
            select(fam, out)
            for m, seeds in METHODS.items():
                evaluate(fam, m, seeds[0], out, test_seeds=TEST_SEEDS[:4])


if __name__ == "__main__":
    main()
