"""Prospective (development-fitted) open-loop schedule controls.

(a) dev_mean: each saved policy's mean schedule over its 16 development
    episodes (DEVSEEDS x four training regimes), replayed open-loop on test.
(b) Standard LR schedules (cosine, step, linear-warmup cosine) whose peak LR and
    other hyperparameters are searched on the existing fixed grid axes and
    selected on the same 8-episode development score as the fixed baselines.
No test metric is read before every schedule is fixed; see README.md.
"""

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

ROOT = c.ROOT
OUT = ROOT / "results/prospective_schedules"
H = 40
KINDS = ["cosine", "step", "warmup_cosine"]
SGD_AXES = dict(lr=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3], momentum=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95],
                weight_decay=[0.0, 1e-6, 1e-4, 0.01])
ADAPTIVE_AXES = dict(lr=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03], momentum=[0.0, 0.5, 0.9, 0.95],
                     second_decay=[0.9, 0.99, 0.999, 0.9999], weight_decay=[0.0, 1e-4, 0.01])
ADAPTIVE_PER_KIND = 160  # 3 kinds x 160 x 8 episodes = 3,840 = compute-matched adaptive budget
SAMPLE_SEED = 238


def lr_factor(kind, t):
    """LR multiplier for control step t (each step = five classifier updates)."""
    if kind == "cosine":
        return 0.5 * (1 + math.cos(math.pi * t / H))
    if kind == "step":  # x0.1 at 50% and again at 75% of training
        return 1.0 if t < 20 else (0.1 if t < 30 else 0.01)
    if kind == "warmup_cosine":  # linear warmup over the first 10% (20 updates)
        return (t + 1) / 4 if t < 4 else 0.5 * (1 + math.cos(math.pi * (t - 4) / (H - 4)))
    if kind == "constant":
        return 1.0
    raise ValueError(kind)


def make_schedule(family, kind, config):
    rest = [config[k] for k in c.columns(family)[1:]]
    return [[config["lr"] * lr_factor(kind, t)] + rest for t in range(H)]


def candidates(family, kind):
    """All SGD grid points (144/kind); a seeded 160-of-288 sample for adaptive families."""
    axes = SGD_AXES if family == "SGD" else ADAPTIVE_AXES
    grid = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
    if family != "SGD":
        rng = np.random.default_rng([SAMPLE_SEED, c.FAMILIES.index(family), KINDS.index(kind)])
        grid = [grid[i] for i in sorted(rng.choice(len(grid), ADAPTIVE_PER_KIND, replace=False))]
    return [dict(kind=kind, config_id=i, **g) for i, g in enumerate(grid)]


# ---------------------------------------------------------------- development fitting

def fit(family, seed):
    folder = OUT / "dev_mean" / f"{family}_{seed}"
    if (folder / "fit.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    c.setup("regular")
    start = time.perf_counter()
    pol, update = c.load_policy(family, seed)
    rows, traces = [], []
    for st in range(0, 16, 8):
        rr, tr = c.rollout(family, c.b.DEVSEEDS[st:st + 8], c.b.DEVORDERS[st:st + 8],
                           (c.e.xd, c.e.yd), pol=pol, trace=True)
        rows += rr
        traces += tr
    expected = {(s, o, step) for s, o in zip(c.b.DEVSEEDS, c.b.DEVORDERS) for step in range(5, 201, 5)}
    if {(int(r["seed"]), r["order"], int(r["step"])) for r in traces} != expected or len(traces) != len(expected):
        raise ValueError("Development traces incomplete")
    schedule = []
    for step in range(5, 201, 5):
        group = [r for r in traces if int(r["step"]) == step]
        schedule.append(dict(step=step, source_episodes=len(group),
                             **{k: math.fsum(float(r[k]) for r in group) / len(group) for k in c.columns(family)}))
    c.e.writecsv(folder / "schedule.csv", schedule)
    values = [[r[k] for k in c.columns(family)] for r in schedule]
    replay_score, replay_rows = c.dev_score(family, n=16, schedule=values)
    for r in rows:
        r.update(method="policy")
    for r in replay_rows:
        r.update(method="dev_mean_replay")
    c.e.writecsv(folder / "development.csv", rows + replay_rows)
    c.write_gz(folder / "development_traces.csv.gz", traces)
    (folder / "fit.json").write_text(json.dumps(dict(
        family=family, policy_seed=seed, checkpoint_update=update,
        checkpoint_sha256=c.sha256(c.checkpoint(family, seed)),
        source="16 development episodes: seeds 1810000-1810015 x mixed/single_train/pair_train/double_train",
        pooling="arithmetic mean of actual settings at each control step over 16 development episodes",
        policy_dev_score=float(np.mean([c.b.score(r) for r in rows])), replay_dev_score=replay_score,
        test_data_used=False, seconds=time.perf_counter() - start,
    ), indent=2) + "\n")
    print("FIT", family, seed, flush=True)


def search(family, kind):
    path = OUT / "search" / f"{family}_{kind}.csv"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    c.setup("regular")
    start = time.perf_counter()
    records = []
    for cand in candidates(family, kind):
        score, rows = c.dev_score(family, schedule=make_schedule(family, kind, cand))
        records.append(dict(family=family, **cand, score=score,
                            dev_accuracy=float(np.mean([r["accuracy"] for r in rows])),
                            failures=sum(r["failed"] for r in rows)))
        if cand["config_id"] % 24 == 0:
            print("SEARCH", family, kind, cand["config_id"], round(time.perf_counter() - start), flush=True)
    tmp = path.with_suffix(".tmp")
    c.e.writecsv(tmp, records)
    tmp.rename(path)
    (path.with_suffix(".json")).write_text(json.dumps(dict(
        candidates=len(records), episodes=8 * len(records), seconds=time.perf_counter() - start), indent=2))


def select(family):
    path = OUT / "selected" / f"{family}.json"
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    c.setup("regular")
    chosen = {}
    for kind in KINDS:
        rows = c.readcsv(OUT / "search" / f"{family}_{kind}.csv")
        best = min(rows, key=lambda r: float(r["score"]))  # first minimum on ties
        chosen[kind] = dict(score=float(best["score"]), config_id=int(best["config_id"]),
                            config={k: float(best[k]) for k in c.columns(family)})
    overall = min(KINDS, key=lambda k: chosen[k]["score"])
    fixed = c.fixed_config(c.FAMILIES[c.FAMILIES.index(family)])
    fixed_values = dict(zip(c.columns(family), c.config_values(family, fixed)))
    # Same-engine development score of the constant fixed configuration, for reference.
    fixed_score, _ = c.dev_score(family, schedule=make_schedule(family, "constant", fixed_values))
    result = dict(family=family, kinds=chosen, overall=overall, fixed_config=fixed_values,
                  fixed_dev_score_same_engine=fixed_score,
                  search_episodes=sum(8 * len(candidates(family, k)) for k in KINDS))
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


# ---------------------------------------------------------------- test evaluation

def method_schedule(family, method, seed=None):
    if method == "dev_mean":
        rows = c.readcsv(OUT / "dev_mean" / f"{family}_{seed}" / "schedule.csv")
        return [[float(r[k]) for k in c.columns(family)] for r in rows]
    if method == "fixed_engine":
        fixed = c.fixed_config(family)
        return make_schedule(family, "constant", dict(zip(c.columns(family), c.config_values(family, fixed))))
    sel = json.loads((OUT / "selected" / f"{family}.json").read_text())
    return make_schedule(family, method, sel["kinds"][method]["config"])


def evaluate(suite, family, method, seed=None):
    name = f"{family}_{method}" + (f"_{seed}" if seed is not None else "")
    folder = OUT / suite / name
    if (folder / "done.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    c.setup(suite)
    data = c.testset()
    schedule = method_schedule(family, method, seed)
    start = time.perf_counter()
    records, traces = c.evaluate_rows(family, data, lambda: dict(schedule=schedule))
    for r in traces:
        want = schedule[int(r["step"]) // 5 - 1]
        got = [float(r[k]) for k in c.columns(family)]
        if not np.allclose(got, np.float32(want), rtol=0, atol=0):
            raise AssertionError(("feedback changed schedule", name, r["step"]))
    for r in records + traces:
        r.update(family=family, variant=method, policy_seed=-1 if seed is None else seed, suite=suite)
    c.e.writecsv(folder / "evaluation.csv", records)
    c.write_gz(folder / "traces.csv.gz", traces)
    c.e.writecsv(folder / "schedule.csv", [dict(step=5 * (t + 1), **dict(zip(c.columns(family), row)))
                                          for t, row in enumerate(schedule)])
    (folder / "done.json").write_text(json.dumps(dict(
        suite=suite, family=family, method=method, policy_seed=seed, episodes=len(records),
        failures=sum(r["failed"] for r in records), threads=c.torch.get_num_threads(),
        torch_version=c.torch.__version__, seconds=time.perf_counter() - start), indent=2) + "\n")
    print("EVALUATED", suite, name, round(time.perf_counter() - start), flush=True)


# ---------------------------------------------------------------- pipeline

def run_jobs(jobs, workers, log):
    py = sys.executable
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def one(args):
        with (OUT / "logs" / ("_".join(args[:4]).replace("--", "") + ".log")).open("a") as handle:
            code = subprocess.call([py, "-u", __file__, *args, "--output", str(OUT)], stdout=handle, stderr=subprocess.STDOUT, env=env)
        print(json.dumps(dict(job=args, code=code, time=time.strftime("%H:%M:%S"))), file=log, flush=True)
        return code

    with ThreadPoolExecutor(workers) as pool:
        codes = list(pool.map(one, jobs))
    if any(codes):
        raise SystemExit(f"{sum(1 for x in codes if x)} jobs failed")


def pipeline(workers, suites):
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    start = time.time()
    with (OUT / "logs" / "pipeline.jsonl").open("a") as log:
        stage1 = [["search", "--family", fam, "--kind", k] for fam in c.FAMILIES for k in KINDS]
        stage1 += [["fit", "--family", fam, "--seed", str(s)] for fam in c.FAMILIES for s in c.POLICY_SEEDS]
        stage1 += [["evaluate", "--suite", su, "--family", fam, "--method", "fixed_engine"]
                   for su in suites for fam in c.FAMILIES]
        run_jobs(stage1, workers, log)
        run_jobs([["select", "--family", fam] for fam in c.FAMILIES], workers, log)
        stage2 = [["evaluate", "--suite", su, "--family", fam, "--method", "dev_mean", "--seed", str(s)]
                  for su in suites for fam in c.FAMILIES for s in c.POLICY_SEEDS]
        stage2 += [["evaluate", "--suite", su, "--family", fam, "--method", k]
                   for su in suites for fam in c.FAMILIES for k in KINDS]
        run_jobs(stage2, workers, log)
    (OUT / "pipeline_done.json").write_text(json.dumps(dict(
        wall_seconds=time.time() - start, workers=workers, suites=suites), indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["fit", "search", "select", "evaluate", "pipeline"])
    p.add_argument("--family", choices=c.FAMILIES)
    p.add_argument("--seed", type=int, choices=c.POLICY_SEEDS)
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--suite", choices=c.SUITES)
    p.add_argument("--method", choices=["dev_mean", "fixed_engine"] + KINDS)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--suites", nargs="+", default=c.SUITES)
    p.add_argument("--output", type=Path, help="Alternative output root (smoke tests)")
    a = p.parse_args()
    global OUT
    if a.output:
        OUT = a.output
    if a.mode == "fit":
        fit(a.family, a.seed)
    elif a.mode == "search":
        search(a.family, a.kind)
    elif a.mode == "select":
        print(json.dumps(select(a.family), indent=2))
    elif a.mode == "evaluate":
        evaluate(a.suite, a.family, a.method, a.seed)
    else:
        if a.workers > 6:
            p.error("At most six concurrent workers on the shared machine")
        pipeline(a.workers, a.suites)


if __name__ == "__main__":
    main()
