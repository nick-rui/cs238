"""Replay pooled mean saved-policy schedules without online feedback.

One schedule per optimizer and policy seed, pooled over ALL regular conditions.
This is retrospective: the source traces are from the evaluation episodes, not
an independent schedule-fitting split. No evaluation metrics select settings.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/optimizer_families"))
import family_engine as f

b, e = f.base, f.e
SOURCE = ROOT / "results/optimizer_families"
DEFAULT_OUT = ROOT / "results/open_loop/regular"
FAMILIES = ["SGD", "Adam", "AdamW", "RMSprop"]
POLICY_SEEDS = [338, 339]
STEPS = list(range(5, 201, 5))
METRICS = ["accuracy", "eval_loss", "path_log_loss"]


def readcsv(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as file:
        return list(csv.DictReader(file))


def trace_path(family):
    path = SOURCE / f"{family}_traces.csv"
    return path if path.exists() else path.with_suffix(".csv.gz")


def columns(family):
    return ["lr", "momentum"] + ([] if family == "SGD" else ["second_decay"]) + ["weight_decay"]


def mean_schedule(family, policy_seed, traces):
    rows = [r for r in traces if r["variant"] == "gradient" and int(r["policy_seed"]) == policy_seed]
    expected = {(s, o, t) for s in b.TESTSEEDS for o in b.ORDERS for t in STEPS}
    keys = [(int(r["seed"]), r["order"], int(r["step"])) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Source must contain each regular episode/control step exactly once")
    result = []
    for step in STEPS:
        group = [r for r in rows if int(r["step"]) == step]
        values = {key: math.fsum(float(r[key]) for r in group) / len(group) for key in columns(family)}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite source hyperparameter")
        result.append(dict(step=step, source_episodes=len(group), **values))
    return result


def replay(family, schedule, seeds, orders, testset=None, trace=True, pol=None):
    # pol is accepted only so checks can prove it is never called.
    values = [[r[key] for key in columns(family)] for r in schedule]
    if family == "SGD":
        return e.rollout(pol, "gradient", seeds, orders, evaluation=testset,
                         schedules={order: values for order in orders}, trace=trace)
    return f.rollout(family, "gradient", seeds, orders, pol=pol,
                     evaluation=testset, schedule=values, trace=trace)


def verify_replay(family, schedule, traces):
    lookup = {int(row["step"]): row for row in schedule}
    for row in traces:
        for key in columns(family):
            # Engines intentionally use float32, while schedule means use float64.
            if float(row[key]) != float(np.float32(lookup[int(row["step"])][key])):
                raise AssertionError((family, row["step"], key, "feedback changed schedule"))


def write_traces(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(family, policy_seed, out, testset):
    # Never overwrite even a partially completed job; use a new --output to retry.
    folder = out / f"{family}_gradient_{policy_seed}"
    folder.mkdir(parents=True, exist_ok=False)
    source_trace = trace_path(family)
    schedule = mean_schedule(family, policy_seed, readcsv(source_trace))
    e.writecsv(folder / "schedule.csv", schedule)
    checkpoint = (ROOT / "results" / ("recovery_study" if family == "SGD" else "optimizer_families")
                  / (f"gradient_{policy_seed}" if family == "SGD" else f"{family}_gradient_{policy_seed}") / "best.pt")
    records, traces = [], []
    start = time.perf_counter()
    for order in b.ORDERS:
        for st in range(0, len(b.TESTSEEDS), 4):
            seeds = b.TESTSEEDS[st:st + 4]
            rr, tr, *_ = replay(family, schedule, seeds, [order] * len(seeds), testset)
            verify_replay(family, schedule, tr)
            for row in rr + tr:
                row.update(family=family, variant="open_loop", policy_seed=policy_seed)
            records += rr
            traces += tr
    e.writecsv(folder / "evaluation.csv", records)
    write_traces(folder / "traces.csv.gz", traces)
    sources = [source_trace, SOURCE / f"{family}_evaluation.csv", checkpoint]
    (folder / "provenance.json").write_text(json.dumps(dict(
        family=family, policy_seed=policy_seed,
        pooling="arithmetic mean of actual settings at each control step over all 10 orders x 20 episode seeds",
        schedule_scope="one schedule per policy, identical across all evaluation conditions and episode seeds",
        retrospective=True, online_policy_calls=0, source_episodes=200,
        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in sources},
        torch_version=torch.__version__, numpy_version=np.__version__, threads=torch.get_num_threads(),
        seconds=time.perf_counter() - start, episodes=len(records), trace_rows=len(traces),
    ), indent=2) + "\n")
    print(f"EVALUATED {family} {policy_seed}: {len(records)} episodes in {time.perf_counter()-start:.1f}s", flush=True)


def comparisons(out):
    paired = []
    for family in FAMILIES:
        original = readcsv(SOURCE / f"{family}_evaluation.csv")
        closed = {(int(r["policy_seed"]), int(r["seed"]), r["order"]): r
                  for r in original if r["variant"] == "gradient"}
        fixed = {(int(r["seed"]), r["order"]): r for r in original if r["variant"] == "fixed"}
        for ps in POLICY_SEEDS:
            path = out / f"{family}_gradient_{ps}" / "evaluation.csv"
            if not path.exists():
                continue
            rows = readcsv(path)
            expected = {(s, o) for s in b.TESTSEEDS for o in b.ORDERS}
            keys = [(int(r["seed"]), r["order"]) for r in rows]
            if len(keys) != len(expected) or set(keys) != expected:
                raise ValueError(f"Incomplete/duplicated replay: {path}")
            for row in rows:
                seed, order = int(row["seed"]), row["order"]
                ref, baseline = closed[ps, seed, order], fixed[seed, order]
                result = dict(family=family, policy_seed=ps, seed=seed, order=order,
                              open_loop_failed=row["failed"], closed_loop_failed=ref["failed"])
                for key in METRICS:
                    result.update({f"open_loop_{key}": float(row[key]),
                                   f"closed_loop_{key}": float(ref[key]),
                                   f"fixed_{key}": float(baseline[key]),
                                   f"closed_minus_open_{key}": float(ref[key]) - float(row[key])})
                paired.append(result)
    if not paired:
        raise ValueError("No completed replay jobs")
    e.writecsv(out / "paired_differences.csv", paired)
    for filename, by_seed, by_order in [("summary.csv", False, False),
                                        ("policy_seed_summary.csv", True, False),
                                        ("condition_summary.csv", True, True)]:
        summary = []
        for family in FAMILIES:
            for ps in POLICY_SEEDS if by_seed else [None]:
                groups = [(o, [o]) for o in b.ORDERS] if by_order else [
                    ("All", b.ORDERS), ("Random", ["mixed"]), ("Biased", b.ORDERS[1:])]
                for label, orders in groups:
                    rows = [r for r in paired if r["family"] == family and r["order"] in orders
                            and (ps is None or r["policy_seed"] == ps)]
                    if not rows:
                        continue
                    result = dict(family=family, ordering=label, episodes=len(rows),
                                  policy_seeds=len({r["policy_seed"] for r in rows}))
                    if by_seed:
                        result["policy_seed"] = ps
                    for key in [k for k in rows[0] if any(k.startswith(prefix) for prefix in
                                ["open_loop_", "closed_loop_", "fixed_", "closed_minus_open_"]) and not k.endswith("failed")]:
                        result[key] = float(np.mean([r[key] for r in rows]))
                    for mode in ["open_loop", "closed_loop"]:
                        result[f"{mode}_failures"] = sum(str(r[f"{mode}_failed"]) == "True" for r in rows)
                    summary.append(result)
        e.writecsv(out / filename, summary)
    print(json.dumps(dict(paired_episodes=len(paired), output=str(out)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["evaluate", "summarize"])
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--policy-seed", type=int, choices=POLICY_SEEDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.mode == "summarize":
        comparisons(args.output)
        return
    test = e.MNIST(str(e.DATA_DIR), train=False, download=True)
    testset = (test.data.reshape(-1, 784).float() / 255, test.targets)
    for family in [args.family] if args.family else FAMILIES:
        for ps in [args.policy_seed] if args.policy_seed else POLICY_SEEDS:
            evaluate(family, ps, args.output, testset)


if __name__ == "__main__":
    main()
