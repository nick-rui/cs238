"""Policy-control ablations with the original training and evaluation protocol."""

import argparse
import csv
import gzip
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer_families"))
import family_engine as f

b, e = f.base, f.e
OUT = e.ROOT / "results/ablations"
START_OUT = e.ROOT / "results/ablations_start_values"
SOURCE = e.ROOT / "results/optimizer_families"
KINDS = ["lr_only", "fixed_memory", "no_feedback"]
JOBS = [(family, kind, seed) for family in ["SGD", "Adam"] for kind in KINDS for seed in [338, 339]]
START_JOBS = [job for job in JOBS if job[1] != "no_feedback"]


def policy(family, kind, seed):
    return e.policy(kind, seed) if family == "SGD" else f.policy(family, kind, seed)


def rollout(family, kind, seeds, orders, pol, *, held_source="baseline", **kwargs):
    held = None
    if held_source == "baseline":
        held = json.loads((SOURCE / f"{family}_baseline.json").read_text())["config"]
    if family == "SGD":
        rows, traces, lp, rewards, _ = e.rollout(pol, kind, seeds, orders, held=held, **kwargs)
        return rows, b.annotate(traces), lp, rewards
    return f.rollout(family, kind, seeds, orders, pol, held=held, **kwargs)


def train(family, kind, seed, updates, root, held_source="baseline"):
    folder = root / f"{family}_{kind}_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "done.json").exists():
        return
    torch.manual_seed(seed)
    pol = policy(family, kind, seed)
    opt = torch.optim.Adam(pol.parameters(), lr=0.003)
    training, dev = [], []
    best = math.inf
    start = time.perf_counter()
    for update in range(updates + 1):
        if update in e.CHECKS or update == updates:
            rr = []
            for st in range(0, 16, 8):
                rr += rollout(family, kind, b.DEVSEEDS[st:st+8], b.DEVORDERS[st:st+8], pol, held_source=held_source, evaluation=(e.xd, e.yd))[0]
            value = float(np.mean([b.score(r) for r in rr]))
            for r in rr:
                r.update(update=update, family=family, kind=kind, policy_seed=seed)
            dev += rr
            state = dict(policy=pol.state_dict(), update=update, family=family, kind=kind, seed=seed)
            torch.save(state, folder / f"checkpoint_{update}.pt")
            if value < best:
                best = value
                torch.save(state, folder / "best.pt")
            e.writecsv(folder / "development.csv", dev)
            print(json.dumps(dict(job=folder.name, update=update, score=value, seconds=time.perf_counter()-start)), flush=True)
        if update == updates:
            break
        seeds = [5000000 + seed * 10000 + update * 8 + i for i in range(8)]
        rr, _, logp, rewards = rollout(family, kind, seeds, b.TRAIN * 2, pol, held_source=held_source, training=True)
        returns = rewards.flip(1).cumsum(1).flip(1)
        adv = returns - returns[torch.tensor([4, 5, 6, 7, 0, 1, 2, 3])]
        opt.zero_grad()
        (-(logp * adv.detach()).mean()).backward()
        assert all(torch.isfinite(q.grad).all() for q in pol.parameters())
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1)
        opt.step()
        training += [dict(update=update+1, seed=r["seed"], order=r["order"], reward=r["return_value"], failed=r["failed"]) for r in rr]
    with gzip.open(folder / "training.csv.gz", "wt") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(training[0]))
        writer.writeheader()
        writer.writerows(training)
    (folder / "done.json").write_text(json.dumps(dict(updates=updates, episodes=updates*8, threads=torch.get_num_threads(), seconds=time.perf_counter()-start, failed=sum(r["failed"] for r in training)), indent=2))


def evaluation_complete(folder):
    expected = len(b.ORDERS) * len(b.TESTSEEDS)
    try:
        with (folder / "evaluation.csv").open() as stream:
            if sum(1 for _ in csv.DictReader(stream)) != expected:
                return False
        with gzip.open(folder / "traces.csv.gz", "rt") as stream:
            return sum(1 for _ in csv.DictReader(stream)) == expected * e.H
    except (OSError, EOFError, zlib.error, csv.Error, UnicodeError):
        return False


def evaluate(jobs, held_source="baseline"):
    root, all_jobs = (START_OUT, START_JOBS) if held_source == "start" else (OUT, JOBS)
    test = e.MNIST(str(e.DATA_DIR), train=False, download=True)
    testset = (test.data.reshape(-1, 784).float()/255, test.targets)
    for family, kind, seed in jobs:
        folder = root / f"{family}_{kind}_{seed}"
        path = folder / "evaluation.csv"
        if evaluation_complete(folder):
            continue
        saved = torch.load(folder / "best.pt", weights_only=True)
        pol = policy(family, kind, seed)
        pol.load_state_dict(saved["policy"])
        results, traces = [], []
        for order in b.ORDERS:
            for st in range(0, 20, 4):
                rr, tr, _, _ = rollout(family, kind, b.TESTSEEDS[st:st+4], [order]*4, pol, held_source=held_source, evaluation=testset, trace=True)
                for r in rr:
                    r.update(family=family, variant=kind, policy_seed=seed, checkpoint=saved["update"])
                for r in tr:
                    r.update(family=family, variant=kind, policy_seed=seed)
                results += rr
                traces += tr
        trace_path = folder / "traces.csv.gz"
        trace_temp = folder / "traces.csv.gz.tmp"
        with gzip.open(trace_temp, "wt") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(traces[0]))
            writer.writeheader()
            writer.writerows(traces)
        trace_temp.replace(trace_path)
        evaluation_temp = folder / "evaluation.csv.tmp"
        e.writecsv(evaluation_temp, results)
        evaluation_temp.replace(path)
        print("EVALUATED", family, kind, seed, saved["update"], flush=True)
    for family in ["SGD", "Adam"]:
        paths = [root / f"{fam}_{kind}_{seed}/evaluation.csv" for fam, kind, seed in all_jobs if fam == family]
        if all(evaluation_complete(p.parent) for p in paths):
            rows = []
            for path in paths:
                with path.open() as stream:
                    rows += list(csv.DictReader(stream))
            e.writecsv(root / f"{family}_evaluation.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate", "smoke"])
    parser.add_argument("--job", choices=[f"{f}_{k}_{s}" for f, k, s in JOBS])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--held-source", choices=["baseline", "start"], default="baseline")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root, all_jobs = (START_OUT, START_JOBS) if args.held_source == "start" else (OUT, JOBS)
    jobs = [job for job in all_jobs if args.job is None or "_".join(map(str, job)) == args.job]
    if not jobs:
        parser.error("--held-source start supports only lr_only and fixed_memory")
    if args.mode == "evaluate":
        evaluate(jobs, held_source=args.held_source)
    else:
        for job in jobs:
            train(*job, updates=1 if args.mode == "smoke" else 500, root=root / "smoke" if args.mode == "smoke" else root, held_source=args.held_source)
