"""Replay the same methods on the seven original biased orderings (seeds 1920000-1920019).

Gives a same-environment, same-code reference for the replication check in
analyze_generated.py. Writes results/extreme_generated/evaluation_original/.
Run after the generated-suite launch (same 28-process budget).
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generated_eval as G  # noqa: E402

ORDERS = ["single_max", "single_late", "pair_max", "half_block", "staircase", "class_sorted", "alternating_halves"]
SEEDS = list(range(1920000, 1920020))
OUT = G.OUT / "evaluation_original"


def path(family, method, seed, order):
    return OUT / f"{family}_{method}_{seed}" / f"chunk_{order}.csv"


def evaluate(family, method, seed, order):
    target = path(family, method, seed, order)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    G.torch.set_num_threads(1)
    run, meta = G.make_runner(family, method, seed, G.testset())
    rows = []
    for st in range(0, len(SEEDS), 4):  # batches of four, as in the original evaluations
        ss = SEEDS[st:st + 4]
        rr = run(ss, [order] * len(ss))
        for x in rr:
            x.update(order=order, family=family, variant=method, policy_seed=seed, threads=1, **meta)
        rows += rr
    tmp = target.with_suffix(".tmp")
    G.e.writecsv(tmp, rows)
    tmp.replace(target)


def launch(workers, methods):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    jobs = [(f, m, s, o) for f in G.FAMILIES for m in methods for s in G.method_seeds(m) for o in ORDERS
            if not path(f, m, s, o).exists()]
    print(len(jobs), "jobs", flush=True)
    log = (G.OUT / "logs" / "originals.log").open("a")

    def job(a):
        f, m, s, o = a
        subprocess.run([sys.executable, __file__, "evaluate", f, m, str(s), o], cwd=G.ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=env)

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(job, j) for j in jobs]):
            fut.result()
    print("ORIGINALS COMPLETE", time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["evaluate", "launch"])
    p.add_argument("args", nargs="*")
    p.add_argument("--workers", type=int, default=28)
    p.add_argument("--methods", nargs="*")
    a = p.parse_args()
    if a.mode == "evaluate":
        f, m, s, o = a.args
        evaluate(f, m, int(s), o)
    else:
        assert a.workers <= 28
        launch(a.workers, a.methods or list(G.METHODS))
