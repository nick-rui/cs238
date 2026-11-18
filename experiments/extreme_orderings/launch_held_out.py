"""Run or resume the complete held-out ordering suite on a CPU."""

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

from held_out import FAMILIES, ORDERS, OUT, POLICY_SEEDS


def launch(workers, smoke):
    root = OUT / "smoke" if smoke else OUT
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("held_out.py")
    env = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )

    def job(args):
        name = "-".join(args)
        path = logs / f"{name}.log"
        with path.open("a") as log:
            subprocess.run(
                [sys.executable, str(runner), *args, *(["--smoke"] if smoke else [])],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        print(f"Finished {name}", flush=True)

    tasks = [("tune", "--family", family) for family in FAMILIES]
    tasks += [
        ("train", "--family", family, "--held-out", held_out, "--seed", str(seed))
        for seed in POLICY_SEEDS
        for held_out in ORDERS
        for family in FAMILIES
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(job, tasks))
        list(
            pool.map(
                job,
                [
                    ("evaluate", "--family", family, "--held-out", held_out)
                    for family in FAMILIES
                    for held_out in ORDERS
                ],
            )
        )
    job(("analyze",))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    launch(args.workers, args.smoke)
