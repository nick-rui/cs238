"""Run the fifth experiment with at most eight one-thread jobs."""

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

import retrain as r
from summarize_retrain import FROZEN_CACHE


def job(script, arguments, log_path):
    name = "_".join(arguments)
    with log_path.open("w") as log:
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name(script)), *arguments],
            cwd=r.e.ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env=dict(
                os.environ,
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            ),
        )
    print("COMPLETED", name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-seeds", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--frozen-results", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if (args.frozen_results is not None or args.smoke) and not args.five_seeds:
        parser.error("--frozen-results and --smoke require --five-seeds")
    out = r.EXTENDED_OUT if args.five_seeds else r.OUT
    smoke = ["--smoke"] if args.smoke else []
    if args.smoke:
        out /= "smoke"
    if args.five_seeds:
        if args.frozen_results is not None and not args.frozen_results.is_dir():
            parser.error("--frozen-results must point to an existing directory")
        if args.frozen_results is None and not (out / FROZEN_CACHE).is_file():
            parser.error(
                "--frozen-results is required until the saved frozen rows are imported"
            )
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    policy_seeds = r.ADDED_POLICY_SEEDS if args.five_seeds else r.ORIGINAL_POLICY_SEEDS
    extension = ["--five-seeds"] if args.five_seeds else []

    def run(arguments):
        job(
            "retrain.py",
            arguments + extension + smoke,
            logs / ("_".join(arguments) + ".log"),
        )

    jobs = [
        ["train", "--family", family, "--seed", str(seed)]
        for seed in policy_seeds
        for family in r.FAMILIES
    ]
    if not args.five_seeds:
        jobs += [["tune", "--family", family] for family in r.FAMILIES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, jobs))
        list(pool.map(run, [["evaluate", "--family", family] for family in r.FAMILIES]))
    if args.five_seeds:
        frozen = (
            ["--frozen-results", str(args.frozen_results.resolve())]
            if args.frozen_results
            else []
        )
        job("summarize_retrain.py", smoke + frozen, logs / "analyze.log")
    else:
        run(["analyze"])
    print("EXPERIMENT COMPLETE", flush=True)
