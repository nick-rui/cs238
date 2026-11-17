"""Run the fifth experiment with at most eight one-thread jobs."""
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "results/extreme_retrained/logs"
FAMILIES = ["SGD", "Adam", "AdamW", "RMSprop"]


def job(arguments):
    name = "_".join(arguments)
    with (LOGS / f"{name}.log").open("w") as log:
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("retrain.py")), *arguments],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                     OPENBLAS_NUM_THREADS="1"),
        )
    print("COMPLETED", name, flush=True)


if __name__ == "__main__":
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = [[ "train", "--family", family, "--seed", str(seed)]
            for seed in [338, 339] for family in FAMILIES]
    jobs += [["tune", "--family", family] for family in FAMILIES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(job, jobs))
        list(pool.map(job, [["evaluate", "--family", family] for family in FAMILIES]))
    job(["analyze"])
    print("EXPERIMENT COMPLETE", flush=True)
