"""Run the CNN CIFAR-10 pipeline on one GPU, skipping completed work.

Stage 1 runs policy training (two seeds per family) and fixed-grid tuning as
concurrent GPU processes; stage 2 evaluates each family once its three inputs
exist. Logs go to results/cnn_cifar10/logs/.
"""

import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cnn_cifar10"
RUN = [sys.executable, "-W", "ignore", str(Path(__file__).with_name("run.py"))]


def folder(family, seed):
    return OUT / (f"gradient_{seed}" if family == "SGD" else f"{family}_gradient_{seed}")


def tuned(family):
    return (OUT / ("baseline_configs.json" if family == "SGD" else f"{family}_baseline.json")).exists()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--families", nargs="+", default=["SGD", "Adam"])
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    jobs = []
    for family in a.families:
        for seed in [338, 339]:
            if not (folder(family, seed) / "done.json").exists():
                jobs.append((f"train_{family}_{seed}",
                             ["train", "--family", family, "--seed", str(seed), "--updates", str(a.updates)]))
        if not tuned(family):
            jobs.append((f"tune_{family}", ["tune", "--family", family]))
    for family in a.families:
        if not (OUT / f"{family}_evaluation.csv").exists():
            jobs.append((f"evaluate_{family}", ["evaluate", "--family", family]))
    running = {}
    start = time.time()
    while jobs or running:
        for name, proc in list(running.items()):
            if proc.poll() is not None:
                print(f"{time.time() - start:7.0f}s DONE {name} rc={proc.returncode}", flush=True)
                if proc.returncode:
                    raise SystemExit(f"{name} failed; see logs/{name}.log")
                del running[name]
        for name, args in list(jobs):
            if len(running) >= a.workers:
                break
            if name.startswith("evaluate_"):
                family = name.split("_", 1)[1]
                if not (tuned(family) and all((folder(family, s) / "done.json").exists() for s in [338, 339])):
                    continue
            log = (OUT / "logs" / f"{name}.log").open("a")
            running[name] = subprocess.Popen(RUN + args, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            jobs.remove((name, args))
            print(f"{time.time() - start:7.0f}s START {name}", flush=True)
        time.sleep(10)
    subprocess.run(RUN + ["summarize"], check=True, cwd=ROOT)
    print(f"ALL DONE {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
