"""Hand-written reactive momentum rule, tuned on development data only.

The rule was written down (DESIGN below, also saved as design.json) before any
test-suite evaluation of it. It starts from each family's tuned fixed
configuration and reads one coordinate of the policy's own observation vector:
the cosine between the upcoming minibatch gradient and optimizer memory
(SGD momentum buffer; Adam/AdamW first moment; RMSprop momentum buffer).

  s_0 = 1
  s_t = kappa * s_{t-1}                  if cos_t < tau        (cut memory trust)
  s_t = s_{t-1} + rho * (1 - s_{t-1})    otherwise             (restore toward 1)
  momentum/beta1_t = m_fixed * s_t
  lr_t             = lr_fixed * s_t ** gamma
  second memory and weight decay stay at the fixed configuration.

At t = 0 memory is empty (cosine is 0 by construction), so s_0 = 1 regardless.
The four constants (tau, kappa, rho, gamma) are selected by grid search on the
same eight development episodes and score as the fixed baselines. One extra
candidate, tau = -1 (the rule can never fire), is exactly the fixed
configuration, so development selection can choose "no feedback".
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/prospective_schedules"))
import common as c  # noqa: E402

OUT = ROOT / "results/reactive_rule"
GRID = dict(tau=[-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5], kappa=[0.25, 0.5, 0.75],
            rho=[0.25, 0.5, 1.0], gamma=[0.0, 0.5, 1.0])
OFF = dict(tau=-1.0, kappa=1.0, rho=1.0, gamma=0.0)
COSINE_INDEX = {"SGD": 7, "Adam": 8, "AdamW": 8, "RMSprop": 8}
DESIGN = dict(
    rule=__doc__.split("\n\n")[1:3],
    observation="cosine(gradient, optimizer memory): policy observation index 7 (SGD) or 8 (others)",
    base="tuned fixed configuration of results/optimizer_families/{family}_baseline.json",
    grid=GRID, off_candidate=OFF,
    candidates=int(np.prod([len(v) for v in GRID.values()])) + 1,
    selection="minimum mean development score (log final dev loss + mean log dev-path loss), "
              "seeds 1810000-1810007, two per training regime; same as fixed tuning",
    budget_episodes=8 * (int(np.prod([len(v) for v in GRID.values()])) + 1),
    budget_reference="compute-matched grid: 4,032 (SGD) / 3,840 (adaptive) episodes",
    evaluation="regular (10 orders) and extreme (8 orders) suites, 20 paired episode seeds, batches of 4",
    comparators="saved fixed rows and saved five-seed (338-342) frozen-policy rows; same-engine fixed replay; "
                "warm-start policies as an initialization-matched secondary comparator",
    pre_registered_decision="Reinforcement learning beats the rule on a suite/family if the policy-minus-rule "
                            "biased-mean 95% paired bootstrap CI excludes zero on the positive side; the rule "
                            "beats RL if it excludes zero on the negative side; otherwise neither is established.",
)


class Rule:
    def __init__(self, family, base, tau, kappa, rho, gamma):
        self.family, self.base = family, [float(v) for v in base]
        self.tau, self.kappa, self.rho, self.gamma = tau, kappa, rho, gamma
        self.s = None

    def __call__(self, t, obs):
        cos = obs[:, COSINE_INDEX[self.family]]
        if t == 0:
            self.s = torch.ones(len(obs))
        else:
            self.s = torch.where(cos < self.tau, self.s * self.kappa, self.s + self.rho * (1 - self.s))
        lr0, m0, *rest = self.base
        return (lr0 * self.s ** self.gamma, m0 * self.s, *rest)


def base(family):
    return c.config_values(family, c.fixed_config(family))


def candidates():
    grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    return [dict(OFF, config_id=0)] + [dict(g, config_id=i + 1) for i, g in enumerate(grid)]


def params(cand):
    return {k: float(cand[k]) for k in ("tau", "kappa", "rho", "gamma")}


def tune(family):
    path = OUT / "tuning" / f"{family}.csv"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    c.setup("regular")
    start = time.perf_counter()
    records = []
    for cand in candidates():
        score, rows = c.dev_score(family, controller=Rule(family, base(family), **params(cand)))
        records.append(dict(family=family, **cand, score=score,
                            dev_accuracy=float(np.mean([r["accuracy"] for r in rows])),
                            failures=sum(r["failed"] for r in rows)))
        if cand["config_id"] % 20 == 0:
            print("TUNE", family, cand["config_id"], round(time.perf_counter() - start), flush=True)
    tmp = path.with_suffix(".tmp")
    c.e.writecsv(tmp, records)
    tmp.rename(path)
    best = min(records, key=lambda r: r["score"])
    (OUT / "tuning" / f"{family}_selected.json").write_text(json.dumps(dict(
        family=family, selected=params(best), config_id=best["config_id"], score=best["score"],
        off_score=records[0]["score"], fixed_base=base(family), candidates=len(records),
        episodes=8 * len(records), seconds=time.perf_counter() - start), indent=2) + "\n")


def evaluate(suite, family):
    folder = OUT / suite / family
    if (folder / "done.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    sel = json.loads((OUT / "tuning" / f"{family}_selected.json").read_text())["selected"]
    c.setup(suite)
    data = c.testset()
    start = time.perf_counter()
    records, traces = c.evaluate_rows(family, data, lambda: dict(controller=Rule(family, base(family), **sel)))
    for r in records + traces:
        r.update(family=family, variant="reactive_rule", policy_seed=-1, suite=suite)
    c.e.writecsv(folder / "evaluation.csv", records)
    c.write_gz(folder / "traces.csv.gz", traces)
    (folder / "done.json").write_text(json.dumps(dict(
        suite=suite, family=family, rule=sel, episodes=len(records),
        failures=sum(r["failed"] for r in records), threads=torch.get_num_threads(),
        torch_version=torch.__version__, seconds=time.perf_counter() - start), indent=2) + "\n")
    print("EVALUATED", suite, family, round(time.perf_counter() - start), flush=True)


def pipeline(workers, suites):
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    start = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def one(args):
        with (OUT / "logs" / ("_".join(args).replace("--", "") + ".log")).open("a") as handle:
            return subprocess.call([sys.executable, "-u", __file__, *args, "--output", str(OUT)],
                                   stdout=handle, stderr=subprocess.STDOUT, env=env)

    for stage in ([["tune", "--family", fam] for fam in c.FAMILIES],
                  [["evaluate", "--suite", s, "--family", fam] for s in suites for fam in c.FAMILIES]):
        with ThreadPoolExecutor(workers) as pool:
            codes = list(pool.map(one, stage))
        if any(codes):
            raise SystemExit(f"jobs failed: {codes}")
    (OUT / "pipeline_done.json").write_text(json.dumps(dict(
        wall_seconds=time.time() - start, workers=workers, suites=suites), indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["design", "tune", "evaluate", "pipeline"])
    p.add_argument("--family", choices=c.FAMILIES)
    p.add_argument("--suite", choices=c.SUITES)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--suites", nargs="+", default=c.SUITES)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    global OUT
    if a.output:
        OUT = a.output
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "design":
        (OUT / "design.json").write_text(json.dumps(DESIGN, indent=2) + "\n")
    elif a.mode == "tune":
        tune(a.family)
    elif a.mode == "evaluate":
        evaluate(a.suite, a.family)
    else:
        if a.workers > 6:
            p.error("At most six concurrent workers on the shared machine")
        pipeline(a.workers, a.suites)


if __name__ == "__main__":
    main()
