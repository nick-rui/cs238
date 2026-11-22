"""Check that the generated-suite runners replay committed extreme-suite rows.

Runs each method's runner (from generated_eval.py) on two original orderings x
four episode seeds and compares final accuracy with the committed evaluation
records. Writes results/extreme_generated/replay_check.json.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generated_eval as G  # noqa: E402

SEEDS = [1920000, 1920001, 1920002, 1920003]
ROOT = G.ROOT


def saved(path, variant=None, policy_seed=None):
    rows = csv.DictReader(open(ROOT / path))
    return {(r["order"], int(r["seed"])): float(r["accuracy"]) for r in rows
            if variant is None or (r["variant"] == variant and r["policy_seed"] == str(policy_seed))}


def main():
    ts = G.testset()
    sources = {
        "warm_extreme": (338, "results/extreme_warm_start/evaluation/{f}_warm_extreme_338.csv", None),
        "policy_retrained": (340, "results/extreme_warm_start/evaluation/{f}_policy_retrained_340.csv", None),
        "fixed_extreme": (-1, "results/extreme_warm_start/evaluation/{f}_fixed_retuned_-1.csv", None),
        "cosine_regular": (-1, "results/prospective_schedules/extreme/{f}_cosine/evaluation.csv", None),
        "policy_frozen": (339, "results/extreme_orderings/{f}_evaluation.csv", "policy"),
        "fixed_regular": (-1, "results/extreme_orderings/{f}_evaluation.csv", "fixed"),
    }
    records = []
    for family in G.FAMILIES:
        for method, (seed, path, variant) in sources.items():
            ref = saved(path.format(f=family), variant, seed)
            run, _ = G.make_runner(family, method, seed, ts)
            for order in ("pair_max", "staircase"):
                rows = run(SEEDS, [order] * len(SEEDS))
                diffs = [abs(x["accuracy"] - ref[order, s]) for x, s in zip(rows, SEEDS)]
                records.append(dict(family=family, method=method, policy_seed=seed, order=order,
                                    source=path.format(f=family), max_abs_accuracy=max(diffs),
                                    exact=sum(d == 0 for d in diffs), episodes=len(diffs)))
                print(json.dumps(records[-1]), flush=True)
    out = G.OUT / "replay_check.json"
    out.write_text(json.dumps(dict(seeds=SEEDS, worst=max(r["max_abs_accuracy"] for r in records),
                                   records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
