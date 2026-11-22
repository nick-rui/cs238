"""Secondary (not pre-registered as headline): best development candidate that can fire.

Where development selection picks the never-firing candidate (tau = -1, i.e.
the fixed configuration), the headline rule is just fixed. This evaluates the
best-scoring candidate with tau > -1, selected from the same tuning CSV
(development only), so the effect of actually reacting is visible on test.

Usage: python experiments/reactive_rule/firing.py --suite regular --family SGD
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("reactive_rule_run", ROOT / "experiments/reactive_rule/run.py")
rule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rule)
c = rule.c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=c.SUITES, required=True)
    p.add_argument("--family", choices=c.FAMILIES, required=True)
    a = p.parse_args()
    out = ROOT / "results/reactive_rule/secondary_firing" / a.suite / a.family
    if (out / "done.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in c.readcsv(ROOT / "results/reactive_rule/tuning" / f"{a.family}.csv") if float(r["tau"]) > -1]
    best = min(rows, key=lambda r: float(r["score"]))
    sel = rule.params(best)
    c.setup(a.suite)
    data = c.testset()
    start = time.perf_counter()
    records, traces = c.evaluate_rows(a.family, data, lambda: dict(controller=rule.Rule(a.family, rule.base(a.family), **sel)))
    for r in records + traces:
        r.update(family=a.family, variant="reactive_rule_firing", policy_seed=-1, suite=a.suite)
    c.e.writecsv(out / "evaluation.csv", records)
    c.write_gz(out / "traces.csv.gz", traces)
    (out / "done.json").write_text(json.dumps(dict(
        suite=a.suite, family=a.family, rule=sel, config_id=int(best["config_id"]), dev_score=float(best["score"]),
        episodes=len(records), failures=sum(r["failed"] for r in records),
        threads=c.torch.get_num_threads(), seconds=time.perf_counter() - start), indent=2) + "\n")


if __name__ == "__main__":
    main()
