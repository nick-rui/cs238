"""Compare isolated fixed-baseline evaluations with the frozen saved records."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["SGD", "Adam", "AdamW", "RMSprop"]


def read(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def paired_accuracy(rows, seeds, orders, policy=False):
    expected = {(str(s), o, str(ps)) for s in seeds for o in orders for ps in ([338, 339] if policy else [-1])}
    actual = {(r["seed"], r["order"], r["policy_seed"]) for r in rows}
    if not expected or actual != expected or len(rows) != len(expected):
        raise ValueError("Missing, extra or duplicate paired evaluation records")
    return 100 * mean(float(r["accuracy"]) for r in rows)


def analyze(folder):
    aggregates, configurations = [], {}
    for family in FAMILIES:
        old = json.loads((ROOT / "results/optimizer_families" / f"{family}_baseline.json").read_text())["config"]
        new = json.loads((folder / f"{family}_baseline.json").read_text())
        configurations[family] = dict(old=old, new=new["config"], changed=old != new["config"], search_seconds=new["seconds"], episodes=new["episodes"])
        for suite, source, seeds, groups in [
            ("regular", "optimizer_families", range(1900000, 1900020), ["Random", "Biased"]),
            ("extreme", "extreme_orderings", range(1920000, 1920020), ["All seven biased"]),
        ]:
            saved = read(ROOT / "results" / source / f"{family}_evaluation.csv")
            fresh = read(folder / suite / f"{family}_evaluation.csv")
            replayed = read(folder / "old_replayed" / suite / f"{family}_evaluation.csv")
            for group in groups:
                orders = sorted({r["order"] for r in saved if (r["order"] == "mixed") == (group == "Random")})
                assert len(orders) == {"Random": 1, "Biased": 9, "All seven biased": 7}[group]
                old_rows = [r for r in saved if r["variant"] == "fixed" and r["order"] in orders]
                policy_rows = [r for r in saved if r["variant"] != "fixed" and r["order"] in orders]
                new_rows = [r for r in fresh if r["order"] in orders]
                replayed_rows = [r for r in replayed if r["order"] in orders]
                old_acc = paired_accuracy(old_rows, seeds, orders)
                replayed_acc = paired_accuracy(replayed_rows, seeds, orders)
                new_acc = paired_accuracy(new_rows, seeds, orders)
                policy_acc = paired_accuracy(policy_rows, seeds, orders, policy=True)
                aggregates.append(dict(
                    family=family, ordering=group, old_fixed_accuracy=old_acc,
                    old_replayed_accuracy=replayed_acc,
                    fixed_accuracy=new_acc, policy_accuracy=policy_acc,
                    difference_pp=policy_acc-new_acc, fixed_improvement_pp=new_acc-replayed_acc,
                    config_changed=old != new["config"],
                ))
    with (folder / "aggregates.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    (folder / "config_comparison.json").write_text(json.dumps(configurations, indent=2))
    print("family,ordering,old_fixed_saved,old_fixed_replayed,new_fixed,policy_saved")
    for row in aggregates:
        print(f'{row["family"]},{row["ordering"]},{row["old_fixed_accuracy"]:.4f},{row["old_replayed_accuracy"]:.4f},{row["fixed_accuracy"]:.4f},{row["policy_accuracy"]:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/compute_matched_baseline")
    analyze(parser.parse_args().results_dir)
