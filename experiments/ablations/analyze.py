"""Compare ablations with committed regular-suite full and fixed evaluations."""

import argparse
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/ablations"
START_OUT = ROOT / "results/ablations_start_values"
METRICS = ["eval_loss", "accuracy", "path_log_loss"]


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def episode_key(row):
    return row["seed"], row["order"], row["policy_seed"]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--held-source", choices=["baseline", "start"], default="baseline")
start_held = parser.parse_args().held_source == "start"
output = START_OUT if start_held else OUT
summary = []
for family in ["SGD", "Adam"]:
    original = read(ROOT / f"results/optimizer_families/{family}_evaluation.csv")
    ablations = read(output / f"{family}_evaluation.csv")
    full = {episode_key(r): r for r in original if r["variant"] == "gradient"}
    if start_held:
        tuned = read(OUT / f"{family}_evaluation.csv")
    variants = ["gradient", "lr_only", "fixed_memory"] if start_held else [
        "fixed", "gradient", "lr_only", "fixed_memory", "no_feedback"
    ]
    for variant in variants:
        rows = [r for r in original + ablations if r["variant"] == variant]
        assert len(rows) == (200 if variant == "fixed" else 400)
        if start_held:
            assert len(full) == len(rows) and {episode_key(r) for r in rows} == full.keys()
        pairs = []
        for r in rows:
            refs = [full[(r["seed"], r["order"], str(s))] for s in [338, 339]] if variant == "fixed" else [full[episode_key(r)]]
            pairs.append({metric: float(r[metric]) - np.mean([float(ref[metric]) for ref in refs]) for metric in METRICS})
        record = dict(family=family, variant=variant,
                      **{metric: float(np.mean([float(r[metric]) for r in rows])) for metric in METRICS},
                      failures=sum(r["failed"] == "True" for r in rows), episodes=len(rows),
                      **{f"difference_{metric}": float(np.mean([p[metric] for p in pairs])) for metric in METRICS})
        if start_held:
            tuned_rows = [r for r in tuned if r["variant"] == variant]
            tuned_pairs = {episode_key(r): r for r in tuned_rows}
            if variant != "gradient":
                assert len(tuned_rows) == len(rows) and tuned_pairs.keys() == full.keys()
            for metric in METRICS:
                record[f"difference_tuned_{metric}"] = (
                    float(np.mean([float(r[metric]) - float(tuned_pairs[episode_key(r)][metric]) for r in rows]))
                    if variant != "gradient" else None
                )
        summary.append(record)
with (output / "summary.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
lines = ["| Family | Variant | Loss | Accuracy (%) | Path log loss | Failures | Δ loss | Δ accuracy (pp) | Δ path log loss |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
if start_held:
    lines[0] += " Δ loss vs tuned-held | Δ accuracy vs tuned-held (pp) | Δ path log loss vs tuned-held |"
    lines[1] += "---:|---:|---:|"
for r in summary:
    line = f"| {r['family']} | {r['variant']} | {r['eval_loss']:.4f} | {100*r['accuracy']:.2f} | {r['path_log_loss']:.4f} | {r['failures']}/{r['episodes']} | {r['difference_eval_loss']:+.4f} | {100*r['difference_accuracy']:+.2f} | {r['difference_path_log_loss']:+.4f} |"
    if start_held:
        line += (
            f" {r['difference_tuned_eval_loss']:+.4f} | {100*r['difference_tuned_accuracy']:+.2f} | {r['difference_tuned_path_log_loss']:+.4f} |"
            if r["variant"] != "gradient" else " — | — | — |"
        )
    lines.append(line)
table = "\n".join(lines)
notes = "Descriptive means over all ten original orderings and twenty episode seeds; two policy seeds. "
if start_held:
    notes += "Differences are start-held minus full (gradient) or start-held minus the same tuned-held ablation, paired by episode seed, ordering, and policy seed. "
else:
    notes += "Differences are variant minus full, paired by episode seed, ordering, and policy seed; fixed uses the mean of both full policies within each episode/ordering. "
(output / "summary.md").write_text(table + "\n\n" + notes + "No significance claim.\n")
print(table)
