"""Compare ablations with committed regular-suite full and fixed evaluations."""

import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/ablations"
METRICS = ["eval_loss", "accuracy", "path_log_loss"]


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


summary = []
for family in ["SGD", "Adam"]:
    original = read(ROOT / f"results/optimizer_families/{family}_evaluation.csv")
    ablations = read(OUT / f"{family}_evaluation.csv")
    full = {(r["seed"], r["order"], r["policy_seed"]): r for r in original if r["variant"] == "gradient"}
    for variant in ["fixed", "gradient", "lr_only", "fixed_memory", "no_feedback"]:
        rows = [r for r in original + ablations if r["variant"] == variant]
        assert len(rows) == (200 if variant == "fixed" else 400)
        pairs = []
        for r in rows:
            refs = [full[(r["seed"], r["order"], str(s))] for s in [338, 339]] if variant == "fixed" else [full[(r["seed"], r["order"], r["policy_seed"])]]
            pairs.append({metric: float(r[metric]) - np.mean([float(ref[metric]) for ref in refs]) for metric in METRICS})
        summary.append(dict(family=family, variant=variant,
                            **{metric: float(np.mean([float(r[metric]) for r in rows])) for metric in METRICS},
                            failures=sum(r["failed"] == "True" for r in rows), episodes=len(rows),
                            **{f"difference_{metric}": float(np.mean([p[metric] for p in pairs])) for metric in METRICS}))
with (OUT / "summary.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
lines = ["| Family | Variant | Loss | Accuracy (%) | Path log loss | Failures | Δ loss | Δ accuracy (pp) | Δ path log loss |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for r in summary:
    lines.append(f"| {r['family']} | {r['variant']} | {r['eval_loss']:.4f} | {100*r['accuracy']:.2f} | {r['path_log_loss']:.4f} | {r['failures']}/{r['episodes']} | {r['difference_eval_loss']:+.4f} | {100*r['difference_accuracy']:+.2f} | {r['difference_path_log_loss']:+.4f} |")
table = "\n".join(lines)
(OUT / "summary.md").write_text(table + "\n\nDescriptive means over all ten original orderings and twenty episode seeds; two policy seeds. Differences are variant minus full, paired by episode seed, ordering, and policy seed; fixed uses the mean of both full policies within each episode/ordering. No significance claim.\n")
print(table)
