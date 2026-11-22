"""Summarize the original ordering suite for fixed and policy-controlled optimizers."""

import csv, os
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATASET = os.environ.get("CS238_DATASET", "mnist")
if DATASET not in ["mnist", "fashion_mnist", "cifar10"]:
    raise ValueError(f"Unknown CS238_DATASET: {DATASET!r}")
RESULTS = ROOT / "results"
if DATASET == "fashion_mnist":
    RESULTS = RESULTS / "fashion_mnist"
elif DATASET == "cifar10":
    RESULTS = RESULTS / "cifar10"
OUT = RESULTS / "optimizer_families"
FAMILIES = ["SGD", "Adam", "AdamW", "RMSprop"]
ORDERS = [
    "mixed",
    "single_early",
    "single_middle",
    "single_late",
    "pair_long",
    "two_shocks",
    "repeat_same",
    "three_shocks",
    "ramp",
    "rapid",
]
rows = []
for family in FAMILIES:
    with (OUT / f"{family}_evaluation.csv").open() as f:
        rows += list(csv.DictReader(f))
summary = []
for family in FAMILIES:
    for order in ORDERS:
        rr = [r for r in rows if r["family"] == family and r["order"] == order]
        fixed = {r["seed"]: float(r["accuracy"]) for r in rr if r["variant"] == "fixed"}
        policy = [r for r in rr if r["variant"] == "gradient"]
        assert len(fixed) == 20 and len(policy) == 40
        summary.append(
            dict(
                family=family,
                order=order,
                fixed_accuracy=100 * np.mean(list(fixed.values())),
                policy_accuracy=100 * np.mean([float(r["accuracy"]) for r in policy]),
                difference_pp=np.mean(
                    [
                        100
                        * (
                            np.mean(
                                [
                                    float(r["accuracy"])
                                    for r in policy
                                    if r["seed"] == seed
                                ]
                            )
                            - acc
                        )
                        for seed, acc in fixed.items()
                    ]
                ),
            )
        )
with (OUT / "summary.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)
aggregates = []
for family in FAMILIES:
    for label, orders in [("Random", ["mixed"]), ("Biased", ORDERS[1:])]:
        rr = [r for r in summary if r["family"] == family and r["order"] in orders]
        aggregates.append(
            dict(
                family=family,
                ordering=label,
                **{
                    k: float(np.mean([r[k] for r in rr]))
                    for k in ["fixed_accuracy", "policy_accuracy", "difference_pp"]
                },
            )
        )
with (OUT / "aggregates.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(aggregates[0]))
    w.writeheader()
    w.writerows(aggregates)
fig, ax = plt.subplots(figsize=(10, 3.6))
fig.subplots_adjust(left=0.12, right=0.9, bottom=0.42, top=0.85)
mat = np.array(
    [
        [
            next(
                r["difference_pp"]
                for r in summary
                if r["family"] == f and r["order"] == o
            )
            for o in ORDERS
        ]
        for f in FAMILIES
    ]
)
lim = max(1, float(np.ceil(abs(mat).max())))
im = ax.imshow(mat, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
for i in range(4):
    for j in range(10):
        ax.text(
            j,
            i,
            f"{mat[i,j]:+.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(mat[i, j]) > lim * 0.6 else "#252A2D",
        )
ax.set(
    xticks=range(10),
    xticklabels=[o.replace("_", " ") for o in ORDERS],
    yticks=range(4),
    yticklabels=FAMILIES,
)
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
ax.tick_params(length=0)
ax.spines[:].set_visible(False)
fig.colorbar(
    im, ax=ax, fraction=0.025, pad=0.02, label="Policy minus fixed (percentage points)"
)
fig.suptitle(
    "Policy-Controlled Optimizers: original ordering suite",
    x=0.025,
    ha="left",
    fontsize=14,
)
fig.text(
    0.025,
    0.03,
    "20 paired episode seeds; two policy-training seeds. Final "
    + {"fashion_mnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}.get(DATASET, "MNIST")
    + " accuracy. Descriptive means, not significance tests.",
    fontsize=9,
)
for ext in ["png", "svg"]:
    fig.savefig(OUT / f"comparison.{ext}", dpi=300, facecolor="white")
for r in aggregates:
    print(r)
