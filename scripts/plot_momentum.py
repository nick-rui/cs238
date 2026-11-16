"""Reproduce the Policy-Controlled SGD figure from saved evaluation traces (variant=gradient; no smoothing)."""

import csv, gzip
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)
source = ROOT / "results/optimizer_families/SGD_traces.csv"
rows = list(
    csv.DictReader(
        source.open() if source.exists() else gzip.open(str(source) + ".gz", "rt")
    )
)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": "#252A2D",
        "axes.labelcolor": "#252A2D",
        "axes.edgecolor": "#A6ADB3",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
fig, ax = plt.subplots(figsize=(6, 1.38))
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.28, top=0.88)
ax.axvspan(80, 96, color="#A6ADB3", alpha=0.23, lw=0)
records = []
for order, color, ls, label in [
    ("mixed", "#7A838B", "--", "Random ordering"),
    ("single_middle", "#466F9D", "-", "Mixed / biased / mixed"),
]:
    med = []
    lo = []
    hi = []
    for end in range(5, 201, 5):
        rr = [
            r
            for r in rows
            if r["variant"] == "gradient"
            and r["order"] == order
            and int(r["step"]) == end
        ]
        assert len(rr) == 40 and len(set(r["policy_seed"] for r in rr)) == 2
        vals = np.array([float(r["momentum"]) for r in rr])
        q = np.quantile(vals, [0.25, 0.5, 0.75])
        lo.append(q[0])
        med.append(q[1])
        hi.append(q[2])
        records.append(
            dict(
                order=order,
                interval_start=end - 5,
                interval_end=end,
                n=len(rr),
                q25=q[0],
                median=q[1],
                q75=q[2],
            )
        )
    # Trace records describe the controls USED over the five updates ending at `step`.
    xx = np.arange(0, 201, 5)
    ax.fill_between(
        xx,
        np.r_[lo, lo[-1]],
        np.r_[hi, hi[-1]],
        step="post",
        color=color,
        alpha=0.14,
        lw=0,
    )
    ax.step(
        xx, np.r_[med, med[-1]], where="post", color=color, ls=ls, lw=1.5, label=label
    )
ax.set(
    xlim=(0, 200),
    ylim=(0, 1),
    xticks=[0, 40, 80, 120, 160, 200],
    yticks=[0, 0.5, 1],
    xlabel="Classifier updates",
    ylabel="Momentum",
)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E7EAED", lw=0.45)
ax.set_axisbelow(True)
ax.text(88, 0.96, "Biased block", ha="center", va="top", fontsize=8)
ax.legend(
    loc="upper right", frameon=False, fontsize=7.5, handlelength=2.3, borderaxespad=0.1
)
for ext in ["pdf", "svg", "png"]:
    fig.savefig(OUT / f"momentum_block.{ext}", dpi=300, facecolor="white")
with (OUT / "momentum_summary.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
