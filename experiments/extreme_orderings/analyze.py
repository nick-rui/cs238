"""Descriptive paired analysis of the frozen-optimizer stress test."""

import csv, gzip, json, math
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "results/extreme_orderings"
families = ["SGD", "Adam", "AdamW", "RMSprop"]
orders = [
    "mixed",
    "single_max",
    "single_late",
    "pair_max",
    "half_block",
    "staircase",
    "class_sorted",
    "alternating_halves",
]
names = [
    "Random",
    "Single class\n20 batches",
    "Late single\n5 to recover",
    "Two classes\n40 batches",
    "Five classes\n100 batches",
    "Class staircase\n160 batches",
    "Fully class\nsorted",
    "Alternating\nhalves",
]
rows = []
traces = []
for f in families:
    rows += list(csv.DictReader((OUT / f"{f}_evaluation.csv").open()))
    traces += list(
        csv.DictReader(
            (OUT / f"{f}_traces.csv").open()
            if (OUT / f"{f}_traces.csv").exists()
            else gzip.open(OUT / f"{f}_traces.csv.gz", "rt")
        )
    )
assert len(rows) == 1920 and len(traces) == 76800
assert (
    len(
        set(
            (r["family"], r["variant"], r["policy_seed"], r["seed"], r["order"])
            for r in rows
        )
    )
    == 1920
)
assert all(
    math.isfinite(float(r[k]))
    for r in rows
    for k in ["accuracy", "eval_loss", "path_log_loss"]
)
summary = []
seed_summary = []
pair_diffs = []
for f in families:
    for o in orders:
        rr = [r for r in rows if r["family"] == f and r["order"] == o]
        fixed = {r["seed"]: float(r["accuracy"]) for r in rr if r["variant"] == "fixed"}
        policy = [r for r in rr if r["variant"] == "policy"]
        assert len(fixed) == 20 and len(policy) == 40
        diffs = [
            100
            * (np.mean([float(r["accuracy"]) for r in policy if r["seed"] == s]) - v)
            for s, v in fixed.items()
        ]
        item = dict(
            family=f,
            order=o,
            fixed_accuracy=100 * np.mean(list(fixed.values())),
            policy_accuracy=100 * np.mean([float(r["accuracy"]) for r in policy]),
            difference_pp=np.mean(diffs),
            paired_episode_sd_pp=np.std(diffs, ddof=1),
            failures=sum(r["failed"] == "True" for r in rr),
        )
        summary.append(item)
        for seed in [338, 339]:
            sr = [r for r in policy if int(r["policy_seed"]) == seed]
            seed_summary.append(
                dict(
                    family=f,
                    order=o,
                    policy_seed=seed,
                    difference_pp=100
                    * np.mean([float(r["accuracy"]) - fixed[r["seed"]] for r in sr]),
                )
            )
        for s, v in fixed.items():
            pair_diffs.append(
                dict(
                    family=f,
                    order=o,
                    episode_seed=s,
                    difference_pp=100
                    * (
                        np.mean(
                            [float(r["accuracy"]) for r in policy if r["seed"] == s]
                        )
                        - v
                    ),
                )
            )


def savecsv(name, rr):
    with (OUT / name).open("w") as file:
        w = csv.DictWriter(file, fieldnames=list(rr[0]))
        w.writeheader()
        w.writerows(rr)


savecsv("summary.csv", summary)
savecsv("policy_seed_summary.csv", seed_summary)
savecsv("paired_differences.csv", pair_diffs)
groups = {
    "Random": ["mixed"],
    "With recovery": orders[1:6],
    "Without shuffled recovery": orders[6:],
    "All seven biased": orders[1:],
}
aggregates = []
for f in families:
    for name, oo in groups.items():
        rr = [r for r in summary if r["family"] == f and r["order"] in oo]
        aggregates.append(
            dict(
                family=f,
                group=name,
                orderings=len(rr),
                **{
                    k: float(np.mean([r[k] for r in rr]))
                    for k in ["fixed_accuracy", "policy_accuracy", "difference_pp"]
                },
            )
        )
savecsv("aggregates.csv", aggregates)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": "#252A2D",
        "axes.labelcolor": "#252A2D",
        "axes.edgecolor": "#A6ADB3",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def save(fig, name):
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


mat = np.array(
    [
        [
            next(
                r["difference_pp"]
                for r in summary
                if r["family"] == f and r["order"] == o
            )
            for o in orders
        ]
        for f in families
    ]
)
lim = max(1, math.ceil(np.abs(mat).max()))
fig, ax = plt.subplots(figsize=(11, 4.5))
fig.subplots_adjust(left=0.12, right=0.88, bottom=0.29, top=0.76)
im = ax.imshow(mat, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
for i in range(4):
    for j in range(8):
        ax.text(
            j,
            i,
            f"{mat[i,j]:+.2f}",
            ha="center",
            va="center",
            fontsize=10,
            color="white" if abs(mat[i, j]) > lim * 0.6 else "#252A2D",
        )
ax.set(xticks=range(8), xticklabels=names, yticks=range(4), yticklabels=families)
ax.tick_params(length=0, labelsize=9)
ax.axvline(5.5, color="white", lw=4)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
cb.set_label("Policy minus fixed (percentage points)")
fig.suptitle(
    "Policy-Controlled Optimizers under stronger ordering shifts",
    x=0.025,
    ha="left",
    y=0.96,
    fontsize=16,
)
fig.text(
    0.025,
    0.86,
    "Identical balanced example bags and training budgets. Policies and tuned fixed baselines are frozen.",
    fontsize=10,
    color="#5E656B",
)
fig.text(
    0.025,
    0.09,
    "Final MNIST test accuracy. 20 paired episode seeds; both policy-training seeds included. Last two columns have no shuffled recovery phase.",
    fontsize=9,
    color="#5E656B",
)
fig.text(
    0.025,
    0.045,
    "Blue favors policy control; red favors fixed settings. Exploratory stress tests, not significance tests or optimized adversarial attacks.",
    fontsize=9,
    color="#5E656B",
)
save(fig, "stress_comparison")
# Show every optimizer on two predefined stringent conditions, avoiding winner selection.
fig, axs = plt.subplots(2, 4, figsize=(11, 5.5))
fig.subplots_adjust(
    left=0.07, right=0.985, bottom=0.16, top=0.81, hspace=0.35, wspace=0.23
)
for row, order in enumerate(["single_late", "class_sorted"]):
    for col, f in enumerate(families):
        ax = axs[row, col]
        if order == "single_late":
            ax.axvspan(175, 195, color="#A6ADB3", alpha=0.23, lw=0)
        else:
            for x in range(20, 200, 20):
                ax.axvline(x, color="#DDE1E5", lw=0.5)
        for variant, color, ls, label in [
            ("fixed", "#7A838B", "--", "Tuned fixed"),
            ("policy", "#466F9D", "-", "Policy-controlled"),
        ]:
            q = []
            for step in range(5, 201, 5):
                vals = [
                    100 * float(r["eval_probe_accuracy"])
                    for r in traces
                    if r["family"] == f
                    and r["order"] == order
                    and r["variant"] == variant
                    and int(r["step"]) == step
                ]
                q.append(np.quantile(vals, [0.25, 0.5, 0.75]))
            q = np.array(q)
            xx = np.arange(5, 201, 5)
            ax.fill_between(xx, q[:, 0], q[:, 2], color=color, alpha=0.12, lw=0)
            ax.plot(xx, q[:, 1], color=color, ls=ls, lw=1.5, label=label)
        ax.set(xlim=(0, 200), ylim=(0, 100), xticks=[0, 100, 200], yticks=[0, 50, 100])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E7EAED", lw=0.45)
        if row == 0:
            ax.set_title(f, fontsize=11)
        if col == 0:
            ax.set_ylabel(
                ("Late single class" if row == 0 else "Fully class sorted")
                + "\nProbe accuracy (%)",
                fontsize=9,
            )
        if row == 1:
            ax.set_xlabel("Classifier updates", fontsize=9)
fig.suptitle(
    "Damage and recovery on two harder streams", x=0.025, ha="left", y=0.97, fontsize=16
)
fig.legend(
    *axs[0, 0].get_legend_handles_labels(),
    loc="upper left",
    bbox_to_anchor=(0.025, 0.91),
    ncol=2,
    frameon=False,
    fontsize=9,
)
fig.text(
    0.025,
    0.04,
    "Medians and middle 50% across episode/policy seeds (not confidence intervals). First 1,000 test images; no policy feedback.",
    fontsize=9,
    color="#5E656B",
)
save(fig, "stress_trajectories")
checks = json.loads((OUT / "checks.json").read_text())
checks.update(
    evaluation_rows=len(rows),
    trace_rows=len(traces),
    evaluation_failures=sum(r["failed"] == "True" for r in rows),
    finite_metrics=True,
    unique_episode_keys=True,
)
(OUT / "final_checks.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(dict(aggregates=aggregates, checks=checks), indent=2))
