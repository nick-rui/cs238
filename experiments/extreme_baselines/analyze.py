"""Summarize the extreme-ordering baselines (reads evaluation CSVs only).

Writes results/extreme_baselines/{summary,per_order,comparisons,drift}.csv,
verdicts.json and tables.md. Differences are accuracy(A) - accuracy(B) in pp
with scripts/paired_bootstrap.py's paired hierarchical bootstrap (10,000
resamples, NumPy seed 238); 90% CIs are percentiles of the same resamples.
Decision rules are the ones prespecified in results/extreme_baselines/README.md.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/prospective_schedules"))
sys.path.insert(0, str(ROOT / "scripts"))
import common as c  # noqa: E402  (no suite setup needed: CSVs only)
import paired_bootstrap as pb  # noqa: E402

OUT = ROOT / "results/extreme_baselines"
FAMILIES = c.FAMILIES
ORDERS = ["mixed", "single_max", "single_late", "pair_max", "half_block", "staircase",
          "class_sorted", "alternating_halves"]
BIASED = ORDERS[1:]
KINDS = ["cosine", "step", "warmup_cosine"]
MARGIN = 1.0
NEW = {  # label -> (method, seeds)
    "cosine (extreme-tuned)": ("cosine_extreme", [-1]),
    "step (extreme-tuned)": ("step_extreme", [-1]),
    "warmup-cosine (extreme-tuned)": ("warmup_cosine_extreme", [-1]),
    "fixed, extreme-tuned (batched engine)": ("fixed_engine", [-1]),
    "random schedule, cold (5 seeds)": ("random_schedule_cold", [0, 1, 2, 3, 4]),
    "random schedule, warm (5 seeds)": ("random_schedule_warm", [0, 1, 2, 3, 4]),
    "untrained policy, cold (5 seeds)": ("untrained_cold", c.POLICY_SEEDS),
    "untrained policy, warm (5 seeds)": ("untrained_warm", c.POLICY_SEEDS),
    "random fixed config (5 draws)": ("random_fixed", [0, 1, 2, 3, 4]),
    "frozen regular policy (5 seeds, fresh)": ("frozen_regular", c.POLICY_SEEDS),
}
PR15 = {
    "fixed, extreme-tuned (PR #15)": ("fixed_retuned", [-1]),
    "cold extreme policy (5 seeds)": ("policy_retrained", c.POLICY_SEEDS),
    "warm extreme policy (5 seeds)": ("warm_extreme", c.POLICY_SEEDS),
}
COS = "cosine (extreme-tuned)"
COMPARISONS = [  # (id, A, B, kind)
    ("P1", "warm extreme policy (5 seeds)", COS, "superiority"),
    ("P2", "cold extreme policy (5 seeds)", COS, "superiority"),
    ("E1", COS, "random schedule, cold (5 seeds)", "equivalence"),
    ("E2", COS, "random schedule, warm (5 seeds)", "equivalence"),
    ("E3", COS, "untrained policy, cold (5 seeds)", "equivalence"),
    ("E4", COS, "untrained policy, warm (5 seeds)", "equivalence"),
    ("E5", COS, "random fixed config (5 draws)", "equivalence"),
    ("F1", COS, "fixed, extreme-tuned (PR #15)", "difference"),
    ("F1e", COS, "fixed, extreme-tuned (batched engine)", "difference"),
    # secondary
    ("S1", "warm extreme policy (5 seeds)", "cosine (regular-tuned, PR #14)", "superiority"),
    ("S2", "cold extreme policy (5 seeds)", "cosine (regular-tuned, PR #14)", "superiority"),
    ("S3", "cosine (regular-tuned, PR #14)", "random schedule, cold (5 seeds)", "equivalence"),
    ("S4", COS, "cosine (regular-tuned, PR #14)", "difference"),
    ("S5", "warm extreme policy (5 seeds)", "dev-best shape (extreme-tuned)", "superiority"),
    ("S6", "cold extreme policy (5 seeds)", "dev-best shape (extreme-tuned)", "superiority"),
    ("S7", "frozen regular policy (5 seeds, fresh)", COS, "superiority"),
    ("S8", "warm extreme policy (5 seeds)", "random schedule, cold (5 seeds)", "superiority"),
    ("S9", "cold extreme policy (5 seeds)", "random schedule, cold (5 seeds)", "superiority"),
    ("S10", "warm extreme policy (5 seeds)", "untrained policy, warm (5 seeds)", "superiority"),
    ("S11", "cold extreme policy (5 seeds)", "untrained policy, cold (5 seeds)", "superiority"),
    ("S12", "fixed, extreme-tuned (batched engine)", "fixed, extreme-tuned (PR #15)", "difference"),
]


SMOKE = {"on": False}


def load(path):
    rows = c.readcsv(path)
    if SMOKE["on"]:  # smoke outputs cover test seeds 1920000-1920003 only
        rows = [r for r in rows if int(r["seed"]) < 1920004]
    c.check_finite(rows)
    return rows


def methods(family, out):
    m = {}
    for label, (method, seeds) in NEW.items():
        rows = []
        for s in seeds:
            rows += load(out / "evaluation" / f"{family}_{method}_{s}.csv")
        m[label] = rows
    for label, (variant, seeds) in PR15.items():
        rows = []
        for s in seeds:
            rows += load(ROOT / f"results/extreme_warm_start/evaluation/{family}_{variant}_{s}.csv")
        m[label] = rows
    m["cosine (regular-tuned, PR #14)"] = load(ROOT / f"results/prospective_schedules/extreme/{family}_cosine/evaluation.csv")
    best = json.loads((out / "selected" / f"{family}.json").read_text())["overall"]
    m["dev-best shape (extreme-tuned)"] = m[{"cosine": COS, "step": "step (extreme-tuned)",
                                             "warmup_cosine": "warmup-cosine (extreme-tuned)"}[best]]
    for label, rows in m.items():
        keys = {(r["policy_seed"], r["order"], r["seed"]) for r in rows}
        n_seeds = len({r["policy_seed"] for r in rows})
        if len(keys) != len(rows) or len(rows) != n_seeds * (32 if SMOKE["on"] else 160) or {r["order"] for r in rows} != set(ORDERS):
            raise ValueError(f"{family} {label}: incomplete or duplicated episodes")
    return m, best


def seed_scores(rows, orders):
    out = {}
    for s in sorted({int(r["policy_seed"]) for r in rows}):
        rr = [r for r in rows if int(r["policy_seed"]) == s]
        out[s] = float(np.mean([np.mean([float(r["accuracy"]) * 100 for r in rr if r["order"] == o]) for o in orders]))
    return out


def describe(rows):
    d = {}
    for scope, orders in (("biased", BIASED), ("all", ORDERS)):
        per = list(seed_scores(rows, orders).values())
        d[scope] = float(np.mean(per))
        d[f"{scope}_sd"] = float(np.std(per, ddof=1)) if len(per) > 1 else None
    for o in ORDERS:
        d[o] = float(np.mean([float(r["accuracy"]) * 100 for r in rows if r["order"] == o]))
    d["failures"] = sum(str(r["failed"]) == "True" for r in rows)
    d["episodes"] = len(rows)
    d["seeds"] = len({r["policy_seed"] for r in rows})
    return d


def compare(a, b):
    mats = pb.paired(a, b)
    if set(mats) != set(ORDERS):
        raise ValueError("Comparison does not cover every ordering")
    groups = {"biased": {k: v for k, v in mats.items() if k != "mixed"}, "all": mats}
    groups.update({o: {o: mats[o]} for o in ORDERS})
    out = {}
    for name, group in groups.items():
        res, samples = pb.bootstrap(group, return_samples=True)
        lo, hi = np.percentile(samples, [5, 95])
        res.update(ci90_low_pp=float(lo), ci90_high_pp=float(hi))
        out[name] = res
    return out


def decide(kind, x):
    if kind == "superiority":
        return "holds" if x["ci_low_pp"] > 0 else "fails" if x["ci_high_pp"] < 0 else "inconclusive"
    if kind == "equivalence":
        if -MARGIN < x["ci90_low_pp"] and x["ci90_high_pp"] < MARGIN:
            return "holds (equivalent)"
        if x["ci_low_pp"] > MARGIN:
            return "fails (A better by >1 pp)"
        if x["ci_high_pp"] < -MARGIN:
            return "fails (B better by >1 pp)"
        return "inconclusive"
    return ""


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def ci(x, level=95):
    lo, hi = ("ci_low_pp", "ci_high_pp") if level == 95 else ("ci90_low_pp", "ci90_high_pp")
    return f'{x["mean_diff_pp"]:+.2f} [{x[lo]:+.2f}, {x[hi]:+.2f}]'


def main(out=OUT):
    summary, per_order, comps, verdicts, drift = [], [], [], {}, []
    md = ["# Extreme baselines: tables", "",
          "Generated by `experiments/extreme_baselines/analyze.py`. Accuracy in %; ± is the SD over policy/random seeds;",
          "biased = mean of the seven non-mixed orderings; all = all eight. Differences A − B in pp, paired hierarchical",
          "bootstrap 95% CI [and 90% CI]; wins = seeds with positive biased-mean difference.", ""]
    all_m = {}
    for family in FAMILIES:
        m, best = methods(family, out)
        all_m[family] = m
        for label, rows in m.items():
            d = describe(rows)
            summary.append(dict(family=family, method=label, **{k: d[k] for k in
                           ("biased", "biased_sd", "all", "all_sd", "failures", "episodes", "seeds")}))
            per_order.append(dict(family=family, method=label, **{o: d[o] for o in ORDERS}))
        verdicts[family] = dict(dev_best_shape=best)
        for cid, a, b, kind in COMPARISONS:
            res = compare(m[a], m[b])
            for scope, x in res.items():
                row = dict(id=cid, family=family, a=a, b=b, scope=scope, test=kind, **x)
                if scope == "biased":
                    row["decision"] = decide(kind, x)
                    if cid[0] in "PE":
                        verdicts[family][cid] = dict(a=a, b=b, decision=row["decision"], mean=x["mean_diff_pp"],
                                                     ci95=[x["ci_low_pp"], x["ci_high_pp"]],
                                                     ci90=[x["ci90_low_pp"], x["ci90_high_pp"]],
                                                     seeds_beating=x["seeds_beating"], n_seeds=x["n_policy_seeds"])
                comps.append(row)
        # drift: fresh frozen regular vs saved rows used elsewhere
        _, saved = c.saved_rows("extreme", family)
        sidx = {(int(r["policy_seed"]), r["order"], int(r["seed"])): float(r["accuracy"]) for r in saved}
        fresh = m["frozen regular policy (5 seeds, fresh)"]
        diffs = [abs(float(r["accuracy"]) - sidx[int(r["policy_seed"]), r["order"], int(r["seed"])]) for r in fresh]
        drift.append(dict(family=family, method="frozen_regular vs saved extreme rows", episodes=len(diffs),
                          exact=sum(d == 0 for d in diffs), max_abs_accuracy_pp=100 * max(diffs),
                          mean_abs_accuracy_pp=100 * float(np.mean(diffs))))
        untrained = {}
        for label in ("untrained policy, cold (5 seeds)", "untrained policy, warm (5 seeds)"):
            rows = m[label]
            by = {}
            for r in rows:
                by.setdefault((r["order"], r["seed"]), set()).add(r["accuracy"])
            untrained[label] = all(len(v) == 1 for v in by.values())
        verdicts[family]["untrained_identical_across_seeds"] = untrained

    import csv

    def write(name, rows):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (out / name).open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write("summary.csv", summary)
    write("per_order.csv", per_order)
    write("comparisons.csv", comps)
    write("drift.csv", drift)
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=2) + "\n")

    # ---------------- markdown
    labels = list(all_m["SGD"])
    labels.remove("dev-best shape (extreme-tuned)")
    for scope in ("biased", "all"):
        md += [f"## Method means, {scope} ({'seven' if scope == 'biased' else 'all eight'} orderings)", "",
               "| Method | " + " | ".join(FAMILIES) + " |", "|---|" + "---:|" * len(FAMILIES)]
        for label in labels:
            cells = []
            for fam in FAMILIES:
                s = next(x for x in summary if x["family"] == fam and x["method"] == label)
                cell = f2(s[scope]) + (f" ± {s[scope + '_sd']:.2f}" if s[scope + "_sd"] is not None else "")
                if s["failures"]:
                    cell += f" ({s['failures']} failed)"
                cells.append(cell)
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["Dev-best shape (extreme-tuned): " + ", ".join(f"{f}: {verdicts[f]['dev_best_shape']}" for f in FAMILIES), ""]
    md += ["Failure counts are over all eight orderings and all seeds of a method.", ""]
    for scope in ("biased", "all"):
        md += [f"## Comparisons, {scope} mean", "",
               "| ID | A − B | " + " | ".join(FAMILIES) + " |", "|---|---|" + "---:|" * len(FAMILIES)]
        for cid, a, b, kind in COMPARISONS:
            cells = []
            for fam in FAMILIES:
                x = next(r for r in comps if r["id"] == cid and r["family"] == fam and r["scope"] == scope)
                cell = ci(x)
                if kind == "equivalence" or cid.startswith("F"):
                    cell += f"; 90% [{x['ci90_low_pp']:+.2f}, {x['ci90_high_pp']:+.2f}]"
                if x["n_policy_seeds"] > 1:
                    cell += f"; {x['seeds_beating']}/{x['n_policy_seeds']}"
                if scope == "biased" and x.get("decision"):
                    cell += f"; **{x['decision']}**"
                cells.append(cell)
            md.append(f"| {cid} | {a} − {b} | " + " | ".join(cells) + " |")
        md.append("")
    for fam in FAMILIES:
        md += [f"## Per ordering: {fam}", "", "| Method | " + " | ".join(ORDERS) + " |", "|---|" + "---:|" * len(ORDERS)]
        for label in labels:
            p = next(x for x in per_order if x["family"] == fam and x["method"] == label)
            md.append(f"| {label} | " + " | ".join(f"{p[o]:.2f}" for o in ORDERS) + " |")
        md += ["", f"Per-ordering differences ({fam}), pp [95% CI]:", "",
               "| ID | " + " | ".join(ORDERS) + " |", "|---|" + "---:|" * len(ORDERS)]
        for cid, a, b, kind in COMPARISONS[:9]:
            cells = []
            for o in ORDERS:
                x = next(r for r in comps if r["id"] == cid and r["family"] == fam and r["scope"] == o)
                cells.append(ci(x))
            md.append(f"| {cid} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Replay drift", "", "| Family | Check | Episodes | Exact | Max |Δ| (pp) | Mean |Δ| (pp) |", "|---|---|---:|---:|---:|---:|"]
    for d in drift:
        md.append(f"| {d['family']} | {d['method']} | {d['episodes']} | {d['exact']} | {d['max_abs_accuracy_pp']:.3f} | {d['mean_abs_accuracy_pp']:.4f} |")
    (out / "tables.md").write_text("\n".join(md) + "\n")
    for fam in FAMILIES:
        print(fam, json.dumps({k: v["decision"] if isinstance(v, dict) and "decision" in v else v
                               for k, v in verdicts[fam].items()}))


if __name__ == "__main__":
    if len(sys.argv) > 1:  # smoke directory
        SMOKE["on"] = True
        main(Path(sys.argv[1]))
    else:
        main()
