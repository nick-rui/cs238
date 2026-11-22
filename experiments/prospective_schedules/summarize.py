"""Summarize prospective schedules and the reactive rule against saved comparators.

Reads only evaluation CSVs (no training or evaluation). Writes
results/prospective_schedules/{summary,comparisons}.csv and
results/reactive_rule/{summary,comparisons}.csv, plus markdown tables
(tables.md) used in each README. Differences are accuracy(A) - accuracy(B) in
percentage points with paired hierarchical bootstrap 95% CIs
(scripts/paired_bootstrap.py: 10,000 resamples, NumPy seed 238).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

ROOT = c.ROOT
PS = ROOT / "results/prospective_schedules"
RR = ROOT / "results/reactive_rule"
KINDS = ["cosine", "step", "warmup_cosine"]
ORDERS = {"regular": None, "extreme": None}


def orders(suite, rows):
    found = list(dict.fromkeys(r["order"] for r in rows))
    return found


def load(path):
    rows = c.readcsv(path)
    c.check_finite(rows)
    return rows


def method_rows(suite, family):
    fixed, policy = c.saved_rows(suite, family)
    out = {"fixed (saved)": fixed, "policy (saved, 5 seeds)": policy}
    if (PS / suite / f"{family}_fixed_engine/done.json").exists():
        out["fixed (same engine)"] = load(PS / suite / f"{family}_fixed_engine/evaluation.csv")
    dm = []
    for s in c.POLICY_SEEDS:
        path = PS / suite / f"{family}_dev_mean_{s}/evaluation.csv"
        if path.exists() and (path.parent / "done.json").exists():
            dm += load(path)
    if len({r["policy_seed"] for r in dm}) == 5:
        out["dev-mean schedule (5 seeds)"] = dm
    sel = PS / "selected" / f"{family}.json"
    if sel.exists():
        overall = json.loads(sel.read_text())["overall"]
        for k in KINDS:
            path = PS / suite / f"{family}_{k}/evaluation.csv"
            if (path.parent / "done.json").exists():
                out[f"{k} schedule"] = load(path)
                if k == overall:
                    out["tuned schedule (dev-best kind)"] = out[f"{k} schedule"]
    if (RR / suite / family / "done.json").exists():
        out["reactive rule"] = load(RR / suite / family / "evaluation.csv")
    if (RR / "secondary_firing" / suite / family / "done.json").exists():
        out["firing rule (secondary)"] = load(RR / "secondary_firing" / suite / family / "evaluation.csv")
    wf, wp = c.warm_rows(suite, family)
    out["warm-start policy (5 seeds, fresh)"] = wp
    out["fixed (warm-start fresh replay)"] = wf
    return out


PS_PAIRS = [
    ("policy (saved, 5 seeds)", "dev-mean schedule (5 seeds)"),
    ("dev-mean schedule (5 seeds)", "fixed (saved)"),
    ("policy (saved, 5 seeds)", "tuned schedule (dev-best kind)"),
    ("tuned schedule (dev-best kind)", "fixed (saved)"),
    ("tuned schedule (dev-best kind)", "fixed (same engine)"),
    ("policy (saved, 5 seeds)", "cosine schedule"),
    ("policy (saved, 5 seeds)", "step schedule"),
    ("policy (saved, 5 seeds)", "warmup_cosine schedule"),
    ("cosine schedule", "fixed (saved)"),
    ("step schedule", "fixed (saved)"),
    ("warmup_cosine schedule", "fixed (saved)"),
    ("fixed (same engine)", "fixed (saved)"),
]
RR_PAIRS = [
    ("reactive rule", "fixed (saved)"),
    ("reactive rule", "fixed (same engine)"),
    ("policy (saved, 5 seeds)", "reactive rule"),
    ("warm-start policy (5 seeds, fresh)", "reactive rule"),
    ("reactive rule", "tuned schedule (dev-best kind)"),
    ("firing rule (secondary)", "fixed (same engine)"),
    ("policy (saved, 5 seeds)", "firing rule (secondary)"),
    ("fixed (same engine)", "fixed (saved)"),
]
PS_METHODS = ["fixed (saved)", "fixed (same engine)", "policy (saved, 5 seeds)", "dev-mean schedule (5 seeds)",
              "cosine schedule", "step schedule", "warmup_cosine schedule", "tuned schedule (dev-best kind)"]
RR_METHODS = ["fixed (saved)", "fixed (same engine)", "reactive rule", "firing rule (secondary)", "policy (saved, 5 seeds)",
              "warm-start policy (5 seeds, fresh)", "fixed (warm-start fresh replay)"]


def build(pairs, methods, suites):
    summary, comps = [], []
    for suite in suites:
        for family in c.FAMILIES:
            rows = method_rows(suite, family)
            oo = orders(suite, rows["fixed (saved)"])
            for m in methods:
                if m in rows:
                    summary.append(dict(suite=suite, family=family, method=m, **c.condition_means(rows[m], oo)))
            for a, bb in pairs:
                if a in rows and bb in rows:
                    res = c.compare(rows[a], rows[bb], oo)
                    for group, r in res.items():
                        comps.append(dict(suite=suite, family=family, a=a, b=bb, group=group, **r))
    return summary, comps


def md(summary, comps, pairs, methods, suites):
    lines = []
    for suite in suites:
        lines += [f"### {suite.capitalize()} suite: mean final accuracy (%)", "",
                  "Biased = equal-weight mean over non-shuffled conditions; seeded rows give policy-seed SD of the biased mean.", "",
                  "| Method | " + " | ".join(f"{f} biased" for f in c.FAMILIES) + " | " + " | ".join(f"{f} shuffled" for f in c.FAMILIES) + " |",
                  "|---|" + "---:|" * 8]
        for m in methods:
            cells, mix = [], []
            for fam in c.FAMILIES:
                r = next((x for x in summary if x["suite"] == suite and x["family"] == fam and x["method"] == m), None)
                if r is None:
                    cells.append("–"); mix.append("–"); continue
                sd = f" ± {r['biased_seed_sd']:.2f}" if "biased_seed_sd" in r else ""
                fl = f" ({r['failures']} failed)" if r["failures"] else ""
                cells.append(f"{r['biased']:.2f}{sd}{fl}")
                mix.append(f"{r['mixed']:.2f}")
            lines.append(f"| {m} | " + " | ".join(cells) + " | " + " | ".join(mix) + " |")
        lines += ["", f"### {suite.capitalize()} suite: paired differences, biased mean (pp) [95% CI]; seeds beating", "",
                  "| A − B | " + " | ".join(c.FAMILIES) + " |", "|---|" + "---:|" * 4]
        for a, bb in pairs:
            cells = []
            for fam in c.FAMILIES:
                r = next((x for x in comps if x["suite"] == suite and x["family"] == fam and x["a"] == a
                          and x["b"] == bb and x["group"] == "biased"), None)
                if r is None:
                    cells.append("–"); continue
                wins = f"; {r['seeds_beating']}/{r['n_policy_seeds']}" if r["n_policy_seeds"] > 1 else ""
                cells.append(c.fmt(r) + wins)
            if any(x != "–" for x in cells):
                lines.append(f"| {a} − {bb} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    for out, pairs, methods, suites in [(PS, PS_PAIRS, PS_METHODS, ["regular", "extreme"]),
                                        (RR, RR_PAIRS, RR_METHODS, ["regular", "extreme"])]:
        summary, comps = build(pairs, methods, suites)
        c.e.writecsv(out / "summary.csv", summary)
        c.e.writecsv(out / "comparisons.csv", comps)
        (out / "tables.md").write_text(md(summary, comps, pairs, methods, suites))
        print(out / "tables.md")


if __name__ == "__main__":
    main()
