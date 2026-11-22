"""Write markdown tables for results/extreme_warm_start from the analysis CSVs."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/extreme_warm_start"
FAMILIES = ("SGD", "Adam", "AdamW", "RMSprop")
ORDERS = ("mixed", "single_max", "single_late", "pair_max", "half_block", "staircase",
          "class_sorted", "alternating_halves")
CONTROLS = ("fixed_retuned", "policy_retrained", "warm_regular")
NAMES = {"fixed_retuned": "extreme-retuned fixed", "policy_retrained": "cold retrained",
         "warm_regular": "regular warm start", "warm_extreme": "extreme warm start"}


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pct(x):
    return 100 * float(x)


def main(out=OUT):
    summary = {(r["family"], r["scope"], r["variant"]): r for r in read(out / "summary.csv")}
    comp = {(r["family"], r["scope"], r["comparison"].split("-", 1)[1]): r for r in read(out / "comparisons.csv")}
    lines = []
    for scope, title in (("biased", "Seven biased orderings"), ("all", "All eight orderings (including mixed)")):
        lines += [f"### {title}", "", "Mean test accuracy (%) ± policy-seed SD [min–max]; fixed has no seed variation.", "",
                  "| Optimizer | Extreme-retuned fixed | Cold retrained | Regular warm start | **Extreme warm start** |",
                  "|---|---:|---:|---:|---:|"]
        for fam in FAMILIES:
            cells = []
            for v in (*CONTROLS, "warm_extreme"):
                s = summary[fam, scope, v]
                cell = f"{pct(s['mean']):.2f}" if v == "fixed_retuned" else \
                    f"{pct(s['mean']):.2f} ± {pct(s['sd']):.2f} [{pct(s['min']):.2f}–{pct(s['max']):.2f}]"
                if int(s["failures"]):
                    cell += f" ({s['failures']} failed)"
                cells.append(cell)
            lines.append(f"| {fam} | " + " | ".join(cells) + " |")
        lines += ["", "Extreme warm start minus control (pp, mean ± SD over the five matched policy seeds; wins out of 5):", "",
                  "| Optimizer | vs extreme-retuned fixed | vs cold retrained | vs regular warm start |", "|---|---:|---:|---:|"]
        for fam in FAMILIES:
            cells = [f"{pct(c['mean_difference']):+.2f} ± {pct(c['sd']):.2f} ({c['wins']}/5)"
                     for c in (comp[fam, scope, v] for v in CONTROLS)]
            lines.append(f"| {fam} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["### Per ordering", "",
              "Mean accuracy (%) of the extreme warm start over five seeds, then its paired difference (pp) "
              "vs extreme-retuned fixed / cold retrained / regular warm start, with wins out of 5.", ""]
    for fam in FAMILIES:
        lines += [f"**{fam}**", "", "| Ordering | Fixed | Cold | Reg. warm | Ext. warm | Δ fixed | Δ cold | Δ reg. warm |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for o in ORDERS:
            means = [f"{pct(summary[fam, o, v]['mean']):.2f}" for v in (*CONTROLS, "warm_extreme")]
            deltas = [f"{pct(c['mean_difference']):+.2f} ({c['wins']})" for c in (comp[fam, o, v] for v in CONTROLS)]
            lines.append(f"| {o} | " + " | ".join(means + deltas) + " |")
        lines.append("")
    text = "\n".join(lines)
    (out / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
