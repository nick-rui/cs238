"""Analyze the generated extreme-ordering suite (plan prespecified in design.json).

Reads results/extreme_generated/evaluation/ and orderings.csv; writes
summary.csv, policy_seed_scores.csv, comparisons.csv, severity_bins.csv,
severity_regression.csv, structure_breakdown.csv, per_ordering_differences.csv,
seven_original_comparisons.csv (if the original-suite replay exists),
tables.md and figures.

Bootstrap: for each of 10,000 resamples (numpy default_rng(238)) draw
orderings with replacement, then the five episode seeds within each drawn
ordering, then (if either side is seeded) the five policy seeds; statistic =
mean difference. Policy - policy comparisons are seed-matched.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/extreme_generated"
FAMILIES = ("SGD", "Adam", "AdamW", "RMSprop")
POLICY_SEEDS = (338, 339, 340, 341, 342)
N_RESAMPLES = 10000
RNG_SEED = 238
SEEDED = {"policy_frozen", "policy_retrained", "warm_extreme"}
BASE_METHODS = ["fixed_regular", "fixed_extreme", "cosine_regular", "policy_frozen", "policy_retrained", "warm_extreme"]
LABELS = {
    "fixed_regular": "Fixed (regular-tuned)",
    "fixed_extreme": "Fixed (extreme-retuned)",
    "cosine_regular": "Cosine (regular-tuned)",
    "policy_frozen": "Frozen regular policy",
    "policy_retrained": "Cold extreme policy",
    "warm_extreme": "Extreme warm-start policy",
}
BASE_COMPARISONS = [
    ("warm_extreme", "fixed_regular"), ("warm_extreme", "fixed_extreme"), ("warm_extreme", "cosine_regular"),
    ("warm_extreme", "policy_frozen"), ("warm_extreme", "policy_retrained"),
    ("policy_retrained", "fixed_regular"), ("policy_retrained", "fixed_extreme"),
    ("policy_retrained", "cosine_regular"), ("policy_retrained", "policy_frozen"),
    ("policy_frozen", "fixed_regular"), ("policy_frozen", "cosine_regular"),
    ("fixed_extreme", "fixed_regular"),
]


def read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_orderings(out):
    rows = read(out / "orderings.csv")
    for r in rows:
        for k in ("k", "n", "L", "G", "R", "start", "biased_batches"):
            r[k] = int(r[k])
        for k in ("mean_kl", "nearest_distance"):
            r[k] = float(r[k])
        r["episode_seeds"] = [int(s) for s in r["episode_seeds"].split()]
    return rows


def load_method(folder, orderings, seeds_by_order, policy_seeds):
    """Return accuracy (%) array (P, N, E) and failure count."""
    names = [o["order"] for o in orderings]
    acc = np.full((len(policy_seeds), len(names), len(seeds_by_order[names[0]])), np.nan)
    failures = 0
    pos = {n: i for i, n in enumerate(names)}
    for p, ps in enumerate(policy_seeds):
        rows = [r for path in sorted((folder(ps)).glob("chunk_*.csv")) for r in read(path)]
        seen = set()
        for r in rows:
            o = r["order"]
            if o not in pos:
                continue
            j = seeds_by_order[o].index(int(r["seed"]))
            key = (o, j)
            assert key not in seen, key
            seen.add(key)
            acc[p, pos[o], j] = 100 * float(r["accuracy"])
            failures += r["failed"] == "True"
        assert len(seen) == acc.shape[1] * acc.shape[2], (folder(ps), len(seen))
    assert np.isfinite(acc).all()
    return acc, failures


def diff_array(A, B):
    """A - B; equal leading dims are seed-matched, a leading dim of 1 broadcasts."""
    return A - B


def bootstrap(D, subset, B=N_RESAMPLES):
    """Two-level (+ policy seed) bootstrap of mean(D[:, subset]).

    Returns replicate means, per-ordering replicate means (B, n) and the drawn
    ordering indices (B, n) into ``subset``. Each call uses default_rng(238).
    """
    D = D[:, subset]
    P, n, e = D.shape
    rng = np.random.default_rng(RNG_SEED)
    o_idx = rng.integers(0, n, size=(B, n))
    e_idx = rng.integers(0, e, size=(B, n, e))
    p_idx = rng.integers(0, P, size=(B, P))
    per_order = np.empty((B, n))
    step = 500
    for s in range(0, B, step):
        sl = slice(s, min(B, s + step))
        m = sl.stop - sl.start
        if P > 1:
            W = np.zeros((m, P))
            np.add.at(W, (np.repeat(np.arange(m), P), p_idx[sl].ravel()), 1.0 / P)
            Dbar = (W @ D.reshape(P, -1)).reshape(m, n, e)
        else:
            Dbar = np.broadcast_to(D[0], (m, n, e))
        g = np.take_along_axis(Dbar, o_idx[sl][:, :, None], axis=1)
        g = np.take_along_axis(g, e_idx[sl], axis=2)
        per_order[sl] = g.mean(2)
    return per_order.mean(1), per_order, o_idx


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm = x - x.mean(-1, keepdims=True)
    slope = (xm * (y - y.mean(-1, keepdims=True))).sum(-1) / (xm ** 2).sum(-1)
    return slope


def fmt_ci(m, lo, hi):
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def analyze(out, extra_methods=(), extra_comparisons=(), extra_labels=None):
    if extra_labels:
        LABELS.update(extra_labels)
    orderings = load_orderings(out)
    names = [o["order"] for o in orderings]
    seeds_by_order = {o["order"]: o["episode_seeds"] for o in orderings}
    N, E = len(names), len(orderings[0]["episode_seeds"])
    methods = BASE_METHODS + [m for m, _ in extra_methods]
    seeds_of = {m: (POLICY_SEEDS if m in SEEDED else (-1,)) for m in BASE_METHODS}
    seeds_of.update(dict(extra_methods))
    comparisons = BASE_COMPARISONS + list(extra_comparisons)
    strict = np.array([not o["near_duplicate_of"] for o in orderings])
    loose = np.array([o["nearest_distance"] > 0.15 for o in orderings])
    subsets = {"all": np.arange(N), "excl_near_duplicates": np.flatnonzero(strict),
               "excl_loose_near": np.flatnonzero(loose)}
    kl = np.array([o["mean_kl"] for o in orderings])
    bb = np.array([o["biased_batches"] for o in orderings])
    edges = np.quantile(kl, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins = np.clip(np.searchsorted(edges, kl, side="right") - 1, 0, 4)

    summary, seed_scores, comps, sev_bins, sev_reg, per_order_rows, struct = [], [], [], [], [], [], []
    data = {}
    for family in FAMILIES:
        A = {}
        for m in methods:
            A[m], fails = load_method(lambda ps, m=m: out / "evaluation" / f"{family}_{m}_{ps}",
                                      orderings, seeds_by_order, seeds_of[m])
            data[family, m] = A[m]
            for sname, sub in subsets.items():
                per_seed = A[m][:, sub].mean((1, 2))
                summary.append(dict(family=family, subset=sname, method=m, orderings=len(sub),
                                    mean=float(per_seed.mean()),
                                    sd=float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
                                    min=float(per_seed.min()), max=float(per_seed.max()),
                                    policy_seeds=len(per_seed), failures=fails if sname == "all" else "",
                                    episodes=int(A[m][:, sub].size)))
                for ps, v in zip(seeds_of[m], per_seed):
                    seed_scores.append(dict(family=family, subset=sname, method=m, policy_seed=ps, accuracy=float(v)))
        for a, c in comparisons:
            if a not in A or c not in A:
                continue
            D = diff_array(A[a], A[c])
            for sname, sub in subsets.items():
                reps, per_rep, o_idx = bootstrap(D, sub)
                d_o = D[:, sub].mean((0, 2))
                seed_d = D[:, sub].mean((1, 2))
                row = dict(family=family, subset=sname, comparison=f"{a}-{c}", orderings=len(sub),
                           mean_difference=float(D[:, sub].mean()), ci_low=ci(reps)[0], ci_high=ci(reps)[1],
                           ci90_low=float(np.percentile(reps, 5)), ci90_high=float(np.percentile(reps, 95)),
                           frac_orderings_positive=float((d_o > 0).mean()),
                           frac_orderings_tied=float((d_o == 0).mean()),
                           seed_sd=float(seed_d.std(ddof=1)) if len(seed_d) > 1 else "",
                           seed_wins=int((seed_d > 0).sum()) if len(seed_d) > 1 else "",
                           policy_seeds=len(seed_d))
                if sname == "all":
                    x = kl[sub]
                    slope = float(ols(x, d_o))
                    slopes = ols(x[o_idx], per_rep)
                    xb = bb[sub]
                    slope_b = float(ols(xb, d_o))
                    slopes_b = ols(xb[o_idx], per_rep)
                    sev_reg.append(dict(family=family, comparison=f"{a}-{c}", predictor="mean_kl",
                                        intercept=float(d_o.mean() - slope * x.mean()), slope=slope,
                                        slope_ci_low=ci(slopes)[0], slope_ci_high=ci(slopes)[1],
                                        r=float(np.corrcoef(x, d_o)[0, 1])))
                    sev_reg.append(dict(family=family, comparison=f"{a}-{c}", predictor="biased_batches_per_100",
                                        intercept=float(d_o.mean() - slope_b * xb.mean()), slope=100 * slope_b,
                                        slope_ci_low=100 * ci(slopes_b)[0], slope_ci_high=100 * ci(slopes_b)[1],
                                        r=float(np.corrcoef(xb, d_o)[0, 1])))
                    for i, o in enumerate(orderings):
                        per_order_rows.append(dict(family=family, comparison=f"{a}-{c}", order=o["order"],
                                                   mean_kl=o["mean_kl"], difference=float(d_o[i])))
                    for bi in range(5):
                        bsub = np.flatnonzero(bins == bi)
                        breps, _, _ = bootstrap(D, bsub)
                        sev_bins.append(dict(family=family, comparison=f"{a}-{c}", bin=bi + 1,
                                             kl_low=float(edges[bi]), kl_high=float(edges[bi + 1]),
                                             orderings=len(bsub), mean_difference=float(D[:, bsub].mean()),
                                             ci_low=ci(breps)[0], ci_high=ci(breps)[1],
                                             frac_orderings_positive=float((D[:, bsub].mean((0, 2)) > 0).mean())))
                    # Exploratory (not prespecified): breakdown by structure.
                    for key, groups in (("k", sorted({o["k"] for o in orderings})),
                                        ("mode", ["single", "repeat", "change", "alternate"]),
                                        ("recovery", ["none", "1-20", "21-100"])):
                        for gval in groups:
                            if key == "recovery":
                                sel = [i for i, o in enumerate(orderings) if
                                       (o["R"] == 0 if gval == "none" else
                                        (1 <= o["R"] <= 20 if gval == "1-20" else o["R"] > 20))]
                            else:
                                sel = [i for i, o in enumerate(orderings) if o[key] == gval]
                            if not sel:
                                continue
                            struct.append(dict(family=family, comparison=f"{a}-{c}", factor=key, level=gval,
                                               orderings=len(sel), mean_difference=float(D[:, sel].mean()),
                                               frac_orderings_positive=float((D[:, sel].mean((0, 2)) > 0).mean())))
                comps.append(row)
    w = lambda name, rows: write_csv(out / name, rows)
    w("summary.csv", summary)
    w("policy_seed_scores.csv", seed_scores)
    w("comparisons.csv", comps)
    w("severity_bins.csv", sev_bins)
    w("severity_regression.csv", sev_reg)
    w("structure_breakdown.csv", struct)
    w("per_ordering_differences.csv", per_order_rows)
    return dict(summary=summary, comparisons=comps, sev_bins=sev_bins, sev_reg=sev_reg, struct=struct,
                methods=methods, comparisons_list=comparisons, subsets=subsets, orderings=orderings)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as handle:
        wr = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


def plot(out, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    keys = [c for c in ("warm_extreme-fixed_extreme", "warm_extreme-policy_retrained",
                        "warm_extreme-cosine_regular", "policy_retrained-fixed_extreme")
            if any(r["comparison"] == c for r in res["sev_bins"])]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.4), sharey=False)
    colors = dict(SGD="#2a6fdb", Adam="#e8743b", AdamW="#19a979", RMSprop="#945ecf")
    for ax, key in zip(np.atleast_1d(axes), keys):
        for i, fam in enumerate(FAMILIES):
            rows = [r for r in res["sev_bins"] if r["family"] == fam and r["comparison"] == key]
            xs = np.array([r["bin"] for r in rows]) + (i - 1.5) * 0.08
            m = np.array([r["mean_difference"] for r in rows])
            lo = np.array([r["ci_low"] for r in rows])
            hi = np.array([r["ci_high"] for r in rows])
            ax.errorbar(xs, m, yerr=[m - lo, hi - m], marker="o", ms=4, capsize=2, lw=1.2,
                        color=colors[fam], label=fam)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_title(key.replace("-", " − "), fontsize=9)
        ax.set_xlabel("severity quintile (mean KL, 1 = mildest)", fontsize=8)
        ax.set_xticks(range(1, 6))
    np.atleast_1d(axes)[0].set_ylabel("accuracy difference (pp)")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "difference_vs_severity.png", dpi=150)
    plt.close(fig)


def tables(out, res, original=None):
    S = {(r["family"], r["subset"], r["method"]): r for r in res["summary"]}
    C = {(r["family"], r["subset"], r["comparison"]): r for r in res["comparisons"]}
    lines = []
    for sname, title in (("all", "All 200 generated orderings"),
                         ("excl_near_duplicates", "Excluding near-duplicates of the seven originals"),
                         ("excl_loose_near", "Excluding orderings within profile distance 0.15 of an original")):
        n = S["SGD", sname, "fixed_regular"]["orderings"]
        lines += [f"### {title} (N = {n})", "",
                  "Mean final test accuracy (%) ± policy-seed SD [min–max]; failures in parentheses (all orderings).", "",
                  "| Method | " + " | ".join(FAMILIES) + " |", "|---|" + "---:|" * len(FAMILIES)]
        for m in res["methods"]:
            cells = []
            for fam in FAMILIES:
                r = S[fam, sname, m]
                cell = f"{r['mean']:.2f}"
                if r["policy_seeds"] > 1:
                    cell += f" ± {r['sd']:.2f} [{r['min']:.2f}–{r['max']:.2f}]"
                if sname == "all" and r["failures"]:
                    cell += f" ({r['failures']} failed)"
                cells.append(cell)
            lines.append(f"| {LABELS.get(m, m)} | " + " | ".join(cells) + " |")
        lines += ["", "Paired difference A − B (pp), mean [95% two-level bootstrap CI]; share of orderings with A > B; "
                  "policy seeds with positive mean difference (of 5) and policy-seed SD.", "",
                  "| A − B | " + " | ".join(FAMILIES) + " |", "|---|" + "---:|" * len(FAMILIES)]
        for a, c in res["comparisons_list"]:
            key = f"{a}-{c}"
            if not all((fam, sname, key) in C for fam in FAMILIES):
                continue
            cells = []
            for fam in FAMILIES:
                r = C[fam, sname, key]
                cell = fmt_ci(r["mean_difference"], r["ci_low"], r["ci_high"]) + f"; {100 * r['frac_orderings_positive']:.0f}%"
                if r["seed_wins"] != "":
                    cell += f"; {r['seed_wins']}/5, SD {r['seed_sd']:.2f}"
                cells.append(cell)
            lines.append(f"| {LABELS.get(a, a)} − {LABELS.get(c, c)} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["### Severity regression (all orderings)", "",
              "OLS of the per-ordering difference on mean KL (pp per nat) and on biased batches (pp per 100 batches); "
              "95% CI from the same ordering-level bootstrap.", "",
              "| A − B | Family | slope vs mean KL | r | slope vs biased batches | r |", "|---|---|---:|---:|---:|---:|"]
    R = {(r["family"], r["comparison"], r["predictor"]): r for r in res["sev_reg"]}
    for a, c in res["comparisons_list"]:
        key = f"{a}-{c}"
        for fam in FAMILIES:
            if (fam, key, "mean_kl") not in R:
                continue
            k1, k2 = R[fam, key, "mean_kl"], R[fam, key, "biased_batches_per_100"]
            lines.append(f"| {LABELS.get(a, a)} − {LABELS.get(c, c)} | {fam} | "
                         f"{fmt_ci(k1['slope'], k1['slope_ci_low'], k1['slope_ci_high'])} | {k1['r']:+.2f} | "
                         f"{fmt_ci(k2['slope'], k2['slope_ci_low'], k2['slope_ci_high'])} | {k2['r']:+.2f} |")
    lines += ["", "### Severity quintiles (all orderings)", "",
              "Mean difference (pp) [95% CI] by quintile of mean KL.", ""]
    B = res["sev_bins"]
    edges = sorted({(r["bin"], round(r["kl_low"], 3), round(r["kl_high"], 3)) for r in B if r["family"] == "SGD"})
    lines += ["| A − B | Family | " + " | ".join(f"Q{b} ({lo:.2f}–{hi:.2f})" for b, lo, hi in edges) + " |",
              "|---|---|" + "---:|" * len(edges)]
    for a, c in res["comparisons_list"]:
        key = f"{a}-{c}"
        for fam in FAMILIES:
            rows = sorted((r for r in B if r["family"] == fam and r["comparison"] == key), key=lambda r: r["bin"])
            if rows:
                lines.append(f"| {LABELS.get(a, a)} − {LABELS.get(c, c)} | {fam} | " +
                             " | ".join(fmt_ci(r["mean_difference"], r["ci_low"], r["ci_high"]) for r in rows) + " |")
    if original is not None:
        lines += ["", original]
    (out / "tables.md").write_text("\n".join(lines) + "\n")


def original_suite(out):
    """Same analysis code on the seven original biased orderings (fresh replay here)."""
    folder = out / "evaluation_original"
    if not folder.exists():
        return None, []
    sys.path.insert(0, str(ROOT / "experiments/extreme_orderings"))
    orders = ["single_max", "single_late", "pair_max", "half_block", "staircase", "class_sorted", "alternating_halves"]
    seeds = list(range(1920000, 1920020))
    orderings = [dict(order=o, episode_seeds=seeds) for o in orders]
    sbo = {o: seeds for o in orders}
    rows = []
    for family in FAMILIES:
        A = {}
        for m in BASE_METHODS:
            ps = POLICY_SEEDS if m in SEEDED else (-1,)
            if not all((folder / f"{family}_{m}_{p}").exists() for p in ps):
                continue
            A[m], _ = load_method(lambda p, m=m: folder / f"{family}_{m}_{p}", orderings, sbo, ps)
        for a, c in BASE_COMPARISONS:
            if a in A and c in A:
                D = diff_array(A[a], A[c])
                reps, _, _ = bootstrap(D, np.arange(len(orders)))
                d_o = D.mean((0, 2))
                seed_d = D.mean((1, 2))
                rows.append(dict(family=family, comparison=f"{a}-{c}", mean_difference=float(D.mean()),
                                 ci_low=ci(reps)[0], ci_high=ci(reps)[1], orderings_positive=int((d_o > 0).sum()),
                                 seed_wins=int((seed_d > 0).sum()) if len(seed_d) > 1 else ""))
    write_csv(out / "seven_original_comparisons.csv", rows)
    return rows


def replication_table(out, res, seven):
    if not seven:
        return None
    G = {(r["family"], r["comparison"]): r for r in res["comparisons"] if r["subset"] == "all"}
    O = {(r["family"], r["comparison"]): r for r in seven}
    lines = ["### Replication: seven original orderings vs generated suite", "",
             "Same code and environment; seven originals use seeds 1920000–1920019 (bootstrap over the 7 orderings, "
             "then episode seeds, then policy seeds). \"Replicates\" = both 95% CIs exclude 0 with the same sign.", "",
             "| A − B | Family | Seven originals | orderings A > B | Generated (N = 200) | orderings A > B | Sign |",
             "|---|---|---:|---:|---:|---:|---|"]
    rep = []
    for a, c in BASE_COMPARISONS:
        key = f"{a}-{c}"
        for fam in FAMILIES:
            if (fam, key) not in O or (fam, key) not in G:
                continue
            o, g = O[fam, key], G[fam, key]
            o_sig = o["ci_low"] > 0 or o["ci_high"] < 0
            g_sig = g["ci_low"] > 0 or g["ci_high"] < 0
            same = np.sign(o["mean_difference"]) == np.sign(g["mean_difference"])
            if o_sig and g_sig:
                verdict = "replicates" if same else "**reverses**"
            elif o_sig:
                verdict = "does not replicate (generated CI includes 0)"
            elif g_sig:
                verdict = "same sign, now CI excludes 0" if same else "**opposite sign, generated CI excludes 0**"
            else:
                verdict = "neither CI excludes 0"
            rep.append(dict(family=fam, comparison=key, seven=o["mean_difference"], generated=g["mean_difference"],
                            verdict=verdict))
            lines.append(f"| {LABELS[a]} − {LABELS[c]} | {fam} | {fmt_ci(o['mean_difference'], o['ci_low'], o['ci_high'])} "
                         f"| {o['orderings_positive']}/7 | {fmt_ci(g['mean_difference'], g['ci_low'], g['ci_high'])} | "
                         f"{100 * g['frac_orderings_positive']:.0f}% | {verdict} |")
    write_csv(out / "replication.csv", rep)
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extra", nargs="*", default=[], help="extra method names (unseeded unless name:seeded)")
    p.add_argument("--extra-comparisons", nargs="*", default=[], help="A-B pairs")
    a = p.parse_args()
    extra = []
    for item in a.extra:
        name, _, kind = item.partition(":")
        extra.append((name, POLICY_SEEDS if kind == "seeded" else (-1,)))
    extra_c = [tuple(x.split("-", 1)) for x in a.extra_comparisons]
    res = analyze(OUT, extra, extra_c)
    seven = original_suite(OUT)
    tables(OUT, res, replication_table(OUT, res, seven))
    plot(OUT, res)
    print((OUT / "tables.md").read_text()[:4000])


if __name__ == "__main__":
    main()
