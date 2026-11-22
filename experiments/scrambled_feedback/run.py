"""Scrambled-feedback control for the frozen regular-suite policies.

Each frozen policy is run on the original regular evaluation episodes, but the
five feedback inputs it observes at every decision (relative control loss,
recent improvement, log gradient norm, gradient-memory cosine and
gradient-previous-gradient cosine) are replaced by the values the same policy
observed at the same decision of a different evaluation episode. Elapsed time
and the policy's own current settings stay real, and the chosen settings act on
the real classifier. Donor observations come from the real-feedback
re-evaluation in results/clean_test/ (same checkpoint, same environment).

Donor maps (fixed, NumPy seed 238, identical for every policy):
  cross       donor has a different episode seed AND a different ordering
              (a permutation of the 200 episodes: each donor used once)
  same_order  donor has the same ordering but a different episode seed
              (a derangement of the 20 seeds within each ordering)

Modes: donors, evaluate, launch, summarize. Outputs: results/scrambled_feedback/.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/clean_test"))
import common as c  # noqa: E402

OUT = c.RESULTS / "scrambled_feedback"
CLEAN = c.RESULTS / "clean_test/units/regular"
DONORS = ["cross", "same_order"]
RNG_SEED = 238


def donor_map(kind):
    """Return {(order, seed): (donor_order, donor_seed)} for the regular suite."""
    orders, seeds = list(c.b.ORDERS), list(c.b.TESTSEEDS)
    episodes = [(o, s) for o in orders for s in seeds]
    rng = np.random.default_rng(RNG_SEED)
    if kind == "same_order":
        mapping = {}
        for o in orders:
            while True:
                perm = rng.permutation(len(seeds))
                if all(perm[i] != i for i in range(len(seeds))):
                    break
            mapping.update({(o, seeds[i]): (o, seeds[perm[i]]) for i in range(len(seeds))})
        return mapping
    if kind != "cross":
        raise ValueError(kind)
    n = len(episodes)
    perm = rng.permutation(n)

    def bad(i, j):
        return episodes[i][0] == episodes[j][0] or episodes[i][1] == episodes[j][1]

    # Repair conflicts by swapping donors with random partners while both stay valid.
    for _ in range(100000):
        conflicts = [i for i in range(n) if bad(i, perm[i])]
        if not conflicts:
            break
        i = conflicts[0]
        k = int(rng.integers(n))
        if not bad(i, perm[k]) and not bad(k, perm[i]):
            perm[i], perm[k] = perm[k], perm[i]
    else:
        raise RuntimeError("no valid cross donor permutation")
    return {episodes[i]: episodes[perm[i]] for i in range(n)}


def check_maps():
    for kind in DONORS:
        m = donor_map(kind)
        assert len(m) == 200 and sorted(m.values()) == sorted(m), kind  # permutation
        for (o, s), (do, ds) in m.items():
            assert ds != s
            assert (do != o) if kind == "cross" else (do == o)
        assert m == donor_map(kind)  # deterministic


def donors(_args):
    check_maps()
    rows = [dict(donor=kind, order=o, seed=s, donor_order=do, donor_seed=ds)
            for kind in DONORS for (o, s), (do, ds) in sorted(donor_map(kind).items())]
    c.write_csv(OUT / "donors.csv", rows)
    print("DONOR MAPS OK", len(rows), flush=True)


def load_observations(family, seed):
    data = np.load(CLEAN / "policy" / f"{family}_{seed}" / "observations.npz")
    return {(str(o), int(s)): torch.from_numpy(x) for o, s, x in zip(data["orders"], data["seeds"], data["obs"])}


def unit_folder(donor, family, seed):
    return OUT / "units" / donor / f"{family}_{seed}"


def evaluate(args):
    family, seed, donor = args.family, args.seed, args.donor
    folder = unit_folder(donor, family, seed) if not args.smoke else OUT / "smoke" / donor / f"{family}_{seed}"
    if (folder / "done.json").exists():
        print("Already completed", folder, flush=True)
        return
    threads = args.threads or c.threads_for("regular", "policy", family, seed)
    torch.set_num_threads(threads)
    orders, seeds = c.suite_design("regular")
    if args.smoke:
        orders, seeds = orders[1:2], seeds[:4]
    official, clean = c.official_testset(), c.clean_testset()
    pol, checkpoint = c.load_policy("policy", family, seed)
    wrapper = c.FeedbackPolicy(pol, family)
    observed = load_observations(family, seed)
    mapping = donor_map(donor)
    cols = c.feedback_columns(family)
    unit = dict(suite="regular", method="policy", family=family, seed=seed)
    rows = []
    start = time.perf_counter()
    for order in orders:
        for st in range(0, len(seeds), 4):
            ss = seeds[st:st + 4]
            sources = [mapping[(order, s)] for s in ss]
            wrapper.start(torch.stack([observed[k][:, cols] for k in sources]))
            rr = c.evaluate_batch(unit, ss, [order] * len(ss), official, clean, pol=wrapper)
            real = wrapper.observations()[:, :, cols]
            for i, r in enumerate(rr):
                own = observed[(order, ss[i])][:, cols]
                donor_obs = observed[sources[i]][:, cols]
                r.update(method=f"scrambled_{donor}", family=family, policy_seed=seed, checkpoint=checkpoint,
                         threads=threads, donor_order=sources[i][0], donor_seed=sources[i][1],
                         # Mean |fed - realized| feedback; and whether the first decision's
                         # feedback (identical initial state statistics) differs.
                         mean_abs_feedback_error=float((donor_obs - real[i]).abs().mean()),
                         mean_abs_donor_vs_real_feedback=float((donor_obs - own).abs().mean()))
            rows += rr
        print(json.dumps(dict(donor=donor, family=family, seed=seed, order=order,
                              seconds=round(time.perf_counter() - start, 1))), flush=True)
    c.write_csv(folder / "evaluation.csv", rows)
    c.write_json(folder / "done.json", dict(donor=donor, family=family, seed=seed, threads=threads,
                                            checkpoint=checkpoint, rows=len(rows),
                                            failures=sum(bool(r["failed"]) for r in rows),
                                            seconds=time.perf_counter() - start, torch=torch.__version__))
    print("DONE", folder, flush=True)


def units():
    return [dict(donor=d, family=fam, seed=s, threads=c.threads_for("regular", "policy", fam, s))
            for d in DONORS for fam in c.FAMILIES for s in c.POLICY_SEEDS]


def launch(args):
    todo = [u for u in units() if not (unit_folder(u["donor"], u["family"], u["seed"]) / "done.json").exists()]
    if args.only:
        todo = [u for u in todo if any(re.fullmatch(p, f"{u['donor']}/{u['family']}_{u['seed']}") for p in args.only)]
    for u in todo:
        assert (CLEAN / "policy" / f"{u['family']}_{u['seed']}" / "done.json").exists(), \
            f"real-feedback observations missing for {u}; run experiments/clean_test/run.py launch first"
    todo.sort(key=lambda u: u["family"] == "SGD")
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    slots = threading.Semaphore(args.cpus)
    print(f"LAUNCH {len(todo)} units with {args.cpus} CPU slots", flush=True)

    def job(u):
        for _ in range(u["threads"]):
            slots.acquire()
        try:
            name = f"{u['donor']}/{u['family']}_{u['seed']}"
            cmd = [sys.executable, str(Path(__file__).resolve()), "evaluate", "--donor", u["donor"],
                   "--family", u["family"], "--seed", str(u["seed"]), "--threads", str(u["threads"])]
            env = {**os.environ, "OMP_NUM_THREADS": str(u["threads"]), "MKL_NUM_THREADS": str(u["threads"])}
            print("START", name, flush=True)
            with (logs / (name.replace("/", "__") + ".log")).open("a") as log:
                code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            print("FINISHED" if code == 0 else f"FAILED({code})", name, flush=True)
            return code
        finally:
            for _ in range(u["threads"]):
                slots.release()

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.cpus) as pool:
        codes = list(pool.map(job, todo))
    print(f"LAUNCH COMPLETE failures={sum(x != 0 for x in codes)} wall_seconds={time.perf_counter() - start:.0f}",
          flush=True)


def arm_rows(arm, family, seed):
    if arm == "real":
        return c.read_csv(CLEAN / "policy" / f"{family}_{seed}" / "evaluation.csv")
    if arm == "fixed":
        return c.read_csv(CLEAN / "fixed" / family / "evaluation.csv")
    if arm == "open_loop":
        return c.read_csv(CLEAN / "open_loop" / f"{family}_{seed}" / "evaluation.csv")
    return c.read_csv(unit_folder(arm.replace("scrambled_", ""), family, seed) / "evaluation.csv")


def scope_mean(rows, metric, scope):
    groups = {}
    for r in rows:
        if scope == "all" or (scope == "biased") == (r["order"] != "mixed"):
            groups.setdefault(r["order"], []).append(float(r[metric]))
    return 100 * c.mean(c.mean(v) for v in groups.values())


ARMS = ["real", "scrambled_cross", "scrambled_same_order", "open_loop", "fixed"]
CONTRASTS = [("real", "scrambled_cross"), ("real", "scrambled_same_order"), ("scrambled_cross", "fixed"),
             ("scrambled_same_order", "fixed"), ("real", "fixed"), ("scrambled_cross", "open_loop"),
             ("real", "open_loop"), ("scrambled_same_order", "scrambled_cross")]


def summarize(_args):
    arms_table, contrasts, seed_rows, diag = [], [], [], []
    for family in c.FAMILIES:
        cache = {(arm, s): arm_rows(arm, family, s) for arm in ARMS for s in c.POLICY_SEEDS
                 if arm != "open_loop" or s in c.ORIGINAL_SEEDS}
        for arm in ARMS[:3]:
            rr = [r for s in c.POLICY_SEEDS for r in cache[arm, s]]
            diag.append(dict(family=family, arm=arm, episodes=len(rr), failures=sum(c.is_true(r["failed"]) for r in rr),
                             **({} if arm == "real" else dict(
                                 mean_abs_donor_vs_real_feedback=c.mean(float(r["mean_abs_donor_vs_real_feedback"]) for r in rr),
                                 mean_abs_fed_vs_realized_feedback=c.mean(float(r["mean_abs_feedback_error"]) for r in rr)))))
        for test in ["official", "clean"]:
            metric = f"{test}_accuracy"
            for scope in ["biased", "shuffled", "all"]:
                value = {}
                for arm in ARMS:
                    for s in c.POLICY_SEEDS:
                        if (arm, s) in cache:
                            value[arm, s] = scope_mean(cache[arm, s], metric, scope)
                            seed_rows.append(dict(family=family, test=test, scope=scope, arm=arm, policy_seed=s,
                                                  accuracy_pct=value[arm, s]))
                for arm in ARMS:
                    vals = [value[arm, s] for s in c.POLICY_SEEDS if (arm, s) in value]
                    vals2 = [value[arm, s] for s in c.ORIGINAL_SEEDS]
                    arms_table.append(dict(family=family, test=test, scope=scope, arm=arm,
                                           seeds=len(vals) if arm != "fixed" else 0,
                                           mean_pct=c.mean(vals), sd_pct=c.sample_sd(vals) if arm != "fixed" else 0.0,
                                           mean_seeds_338_339_pct=c.mean(vals2)))
                for a, bname in CONTRASTS:
                    seeds = c.ORIGINAL_SEEDS if "open_loop" in (a, bname) else c.POLICY_SEEDS
                    d = [value[a, s] - value[bname, s] for s in seeds]
                    contrasts.append(dict(family=family, test=test, scope=scope, contrast=f"{a} - {bname}",
                                          seeds=len(seeds), mean_pp=c.mean(d), sd_pp=c.sample_sd(d),
                                          wins=sum(x > 0 for x in d), per_seed_pp=" ".join(f"{x:+.3f}" for x in d)))
    c.write_csv(OUT / "arms.csv", arms_table)
    c.write_csv(OUT / "contrasts.csv", contrasts)
    c.write_csv(OUT / "policy_seed_accuracy.csv", seed_rows)
    c.write_csv(OUT / "diagnostics.csv", diag)
    lines = []
    for test in ["official", "clean"]:
        lines += [f"\n### {test} test, regular biased (nine orderings)\n",
                  "| Optimizer | Fixed | Open loop (338–339) | Real feedback | Scrambled, cross | Scrambled, same order | Real − cross (wins/5) | Cross − fixed (wins/5) | Cross − open loop (338–339) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for family in c.FAMILIES:
            a = {r["arm"]: r for r in arms_table if r["family"] == family and r["test"] == test and r["scope"] == "biased"}
            k = {r["contrast"]: r for r in contrasts if r["family"] == family and r["test"] == test and r["scope"] == "biased"}

            def cell(name):
                r = k[name]
                return f"{r['mean_pp']:+.2f} ± {r['sd_pp']:.2f} ({r['wins']}/{r['seeds']})"
            lines.append(f"| {family} | {a['fixed']['mean_pct']:.2f} | {a['open_loop']['mean_pct']:.2f} | "
                         f"{a['real']['mean_pct']:.2f} ± {a['real']['sd_pct']:.2f} | "
                         f"{a['scrambled_cross']['mean_pct']:.2f} ± {a['scrambled_cross']['sd_pct']:.2f} | "
                         f"{a['scrambled_same_order']['mean_pct']:.2f} ± {a['scrambled_same_order']['sd_pct']:.2f} | "
                         f"{cell('real - scrambled_cross')} | {cell('scrambled_cross - fixed')} | "
                         f"{cell('scrambled_cross - open_loop')} |")
    (OUT / "summary_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["donors", "evaluate", "launch", "summarize"])
    parser.add_argument("--donor", choices=DONORS)
    parser.add_argument("--family", choices=c.FAMILIES)
    parser.add_argument("--seed", type=int, choices=c.POLICY_SEEDS)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cpus", type=int, default=6)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.mode == "evaluate" and not (args.donor and args.family and args.seed):
        parser.error("evaluate needs --donor, --family and --seed")
    {"donors": donors, "evaluate": evaluate, "launch": launch, "summarize": summarize}[args.mode](args)
