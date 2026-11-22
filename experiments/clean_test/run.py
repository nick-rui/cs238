"""Re-evaluate saved policies and fixed baselines on the 6,000 unused training-split images.

Modes:
  audit      verify the 6,000 images are never touched by training, control,
             development, tuning or test code, now or in any committed revision
  evaluate   one unit (suite/method/family/seed), scoring each final classifier
             on both the official test split and the clean split
  launch     all units, resuming completed ones, within a CPU-slot budget
  summarize  tables and CSVs from completed units
  check      fast self-checks (capture equivalence, wrapper pass-through)

Outputs go to results/clean_test/ only. No training or selection happens here.
"""

import argparse
import hashlib
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

OUT = c.RESULTS / "clean_test"


# --------------------------------------------------------------------------- audit

STATIC_PATTERN = r"\b(e\.)?(X|Y|ix)\[|\be\.(X|Y|ix)\b|train=True|randperm\(60000|randperm\(len\(train\)"
# Every line that may touch the full 60,000-image training split. Anything else
# matching STATIC_PATTERN in any revision fails the audit.
ALLOWED_LINES = {
    'data = MNIST(str(DATA_DIR), train=True, download=True)',
    'ix = torch.randperm(60000, generator=torch.Generator().manual_seed(238))',
    'x, y = X[ix[:50000]], Y[ix[:50000]]',
    'xc, yc = X[ix[50000:52000]], Y[ix[50000:52000]]',
    'xd, yd = X[ix[52000:54000]], Y[ix[52000:54000]]',
    # CIFAR-10 timing harness: same 50k/2k/2k slices (40k for CIFAR); timing only.
    'train = dataset_class(str(e.DATA_DIR), train=True, download=True)',
    'ix = torch.randperm(len(train), generator=torch.Generator().manual_seed(238))',
    'e.x, e.y = xx[ix[:ntrain]], yy[ix[:ntrain]]',
    'e.xc, e.yc = xx[ix[ntrain:ntrain + 2000]], yy[ix[ntrain:ntrain + 2000]]',
    'e.xd, e.yd = xx[ix[ntrain + 2000:ntrain + 4000]], yy[ix[ntrain + 2000:ntrain + 4000]]',
}
# This audit's own module reads the clean split by design.
AUDIT_EXEMPT = ("experiments/clean_test/", "experiments/scrambled_feedback/")


class Forbidden:
    """Stand-in for the full training-split tensors: any use raises."""

    def __init__(self, name):
        self.name = name

    def __getattr__(self, attr):
        raise AssertionError(f"full training split accessed: {self.name}.{attr}")

    def __getitem__(self, key):
        raise AssertionError(f"full training split indexed: {self.name}[...]")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=c.ROOT, text=True)


def static_scan():
    # Every pushed revision (main and all PR branches) plus this branch. Other
    # sessions' unpushed local branches are in-progress work, not committed runs.
    revisions = git("rev-list", "--remotes", "HEAD").split()
    hits, unexpected = {}, []
    for rev in revisions:
        out = subprocess.run(["git", "grep", "-n", "-E", STATIC_PATTERN, rev, "--", "*.py", "*.ipynb"],
                             cwd=c.ROOT, text=True, capture_output=True).stdout
        for line in out.splitlines():
            _, path, _, text = line.split(":", 3)
            if path.startswith(AUDIT_EXEMPT):
                continue
            text = text.strip()
            hits.setdefault((path, text), set()).add(rev[:10])
            if text not in ALLOWED_LINES:
                unexpected.append(dict(revision=rev, path=path, line=text))
    # Also scan uncommitted working-tree files of other experiments.
    for path in c.ROOT.rglob("*.py"):
        rel = str(path.relative_to(c.ROOT))
        if rel.startswith((".venv",) + AUDIT_EXEMPT):
            continue
        for text in path.read_text().splitlines():
            if re.search(STATIC_PATTERN, text):
                hits.setdefault((rel, text.strip()), set()).add("working_tree")
                if text.strip() not in ALLOWED_LINES:
                    unexpected.append(dict(revision="working_tree", path=rel, line=text.strip()))
    split_history = git("log", "--remotes", "HEAD", "--format=%h %s", "-G", r"ix\[|randperm\(60000", "--", "*.py").splitlines()
    return dict(revisions_scanned=len(revisions), pattern=STATIC_PATTERN,
                matches=[dict(path=p, line=t, revisions=len(r)) for (p, t), r in sorted(hits.items())],
                unexpected=unexpected, commits_touching_split_lines=split_history)


def runtime_check():
    """Run every kind of rollout the experiments perform with X/Y/ix poisoned."""
    e, b, f = c.e, c.b, c.f
    saved = (e.X, e.Y, e.ix, e.data)
    torch.set_num_threads(1)
    test = c.official_testset()
    e.X, e.Y, e.ix, e.data = (Forbidden(n) for n in ["X", "Y", "ix", "data"])
    exercised = []
    try:
        pol = e.policy("gradient", 338)
        e.rollout(pol, "gradient", [5000000, 5000001], ["single_train", "pair_train"], training=True)
        e.rollout(pol, "gradient", b.DEVSEEDS[:2], b.DEVORDERS[:2], evaluation=(e.xd, e.yd))
        e.rollout(pol, "gradient", b.TESTSEEDS[:2], ["single_middle"] * 2, evaluation=test, trace=True)
        b.baseline(c.fixed_config("SGD", "fixed"), b.DEVSEEDS[0], b.DEVORDERS[0], (e.xd, e.yd))
        b.baseline(c.fixed_config("SGD", "fixed"), b.TESTSEEDS[0], "mixed", test, True)
        exercised += ["SGD policy train/dev/test", "SGD fixed tune/test"]
        for family in f.FAMILIES:
            pol = f.policy(family, "gradient", 338)
            f.rollout(family, "gradient", [5000000, 5000001], ["double_train", "mixed"], pol, training=True)
            f.rollout(family, "gradient", b.DEVSEEDS[:2], b.DEVORDERS[:2], pol, evaluation=(e.xd, e.yd))
            f.rollout(family, "gradient", b.TESTSEEDS[:2], ["ramp"] * 2, pol, evaluation=test, trace=True)
            f.rollout(family, "gradient", b.DEVSEEDS[:2], b.DEVORDERS[:2], fixed=c.fixed_config(family, "fixed"),
                      evaluation=(e.xd, e.yd))
            exercised.append(f"{family} policy train/dev/test and fixed tune")
        stress = c.load_module("stress_streams_audit", "experiments/extreme_orderings/run.py")
        f.rollout("Adam", "gradient", stress.SEEDS[:2], ["class_sorted", "half_block"],
                  f.policy("Adam", "gradient", 338), evaluation=test)
        exercised.append("extreme streams")
    finally:
        e.X, e.Y, e.ix, e.data = saved
    return exercised


def audit(_args):
    e = c.e
    ix = e.ix
    n = len(ix)
    assert n == 60000 and torch.equal(torch.sort(ix).values, torch.arange(60000))
    assert torch.equal(ix, torch.randperm(60000, generator=torch.Generator().manual_seed(238)))
    parts = {"classifier_train": ix[:50000], "control": ix[50000:52000],
             "development": ix[52000:54000], "clean": ix[54000:]}
    sets = {k: set(v.tolist()) for k, v in parts.items()}
    for a in sets:
        for bname in sets:
            if a < bname:
                assert not sets[a] & sets[bname], (a, bname)
    assert sum(len(s) for s in sets.values()) == 60000
    # The engine's working arrays are exactly the first three slices.
    assert torch.equal(e.x, e.X[parts["classifier_train"]]) and torch.equal(e.y, e.Y[parts["classifier_train"]])
    assert torch.equal(e.xc, e.X[parts["control"]]) and torch.equal(e.xd, e.X[parts["development"]])
    # Stream generators index only e.x (class_ids are positions in e.y, < 50,000).
    assert all(int(q.max()) < 50000 for q in c.b.class_ids)
    xs, ys = c.clean_testset()
    official = c.official_testset()

    def digests(t):
        return {hashlib.sha1(row.numpy().tobytes()).hexdigest() for row in (t * 255).round().to(torch.uint8)}

    clean_h = digests(xs)
    used_h = digests(torch.cat([e.x, e.xc, e.xd]))
    test_h = digests(official[0])
    static = static_scan()
    exercised = runtime_check()
    result = dict(
        permutation="torch.randperm(60000, Generator.manual_seed(238)) recomputed and equal to engine.ix",
        slices={k: [int(v.min()), int(v.max()), len(v)] for k, v in parts.items()},
        disjoint=True,
        clean_label_counts=torch.bincount(ys, minlength=10).tolist(),
        clean_pixel_duplicates_of_used_images=len(clean_h & used_h),
        clean_pixel_duplicates_of_official_test=len(clean_h & test_h),
        clean_unique_images=len(clean_h),
        static_scan=static,
        runtime_poisoned_rollouts=exercised,
        runtime_result="engine.X, engine.Y, engine.ix and engine.data replaced by objects that raise on any use; "
                       "all exercised training, development, tuning and test paths ran without touching them",
    )
    if static["unexpected"]:
        result["passed"] = False
        c.write_json(OUT / "audit.json", result)
        raise SystemExit(f"Unexpected split access: {static['unexpected'][:5]}")
    result["passed"] = True
    c.write_json(OUT / "audit.json", result)
    print("AUDIT PASSED", json.dumps({k: result[k] for k in ["slices", "clean_label_counts",
          "clean_pixel_duplicates_of_used_images", "clean_pixel_duplicates_of_official_test"]}), flush=True)


# ---------------------------------------------------------------------- evaluate

def unit_folder(unit):
    return OUT / "units" / c.unit_name(unit["suite"], unit["method"], unit["family"], unit["seed"])


def evaluate(args):
    unit = dict(suite=args.suite, method=args.method, family=args.family, seed=args.seed,
                threads=args.threads or c.threads_for(args.suite, args.method, args.family, args.seed))
    folder = unit_folder(unit) if not args.smoke else OUT / "smoke" / c.unit_name(
        unit["suite"], unit["method"], unit["family"], unit["seed"])
    if (folder / "done.json").exists():
        print("Already completed", folder, flush=True)
        return
    orders, seeds = c.suite_design(unit["suite"])
    torch.set_num_threads(unit["threads"])
    if args.smoke:
        orders, seeds = orders[1:2], seeds[:4]
    official, clean = c.official_testset(), c.clean_testset()
    method, family, seed = unit["method"], unit["family"], unit["seed"]
    pol = config = schedule = warm = None
    checkpoint = -1
    if method in ["fixed", "fixed_retuned"]:
        config = c.fixed_config(family, method)
    elif method == "open_loop":
        schedule = c.open_loop_schedule(family, seed)
    else:
        pol, checkpoint = c.load_policy(method, family, seed)
        if method == "warm":
            warm = c.fixed_config(family, "fixed")
    recorder = c.FeedbackPolicy(pol, family) if pol is not None else None
    rows, obs = [], {}
    start = time.perf_counter()
    for order in orders:
        for st in range(0, len(seeds), 4):
            ss = seeds[st:st + 4]
            if recorder is not None:
                recorder.start()
            rr = c.evaluate_batch(unit, ss, [order] * len(ss), official, clean,
                                  pol=recorder, config=config, schedule=schedule, warm=warm)
            if recorder is not None:
                o = recorder.observations()
                for i, s in enumerate(ss):
                    obs[(order, s)] = o[i].numpy()
            for r in rr:
                r.update(suite=unit["suite"], method=method, family=family, policy_seed=seed,
                         checkpoint=checkpoint, threads=unit["threads"])
            rows += rr
        print(json.dumps(dict(unit=c.unit_name(unit["suite"], method, family, seed), order=order,
                              seconds=round(time.perf_counter() - start, 1))), flush=True)
    c.write_csv(folder / "evaluation.csv", rows)
    if recorder is not None:
        keys = sorted(obs)
        np.savez_compressed(folder / "observations.npz", orders=np.array([k[0] for k in keys]),
                            seeds=np.array([k[1] for k in keys]),
                            obs=np.stack([obs[k] for k in keys]).astype(np.float32))
    drift = {}
    if not args.smoke:
        ref = c.committed_reference(unit["suite"], method, family, seed)
        assert set(ref) == {(r["order"], r["seed"]) for r in rows}, "episode grid differs from committed run"
        d = [r["official_accuracy"] - float(ref[(r["order"], r["seed"])]["accuracy"]) for r in rows]
        drift = dict(committed_rows=len(ref), max_abs_accuracy_drift=float(np.max(np.abs(d))),
                     mean_accuracy_drift_pp=100 * float(np.mean(d)),
                     rows_exact=int(sum(abs(x) < 1e-7 for x in d)),
                     failed_mismatches=sum(c.is_true(ref[(r["order"], r["seed"])]["failed"]) != bool(r["failed"])
                                           for r in rows))
    c.write_json(folder / "done.json", dict(unit, rows=len(rows), checkpoint=checkpoint,
                                            seconds=time.perf_counter() - start,
                                            failures=sum(bool(r["failed"]) for r in rows),
                                            torch=torch.__version__, **drift))
    print("DONE", folder, json.dumps(drift), flush=True)


def launch(args):
    todo = [u for u in c.units() if not (unit_folder(u) / "done.json").exists()]
    if args.only:
        todo = [u for u in todo if any(re.fullmatch(p, c.unit_name(u["suite"], u["method"], u["family"], u["seed"]))
                                       for p in args.only)]
    # Longest first: adaptive families and extreme/regular policies dominate.
    todo.sort(key=lambda u: (u["family"] == "SGD", u["method"] in ["fixed", "fixed_retuned", "open_loop"]))
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    slots = threading.Semaphore(args.cpus)
    print(f"LAUNCH {len(todo)} units with {args.cpus} CPU slots", flush=True)

    def job(u):
        for _ in range(u["threads"]):
            slots.acquire()
        try:
            name = c.unit_name(u["suite"], u["method"], u["family"], u["seed"])
            cmd = [sys.executable, str(Path(__file__).resolve()), "evaluate", "--suite", u["suite"],
                   "--method", u["method"], "--family", u["family"], "--seed", str(u["seed"]),
                   "--threads", str(u["threads"])]
            env = {**os.environ, "OMP_NUM_THREADS": str(u["threads"]), "MKL_NUM_THREADS": str(u["threads"])}
            print("START", name, flush=True)
            with (logs / (name.replace("/", "__") + ".log")).open("a") as log:
                code = subprocess.run(cmd, cwd=c.ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
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


# --------------------------------------------------------------------- summarize

def load_units():
    data = {}
    for u in c.units():
        folder = unit_folder(u)
        if not (folder / "done.json").exists():
            raise SystemExit(f"Incomplete unit: {folder}")
        data[u["suite"], u["method"], u["family"], u["seed"]] = c.read_csv(folder / "evaluation.csv")
    return data


def condition_mean(rows, metric, scope):
    keep = [r for r in rows if scope_match(r["order"], scope)]
    by_order = {}
    for r in keep:
        by_order.setdefault(r["order"], []).append(float(r[metric]))
    return c.mean(c.mean(v) for v in by_order.values())


def scope_match(order, scope):
    return {"biased": order != "mixed", "shuffled": order == "mixed", "all": True}[scope]


COMPARISONS = [
    # name, suite, scope, method, comparator method, comparator seeds matched?
    ("frozen policy vs fixed", "regular", "biased", "policy", "fixed"),
    ("frozen policy vs fixed", "regular", "shuffled", "policy", "fixed"),
    ("frozen policy vs fixed", "extreme", "biased", "policy", "fixed"),
    ("retrained vs extreme-retuned fixed", "extreme", "biased", "retrained", "fixed_retuned"),
    ("retrained vs saved fixed", "extreme", "biased", "retrained", "fixed"),
    ("retrained vs frozen (seed-matched)", "extreme", "biased", "retrained", "policy"),
    ("warm vs fixed", "regular", "biased", "warm", "fixed"),
    ("warm vs fixed", "extreme", "all", "warm", "fixed"),
    ("warm vs full (seeds 338-339)", "regular", "biased", "warm", "policy"),
    ("open loop vs fixed (seeds 338-339)", "regular", "biased", "open_loop", "fixed"),
    ("frozen policy vs open loop (seeds 338-339)", "regular", "biased", "policy", "open_loop"),
]


def summarize(_args):
    data = load_units()
    summary, seed_rows, drift_rows = [], [], []
    for (name, suite, scope, method, comparator) in COMPARISONS:
        for family in c.FAMILIES:
            seeds = c.ORIGINAL_SEEDS if ("338-339" in name) else c.POLICY_SEEDS
            out = dict(comparison=name, suite=suite, scope=scope, family=family, policy_seeds=len(seeds))
            for test in ["official", "clean"]:
                metric = f"{test}_accuracy"
                values, deltas = [], []
                for s in seeds:
                    rows = data[suite, method, family, s]
                    comp_seed = s if data.get((suite, comparator, family, s)) is not None else -1
                    comp = data[suite, comparator, family, comp_seed]
                    v = 100 * condition_mean(rows, metric, scope)
                    cv = 100 * condition_mean(comp, metric, scope)
                    values.append(v)
                    deltas.append(v - cv)
                    seed_rows.append(dict(comparison=name, suite=suite, scope=scope, family=family, test=test,
                                          policy_seed=s, method_pct=v, comparator_pct=cv, delta_pp=v - cv))
                comp_values = [r["comparator_pct"] for r in seed_rows[-len(seeds):]]
                out.update({f"{test}_method_mean_pct": c.mean(values), f"{test}_method_sd_pct": c.sample_sd(values),
                            f"{test}_comparator_mean_pct": c.mean(comp_values),
                            f"{test}_delta_pp": c.mean(deltas), f"{test}_delta_sd_pp": c.sample_sd(deltas),
                            f"{test}_wins": sum(d > 0 for d in deltas)})
            out["clean_minus_official_method_pp"] = out["clean_method_mean_pct"] - out["official_method_mean_pct"]
            out["clean_minus_official_comparator_pp"] = out["clean_comparator_mean_pct"] - out["official_comparator_mean_pct"]
            out["clean_minus_official_delta_pp"] = out["clean_delta_pp"] - out["official_delta_pp"]
            summary.append(out)
    for u in c.units():
        done = json.loads((unit_folder(u) / "done.json").read_text())
        rows = data[u["suite"], u["method"], u["family"], u["seed"]]
        drift_rows.append(dict(unit=c.unit_name(u["suite"], u["method"], u["family"], u["seed"]), threads=u["threads"],
                               rows=len(rows), failures=done["failures"],
                               fresh_official_mean_pct=100 * c.mean(float(r["official_accuracy"]) for r in rows),
                               clean_mean_pct=100 * c.mean(float(r["clean_accuracy"]) for r in rows),
                               rows_matching_committed=done["rows_exact"],
                               max_abs_accuracy_drift_vs_committed=done["max_abs_accuracy_drift"],
                               mean_accuracy_drift_vs_committed_pp=done["mean_accuracy_drift_pp"],
                               failed_flag_mismatches=done["failed_mismatches"]))
    per_order = []
    for (suite, method, family, s), rows in sorted(data.items()):
        for order in sorted({r["order"] for r in rows}):
            rr = [r for r in rows if r["order"] == order]
            per_order.append(dict(suite=suite, method=method, family=family, policy_seed=s, order=order,
                                  episodes=len(rr), official_accuracy_pct=100 * c.mean(float(r["official_accuracy"]) for r in rr),
                                  clean_accuracy_pct=100 * c.mean(float(r["clean_accuracy"]) for r in rr),
                                  failures=sum(c.is_true(r["failed"]) for r in rr)))
    c.write_csv(OUT / "summary.csv", summary)
    c.write_csv(OUT / "policy_seed_differences.csv", seed_rows)
    c.write_csv(OUT / "replay_drift.csv", drift_rows)
    c.write_csv(OUT / "per_order.csv", per_order)
    lines = ["| Comparison | Suite (scope) | Optimizer | Comparator official → clean | Method official → clean (mean ± SD) | Delta official → clean (pp) | Wins official / clean |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(
            f"| {r['comparison']} | {r['suite']} ({r['scope']}) | {r['family']} | "
            f"{r['official_comparator_mean_pct']:.2f} → {r['clean_comparator_mean_pct']:.2f} | "
            f"{r['official_method_mean_pct']:.2f} ± {r['official_method_sd_pct']:.2f} → "
            f"{r['clean_method_mean_pct']:.2f} ± {r['clean_method_sd_pct']:.2f} | "
            f"{r['official_delta_pp']:+.2f} → {r['clean_delta_pp']:+.2f} | "
            f"{r['official_wins']}/{r['policy_seeds']} / {r['clean_wins']}/{r['policy_seeds']} |")
    (OUT / "summary_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ------------------------------------------------------------------------- check

def check(_args):
    """Capture recomputation equals engine metrics; wrapper passes through exactly."""
    torch.set_num_threads(1)
    c.suite_design("regular")
    official, clean = c.official_testset(), c.clean_testset()
    results = []
    for family in c.FAMILIES:
        unit = dict(suite="regular", method="policy", family=family, seed=338)
        pol, _ = c.load_policy("policy", family, 338)
        seeds, orders = c.b.TESTSEEDS[:2], ["single_middle", "rapid"]
        if family == "SGD":
            plain, *_ = c.e.rollout(pol, "gradient", seeds, orders, evaluation=official, trace=True)
        else:
            plain, *_ = c.f.rollout(family, "gradient", seeds, orders, pol, evaluation=official, trace=True)
        wrapper = c.FeedbackPolicy(pol, family)
        wrapper.start()
        rows = c.evaluate_batch(unit, seeds, orders, official, clean, pol=wrapper)
        real = wrapper.observations()
        assert [r["official_accuracy"] for r in rows] == [r["accuracy"] for r in plain], family
        # Self-donor: replacing feedback with the episode's own feedback is a no-op.
        wrapper.start(real[:, :, c.feedback_columns(family)])
        again = c.evaluate_batch(unit, seeds, orders, official, clean, pol=wrapper)
        assert [r["clean_accuracy"] for r in again] == [r["clean_accuracy"] for r in rows], family
        assert torch.equal(wrapper.observations(), real)
        # Engine identity: the clean set really is different data.
        assert all(abs(r["clean_accuracy"] - r["official_accuracy"]) > 0 for r in rows)
        results.append(dict(family=family, wrapper_passthrough_exact=True, self_donor_noop=True,
                            capture_matches_engine=True))
    c.write_json(OUT / "checks.json", dict(checks=results))
    print("CHECKS PASSED", json.dumps(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["audit", "evaluate", "launch", "summarize", "check"])
    parser.add_argument("--suite", choices=["regular", "extreme"])
    parser.add_argument("--method", choices=c.METHODS)
    parser.add_argument("--family", choices=c.FAMILIES)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cpus", type=int, default=6, help="CPU-thread slots for launch (2-thread units take two)")
    parser.add_argument("--only", nargs="+", help="Regexes over unit names for launch")
    parser.add_argument("--smoke", action="store_true", help="Evaluate four episodes of one ordering into smoke/")
    args = parser.parse_args()
    if args.mode == "evaluate" and not (args.suite and args.method and args.family):
        parser.error("evaluate needs --suite, --method and --family")
    {"audit": audit, "evaluate": evaluate, "launch": launch, "summarize": summarize, "check": check}[args.mode](args)
