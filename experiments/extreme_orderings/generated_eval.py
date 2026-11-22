"""Evaluate frozen methods on procedurally generated extreme orderings.

No policy is trained and no configuration is tuned here. Every method was
selected earlier on development data; this script only replays them on a
prespecified family of generated orderings (see generated.py and
results/extreme_generated/design.json).

Modes:
  design    write design.json and orderings.csv (prespecification; no evaluation)
  evaluate  one (family, method, policy seed, chunk of orderings)
  launch    run every missing evaluation job with a process pool
  smoke     time a few episodes on smoke-only orderings and seeds (not the suite)
"""

import argparse
import concurrent.futures
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/extreme_orderings"))
import generated as g  # noqa: E402  (patches the stream constructors)
import retrain as r  # noqa: E402
import torch  # noqa: E402

e, f, b = g.e, g.f, g.b
OUT = ROOT / "results/extreme_generated"
FAMILIES = ("SGD", "Adam", "AdamW", "RMSprop")
POLICY_SEEDS = (338, 339, 340, 341, 342)

# ---------------------------------------------------------------- prespecified design
N_ORDERINGS = 200
GENERATOR_SEEDS = tuple(range(8100000, 8100000 + N_ORDERINGS))
EPISODES_PER_ORDERING = 5
EPISODE_SEED_BASE = 1986000
CHUNK = 20  # orderings per evaluation job
BATCH = EPISODES_PER_ORDERING  # one rollout = the five episode seeds of one ordering
SMOKE_GENERATOR_SEEDS = tuple(range(9900000, 9900004))
SMOKE_EPISODE_BASE = 9990000


def ordering_name(i):
    return f"gen{i:03d}"


def episode_seeds(i, base=EPISODE_SEED_BASE):
    return [base + EPISODES_PER_ORDERING * i + j for j in range(EPISODES_PER_ORDERING)]


SPECS = {ordering_name(i): g.sample_spec(s) for i, s in enumerate(GENERATOR_SEEDS)}
for _name, _spec in SPECS.items():
    g.register(_name, _spec)
ORDERINGS = tuple(SPECS)


def used_seed_pools():
    """Every episode/stream seed pool used by earlier experiments (for disjointness)."""
    pools = {
        "regular_dev": set(b.DEVSEEDS),
        "regular_test": set(b.TESTSEEDS),
        "extreme_test": set(r.SEEDS),
        "extreme_dev": set(r.DEVSEEDS),
        "held_out_dev": set(range(1940000, 1940016)),
        "open_loop_and_tuning_dev": set(range(1810000, 1810016)),
    }
    seeds = range(330, 350)
    for name, base in (("regular_train", 5000000), ("extreme_train", 6000000), ("held_out_train", 7000000)):
        pools[name] = {base + s * 10000 + u * 8 + i for s in seeds for u in range(500) for i in range(8)}
    return pools


def check_design():
    test = {s for i in range(N_ORDERINGS) for s in episode_seeds(i)}
    gen = set(GENERATOR_SEEDS)
    smoke = set(SMOKE_GENERATOR_SEEDS) | {SMOKE_EPISODE_BASE + j for j in range(20)}
    assert len(test) == N_ORDERINGS * EPISODES_PER_ORDERING
    for name, pool in used_seed_pools().items():
        assert test.isdisjoint(pool), name
        assert gen.isdisjoint(pool), name
        assert smoke.isdisjoint(pool), name
    assert test.isdisjoint(gen) and smoke.isdisjoint(test | gen)
    # The generator seeds also never collide with the stream RNG seeds (seed + 700000)
    # of any episode used anywhere.
    for pool in list(used_seed_pools().values()) + [test]:
        assert gen.isdisjoint({s + 700000 for s in pool})


# ---------------------------------------------------------------- methods

def extreme_config(family):
    return json.loads((ROOT / f"results/extreme_retrained/{family}_baseline.json").read_text())["config"]


def regular_config(family):
    return json.loads((ROOT / f"results/optimizer_families/{family}_baseline.json").read_text())["config"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prospective():
    sys.path.insert(0, str(ROOT / "experiments/prospective_schedules"))
    return _load(ROOT / "experiments/prospective_schedules/run.py", "prospective_run")


def regular_cosine_schedule(family):
    ps = prospective()
    sel = json.loads((ROOT / "results/prospective_schedules/selected" / f"{family}.json").read_text())
    return ps.make_schedule(family, "cosine", sel["kinds"]["cosine"]["config"])


def frozen_checkpoint(family, seed):
    if seed in (338, 339):
        return (ROOT / "results" / ("recovery_study" if family == "SGD" else "optimizer_families")
                / (f"gradient_{seed}" if family == "SGD" else f"{family}_gradient_{seed}") / "best.pt")
    return ROOT / "results/additional_seeds/training" / f"{family}_gradient_{seed}" / "best.pt"


def cold_checkpoint(family, seed):
    root = r.OUT if seed in r.ORIGINAL_POLICY_SEEDS else r.EXTENDED_OUT
    return root / f"{family}_gradient_{seed}" / "best.pt"


def warm_checkpoint(family, seed):
    return ROOT / "results/extreme_warm_start" / f"{family}_gradient_{seed}" / "best.pt"


# name -> (seeded, description)
METHODS = {
    "fixed_regular": (False, "Fixed configuration tuned on regular development data (results/optimizer_families)"),
    "fixed_extreme": (False, "Fixed configuration retuned on extreme development data (results/extreme_retrained)"),
    "cosine_regular": (False, "Cosine LR schedule tuned on regular development data (results/prospective_schedules)"),
    "policy_frozen": (True, "Regular-trained policies, seeds 338-342 (recovery_study/optimizer_families, additional_seeds)"),
    "policy_retrained": (True, "Cold-start extreme-retrained policies (extreme_retrained 338-339, extreme_retrained_five_seeds 340-342)"),
    "warm_extreme": (True, "Extreme warm-start policies (results/extreme_warm_start), warm from the extreme-retuned fixed config"),
}
# Extra methods from the sibling baseline experiment are registered here once merged
# (see results/extreme_generated/README.md); each entry maps a name to a loader.
EXTRA = {}
BASELINE_METHODS = {
    "cosine_extreme": (-1,),
    "warmup_cosine_extreme": (-1,),  # RMSprop's development-best shape (all families evaluated)
    "random_schedule_cold": (0, 1, 2, 3, 4),
    "random_schedule_warm": (0, 1, 2, 3, 4),
    "untrained_cold": POLICY_SEEDS,
    "untrained_warm": POLICY_SEEDS,
    "random_fixed": (0, 1, 2, 3, 4),
}


def register_baselines():
    """Register the sibling PR's methods (experiments/extreme_baselines/run.py).

    Their module patches the stream constructors to the original extreme suite on
    import, so the generated dispatcher is re-installed afterwards. Their random
    schedule draws use default_rng([238, 7, random_seed, episode_seed, order_index])
    with order_index = position in their ORDERS list; generated ordering i gets
    order_index 8 + i (after the eight original orderings).
    """
    path = ROOT / "experiments/extreme_baselines/run.py"
    if not path.exists() or EXTRA:
        return
    xb = _load(path, "extreme_baselines_run")
    b.stream, b.batches, e.batches = g.stream, g.batches, g.batches
    assert list(xb.ORDERS) == list(r.ORDERS)
    xb.ORDERS = list(r.ORDERS) + list(ORDERINGS)
    xb_out = ROOT / "results/extreme_baselines"

    def make(method):
        def runner(family, seed, testset):
            build, meta = xb.method_spec(family, method, seed, xb_out)

            def run(seeds, orders):
                return xb.rollout(family, list(seeds), list(orders), testset, **build(seeds, orders))[0]
            info = {k: v for k, v in meta.items() if k in ("checkpoint_update", "grid_index")}
            return run, info
        return runner

    for method, seeds in BASELINE_METHODS.items():
        EXTRA[method] = dict(seeds=seeds, runner=make(method))


def method_seeds(method):
    if method in EXTRA:
        return EXTRA[method]["seeds"]
    return POLICY_SEEDS if METHODS[method][0] else (-1,)


def make_runner(family, method, seed, testset):
    """Return run(seeds, orders) -> rows, plus metadata."""
    pol, warm, fixed, schedule, update = None, None, None, None, -1
    if method in EXTRA:
        return EXTRA[method]["runner"](family, seed, testset)
    if method == "fixed_regular":
        fixed = regular_config(family)
    elif method == "fixed_extreme":
        fixed = extreme_config(family)
    elif method == "cosine_regular":
        schedule = regular_cosine_schedule(family)
    else:
        path = dict(policy_frozen=frozen_checkpoint, policy_retrained=cold_checkpoint,
                    warm_extreme=warm_checkpoint)[method](family, seed)
        saved = torch.load(path, weights_only=True)
        update = saved["update"]
        pol = r.make_policy(family, seed)
        pol.load_state_dict(saved["policy"])
        pol.eval()
        if method == "warm_extreme":
            warm = extreme_config(family)
    return policy_runner(family, testset, pol=pol, warm=warm, fixed=fixed, schedule=schedule), dict(checkpoint=update)


def policy_runner(family, testset, pol=None, warm=None, fixed=None, schedule=None):
    def run(seeds, orders):
        if family == "SGD" and fixed is not None:
            # Same implementation as every saved SGD fixed row (torch.optim.SGD).
            return [b.baseline(fixed, s, o, testset)[0] for s, o in zip(seeds, orders)]
        if family == "SGD":
            schedules = None if schedule is None else {o: schedule for o in set(orders)}
            return e.rollout(pol, "gradient", list(seeds), list(orders), evaluation=testset,
                             warm=warm, schedules=schedules)[0]
        return f.rollout(family, "gradient", list(seeds), list(orders), pol=pol, fixed=fixed,
                         evaluation=testset, warm=warm, schedule=schedule)[0]
    return run


def testset():
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    return test.data.reshape(-1, 784).float() / 255, test.targets


def chunks():
    return range((N_ORDERINGS + CHUNK - 1) // CHUNK)


def eval_path(out, family, method, seed, chunk):
    return out / "evaluation" / f"{family}_{method}_{seed}" / f"chunk_{chunk:02d}.csv"


def evaluate(out, family, method, seed, chunk):
    path = eval_path(out, family, method, seed, chunk)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    run, meta = make_runner(family, method, seed, testset())
    rows = []
    start = time.time()
    for i in range(chunk * CHUNK, min(N_ORDERINGS, (chunk + 1) * CHUNK)):
        name = ordering_name(i)
        seeds = episode_seeds(i)
        rr = run(seeds, [name] * len(seeds))
        assert [int(x["seed"]) for x in rr] == seeds
        for x in rr:
            x.update(order=name, family=family, variant=method, policy_seed=seed, threads=1, **meta)
        rows += rr
    tmp = path.with_suffix(".tmp")
    e.writecsv(tmp, rows)
    tmp.replace(path)
    print(json.dumps(dict(family=family, method=method, seed=seed, chunk=chunk,
                          seconds=round(time.time() - start, 1))), flush=True)


def all_jobs(methods):
    jobs = []
    for family in ("Adam", "AdamW", "RMSprop", "SGD"):
        for method in methods:
            for seed in method_seeds(method):
                for chunk in chunks():
                    jobs.append((family, method, seed, chunk))
    return jobs


def launch(out, workers, methods):
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    jobs = [j for j in all_jobs(methods) if not eval_path(out, *j).exists()]
    print(f"{len(jobs)} jobs pending", flush=True)
    log = (logs / "evaluate.log").open("a")

    def job(args):
        family, method, seed, chunk = args
        subprocess.run([sys.executable, __file__, "evaluate", "--family", family, "--method", method,
                        "--seed", str(seed), "--chunk", str(chunk)] + (["--baselines"] if EXTRA else []), cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=env)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(job, j) for j in jobs]):
            fut.result()
    print("LAUNCH COMPLETE", time.strftime("%H:%M:%S"), flush=True)


def design(out):
    check_design()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, spec) in enumerate(SPECS.items()):
        rows.append(dict(order=name, generator_seed=GENERATOR_SEEDS[i],
                         episode_seeds=" ".join(map(str, episode_seeds(i))),
                         **g.spec_dict(spec), **g.severity(spec), **g.nearness(spec)))
    e.writecsv(out / "orderings.csv", rows)
    originals = [dict(order=name, **g.spec_dict(s), **g.severity(s)) for name, s in g.ORIGINAL_SPECS.items()]
    e.writecsv(out / "original_specs.csv", originals)
    (out / "design.json").write_text(json.dumps(dict(
        status="Prespecified before any evaluation on the generated suite",
        n_orderings=N_ORDERINGS,
        generator_seeds=[GENERATOR_SEEDS[0], GENERATOR_SEEDS[-1]],
        generator="experiments/extreme_orderings/generated.py: sample_spec(generator_seed) with numpy default_rng(generator_seed)",
        sampling_distribution=dict(
            k="classes per biased block, uniform on {1,2,3,5}",
            n="number of blocks, uniform on {1..10}",
            mode="for n>1, uniform on {repeat, change, alternate}; n=1 is 'single'. "
                 "repeat: same k digits every block; change: block i uses the next k digits of a cyclic walk "
                 "through the episode's digit permutation; alternate: two disjoint k-digit sets alternate. "
                 "For k=5, change and alternate give identical streams.",
            R="shuffled recovery after the last block: 0 with probability 0.2, else uniform on {1..100}",
            L="block length (batches): uniform on {max(2, ceil(Lmax/2))..Lmax}, Lmax the largest length that keeps "
              "every digit within its 1,280 draws and the blocks (with the minimum gap) inside the 200 - R batches",
            G="gap of mixed batches between consecutive blocks: 0 for n=1; repeat: uniform on {1..Gmax}; "
              "change/alternate: 0 with probability 0.5, else uniform on {1..Gmax}; Gmax=floor((200-R-nL)/(n-1))",
            start="derived: 200 - R - (nL + (n-1)G)",
        ),
        stream_invariants="Same as run.stream: seeded balanced bag of 1,280 draws per digit (12,800 examples), digits "
                          "permuted per episode, block labels spread over each block with np.resize, remaining bag "
                          "shuffled into non-block positions, within-batch shuffle, 200 batches of 64. Same episode "
                          "seed => same multiset, classifier initialization and probe set for every method.",
        episode_seeds=f"{EPISODES_PER_ORDERING} per ordering, distinct across orderings: "
                      f"{EPISODE_SEED_BASE} + {EPISODES_PER_ORDERING}*i + j for ordering i, j=0..4 "
                      f"({EPISODE_SEED_BASE}-{EPISODE_SEED_BASE + EPISODES_PER_ORDERING * N_ORDERINGS - 1})",
        seed_disjointness="Checked against every earlier dev/test/training seed pool (check_design)",
        severity=dict(
            primary="mean_kl: mean over the 200 batches of KL(expected batch class frequencies || uniform), from the "
                    "spec; block batches have k equal classes, non-block batches the expected composition of the "
                    "remaining bag (depleted of block digits)",
            secondary="biased_batches = n*L",
        ),
        nearness=dict(
            distance="mean over batches of total-variation distance between expected class-frequency profiles "
                     "(in permutation-slot space) to each of the seven original specs; nearest original reported",
            near_duplicate="same k, n and mode as an original, and L, start and R each within 10 batches; "
                           "results reported with and without near-duplicates",
            near_duplicate_loose="secondary exclusion: profile distance <= 0.15 to any original",
        ),
        methods={m: d for m, (_, d) in METHODS.items()},
        configs=dict(fixed_regular={fam: regular_config(fam) for fam in FAMILIES},
                     fixed_extreme={fam: extreme_config(fam) for fam in FAMILIES}),
        sibling_methods="Extreme-retuned cosine and random baselines from the PR 'Add extreme-ordering schedule and "
                        "random baselines', evaluated with their own development-selected configurations once merged.",
        evaluation="Official MNIST test set (10,000 images) for final accuracy; one torch thread; one rollout per "
                   "ordering with its five episode seeds; SGD fixed configurations via torch.optim.SGD (b.baseline) "
                   "as in every saved SGD fixed row; failed episodes retained at their recorded accuracy.",
        analysis=dict(
            unit="ordering (resampling unit), then episode seed, then policy seed",
            mean_accuracy="mean over orderings of the per-ordering mean over episode seeds (and policy seeds); "
                          "policy-seed SD and range of per-seed means; failures counted",
            paired_differences=[
                "warm_extreme - {fixed_regular, fixed_extreme, cosine_regular, policy_frozen, policy_retrained}",
                "policy_retrained - {fixed_regular, fixed_extreme, cosine_regular, policy_frozen}",
                "policy_frozen - {fixed_regular, cosine_regular}",
                "fixed_extreme - fixed_regular",
                "once available: cosine_extreme - fixed_extreme; cosine_{regular,extreme} - random; "
                "warm_extreme and policy_retrained - {cosine_extreme, random}",
            ],
            policy_pairing="policy - policy differences are seed-matched (same seed number); policy - unseeded "
                           "comparator broadcasts the comparator",
            bootstrap="two-level (plus policy seeds): each of 10,000 resamples (numpy default_rng(238)) draws "
                      "orderings with replacement, then the 5 episode seeds within each drawn ordering with "
                      "replacement, then (seeded methods) the 5 policy seeds with replacement; 95% percentile CI",
            wins="fraction of orderings whose mean difference (over episode and policy seeds) is > 0; ties reported; "
                 "policy seeds with positive overall mean difference out of 5",
            severity="per-ordering difference vs mean_kl: quintile bins (mean and bootstrap CI per bin, plot) and "
                     "OLS slope with the ordering-level bootstrap CI; also vs biased_batches",
            subsets="all 200 orderings (primary); excluding near-duplicates of the seven originals; excluding the "
                    "loose near set (profile distance <= 0.15)",
            replication="compare signs and magnitudes with the seven-ordering biased means in "
                        "results/extreme_warm_start and results/prospective_schedules; state disagreements",
        ),
    ), indent=2) + "\n")
    print("DESIGN WRITTEN", out)


def smoke():
    """Time one rollout per method kind on smoke-only orderings and seeds."""
    torch.set_num_threads(1)
    names = []
    for i, s in enumerate(SMOKE_GENERATOR_SEEDS):
        names.append(f"smoke{i}")
        g.register(names[-1], g.sample_spec(s))
    ts = testset()
    for family in FAMILIES:
        for method in METHODS:
            seed = method_seeds(method)[0]
            run, _ = make_runner(family, method, seed, ts)
            start = time.time()
            seeds = [SMOKE_EPISODE_BASE + j for j in range(BATCH)]
            rows = run(seeds, [names[0]] * BATCH)
            print(json.dumps(dict(family=family, method=method, episodes=len(rows),
                                  seconds_per_episode=round((time.time() - start) / len(rows), 3),
                                  failed=sum(x["failed"] for x in rows))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["design", "evaluate", "launch", "smoke"])
    p.add_argument("--family", choices=FAMILIES)
    p.add_argument("--method")
    p.add_argument("--methods", nargs="*")
    p.add_argument("--seed", type=int)
    p.add_argument("--chunk", type=int)
    p.add_argument("--workers", type=int, default=28)
    p.add_argument("--baselines", action="store_true", help="register the sibling PR's baseline methods")
    a = p.parse_args()
    if a.baselines:
        register_baselines()
    torch.set_num_threads(1)
    if a.mode == "design":
        design(OUT)
    elif a.mode == "smoke":
        smoke()
    elif a.mode == "evaluate":
        assert a.method in METHODS or a.method in EXTRA, a.method
        assert a.seed in method_seeds(a.method) and a.chunk in chunks()
        evaluate(OUT, a.family, a.method, a.seed, a.chunk)
    else:
        if a.workers > 28:
            p.error("at most 28 workers")
        launch(OUT, a.workers, a.methods or list(METHODS) + list(EXTRA))
