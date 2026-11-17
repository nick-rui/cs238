"""Retrain on extreme orderings without changing the frozen experiment."""
import argparse
import csv
import gzip
import itertools
import json
import math
import time
from collections import defaultdict
import numpy as np
import torch
import run as extreme

f, b, e = extreme.f, extreme.b, extreme.e
ORDERS, SEEDS = extreme.ORDERS, extreme.SEEDS
OUT = e.ROOT / "results/extreme_retrained"
DEVSEEDS = list(range(1930000, 1930016))
DEVORDERS = ORDERS * 2
FAMILIES = ["SGD", "Adam", "AdamW", "RMSprop"]
UPDATES = 500


def train_seed(seed, update, i):
    return 6000000 + seed * 10000 + update * 8 + i


def assert_disjoint():
    groups = [{train_seed(s, u, i) for u in range(500) for i in range(8)}
              for s in [338, 339]]
    groups += [set(DEVSEEDS), set(SEEDS), set(b.TESTSEEDS), set(b.DEVSEEDS)]
    groups += [{5000000 + s * 10000 + u * 8 + i
                for s in [338, 339] for u in range(500) for i in range(8)}]
    assert all(not a & c for i, a in enumerate(groups) for c in groups[i + 1:])
    print("SEED DISJOINTNESS PASSED", flush=True)


def training_orders(update):
    return ORDERS[4 * (update % 2):4 * (update % 2 + 1)] * 2


def make_policy(family, seed):
    return e.policy("gradient", seed) if family == "SGD" else f.policy(family, "gradient", seed)


def rollout(family, kind, seeds, orders, pol=None, **kwargs):
    if family == "SGD":
        return e.rollout(pol, kind, seeds, orders, **kwargs)[:4]
    return f.rollout(family, kind, seeds, orders, pol, **kwargs)


def writegzip(path, rows):
    with gzip.open(path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def train(family, seed):
    kind = 'gradient'
    folder = OUT / f"{family}_{kind}_{seed}"
    folder.mkdir(exist_ok=True)
    if (folder / "done.json").exists():
        return
    torch.manual_seed(seed)
    pol = make_policy(family, seed)
    opt = torch.optim.Adam(pol.parameters(), lr=0.003)
    training = []
    dev = []
    best = math.inf
    start = time.perf_counter()
    for update in range(UPDATES + 1):
        if update in [0, 25, 50, 100, 200, 300, 400, 500] or update == UPDATES:
            rr = []
            for st in range(0, 16, 8):
                rr += rollout(
                    family,
                    kind,
                    DEVSEEDS[st : st + 8],
                    DEVORDERS[st : st + 8],
                    pol,
                    evaluation=(e.xd, e.yd),
                )[0]
            value = float(np.mean([b.score(r) for r in rr]))
            for r in rr:
                r.update(update=update, family=family, kind=kind, policy_seed=seed)
            dev += rr
            state = dict(
                policy=pol.state_dict(),
                update=update,
                family=family,
                kind=kind,
                seed=seed,
            )
            torch.save(state, folder / f"checkpoint_{update}.pt")
            if value < best:
                best = value
                torch.save(state, folder / "best.pt")
            e.writecsv(folder / "development.csv", dev)
            print(
                json.dumps(
                    dict(
                        job=folder.name,
                        update=update,
                        score=value,
                        accuracy=float(np.mean([r["accuracy"] for r in rr])),
                        seconds=time.perf_counter() - start,
                    )
                ),
                flush=True,
            )
        if update == UPDATES:
            break
        seeds = [train_seed(seed, update, i) for i in range(8)]
        rr, _, logp, rewards = rollout(
            family, kind, seeds, training_orders(update), pol, training=True
        )
        returns = rewards.flip(1).cumsum(1).flip(1)
        adv = returns - returns[torch.tensor([4, 5, 6, 7, 0, 1, 2, 3])]
        opt.zero_grad()
        (-(logp * adv.detach()).mean()).backward()
        assert all(torch.isfinite(q.grad).all() for q in pol.parameters())
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1)
        opt.step()
        for r in rr:
            training.append(
                dict(
                    update=update + 1,
                    seed=r["seed"],
                    order=r["order"],
                    reward=r["return_value"],
                    failed=r["failed"],
                )
            )
    e.writecsv(folder / "training.csv", training)
    (folder / "done.json").write_text(
        json.dumps(
            dict(
                updates=UPDATES,
                episodes=UPDATES * 8,
                threads=torch.get_num_threads(),
                seconds=time.perf_counter() - start,
                failed=sum(r["failed"] for r in training),
            )
        )
    )

def family_configs(family):
    result = []
    for lr, mom, decay, wd in itertools.product(
        [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03],
        [0.0, 0.5, 0.9, 0.95],
        [0.9, 0.99, 0.999, 0.9999],
        [0.0, 1e-4, 0.01],
    ):
        r = dict(family=family, lr=lr, weight_decay=wd)
        if family == "RMSprop":
            r.update(momentum=mom, alpha=decay)
        else:
            r.update(beta1=mom, beta2=decay)
        result.append(r)
    return result


def configs(family):
    if family != "SGD":
        return family_configs(family)
    return [dict(family=family, lr=lr, momentum=mu, weight_decay=wd)
            for lr, mu, wd in itertools.product(
                [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                [0.0, 0.25, 0.5, 0.75, 0.9, 0.95],
                [0.0, 1e-6, 1e-4, 0.01])]

def tune(family, smoke=False):
    path = OUT / f"{family}_baseline.json"
    if path.exists():
        return
    candidates = configs(family)
    if smoke:
        candidates = candidates[:1]
    records = []
    scores = []
    for ci, c in enumerate(candidates):
        if family == "SGD":
            rr = [b.baseline(c, seed, order, (e.xd, e.yd))[0]
                  for seed, order in zip(DEVSEEDS[:8], DEVORDERS[:8])]
        else:
            rr = f.rollout(family, "gradient", DEVSEEDS[:8], DEVORDERS[:8],
                           fixed=c, evaluation=(e.xd, e.yd))[0]
        value = float(np.mean([b.score(r) for r in rr]))
        scores.append(value)
        for r in rr:
            r.update(family=family, config_id=ci)
        records += rr
        if ci % 24 == 0:
            print("TUNE", family, ci, "of", len(candidates), flush=True)
    index = int(np.argmin(scores))
    path.write_text(
        json.dumps(
            dict(
                config=candidates[index],
                score=scores[index],
                candidates=len(candidates),
                episodes_per_candidate=8,
            ),
            indent=2,
        )
    )
    e.writecsv(OUT / f"{family}_tuning.csv", records)
    print("SELECTED", family, candidates[index], scores[index], flush=True)

def evaluate(family):
    path = OUT / f"{family}_evaluation.csv"
    if path.exists():
        print("Already completed " + family, flush=True)
        return
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    testset = (test.data.reshape(-1, 784).float() / 255, test.targets)
    config = json.loads((OUT / f"{family}_baseline.json").read_text())["config"]
    rows = []
    traces = []
    start = time.time()
    for variant, ps in [("fixed_retuned", -1), ("policy_retrained", 338), ("policy_retrained", 339)]:
        pol = None
        checkpoint = -1
        if variant == "policy_retrained":
            folder = OUT / f"{family}_gradient_{ps}"
            saved = torch.load(folder / "best.pt", weights_only=True)
            checkpoint = saved["update"]
            pol = (
                e.policy("gradient", ps)
                if family == "SGD"
                else f.policy(family, "gradient", ps)
            )
            pol.load_state_dict(saved["policy"])
            pol.eval()
        for order in ORDERS:
            for st in range(0, len(SEEDS), 4):
                seeds = SEEDS[st : st + 4]
                if family == "SGD" and variant == "fixed_retuned":
                    pairs = [
                        b.baseline(config, seed, order, testset, True) for seed in seeds
                    ]
                    rr = [p[0] for p in pairs]
                    tr = [t for p in pairs for t in p[1]]
                elif family == "SGD":
                    rr, tr, *_ = e.rollout(
                        pol,
                        "gradient",
                        seeds,
                        [order] * len(seeds),
                        evaluation=testset,
                        trace=True,
                    )
                else:
                    rr, tr, *_ = f.rollout(
                        family,
                        "gradient",
                        seeds,
                        [order] * len(seeds),
                        pol=pol,
                        fixed=config if variant == "fixed_retuned" else None,
                        evaluation=testset,
                        trace=True,
                    )
                for r in rr:
                    r.update(
                        family=family,
                        variant=variant,
                        policy_seed=ps,
                        checkpoint=checkpoint,
                    )
                for r in tr:
                    r.update(family=family, variant=variant, policy_seed=ps)
                rows += rr
                traces += tr
            print(
                json.dumps(
                    dict(
                        family=family,
                        variant=variant,
                        policy_seed=ps,
                        order=order,
                        rows=len(rows),
                        seconds=round(time.time() - start, 1),
                    )
                ),
                flush=True,
            )
    e.writecsv(path, rows)
    writegzip(OUT / f"{family}_traces.csv.gz", traces)
    (OUT / f"{family}_done.json").write_text(
        json.dumps(
            dict(
                rows=len(rows),
                traces=len(traces),
                seconds=time.time() - start,
                failures=sum(r["failed"] for r in rows),
            ),
            indent=2,
        )
    )


def analyze():
    data = []
    for family in FAMILIES:
        for root, variants in [(extreme.OUT, {"fixed", "policy"}),
                               (OUT, {"fixed_retuned", "policy_retrained"})]:
            with (root / f"{family}_evaluation.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            expected = {(variant, str(ps), str(seed), order)
                        for variant in variants
                        for ps in ([-1] if variant.startswith("fixed") else [338, 339])
                        for seed in SEEDS for order in ORDERS}
            actual = {(r["variant"], r["policy_seed"], r["seed"], r["order"]) for r in rows}
            assert len(rows) == len(expected) and actual == expected
            for row in rows:
                row["accuracy"] = float(row["accuracy"])
                row["failed"] = row["failed"] == "True"
            data += rows
    fixed = {(r["family"], r["order"], r["seed"]): r["accuracy"]
             for r in data if r["variant"] == "fixed"}
    summaries, differences, per_order = [], [], []
    for scope in ["per_order", "all", "biased"]:
        groups = defaultdict(list)
        for row in data:
            if scope == "biased" and row["order"] == "mixed":
                continue
            key = (row["family"], row["variant"], row["policy_seed"],
                   row["order"] if scope == "per_order" else "")
            groups[key].append(row)
        for (family, variant, ps, order), rows in sorted(groups.items()):
            record = dict(family=family, order=order, variant=variant,
                          policy_seed=ps, scope=scope)
            summary = dict(record, accuracy=float(np.mean([r["accuracy"] for r in rows])),
                           failures=sum(r["failed"] for r in rows), episodes=len(rows))
            summaries.append(summary)
            if scope == "per_order":
                per_order.append(summary)
            by_seed = defaultdict(list)
            for r in rows:
                by_seed[r["seed"]].append(
                    r["accuracy"] - fixed[(family, r["order"], r["seed"])])
            delta = [float(np.mean(values)) for values in by_seed.values()]
            assert len(delta) == 20
            differences.append(dict(record, mean_difference=float(np.mean(delta)),
                                    std_err=float(np.std(delta, ddof=1) / math.sqrt(20)),
                                    wins=sum(d > 0 for d in delta), pairs=20))
    e.writecsv(OUT / "per_order.csv", per_order)
    e.writecsv(OUT / "summary.csv", summaries)
    e.writecsv(OUT / "paired_differences.csv", differences)
    for row in summaries:
        if row["scope"] != "per_order":
            print(json.dumps(row), flush=True)


def design(smoke=False):
    (OUT / "design.json").write_text(json.dumps(dict(
        smoke=smoke, orders=ORDERS, policy_seeds=[338, 339],
        training_seed_formula="6000000 + policy_seed*10000 + update*8 + episode",
        training_updates=UPDATES, episodes_per_update=8,
        training_orders="Alternate first/last four ORDERS, repeated twice; i pairs with i+4",
        development_seeds=DEVSEEDS, development_orders=DEVORDERS,
        tuning_seeds=DEVSEEDS[:8], tuning_orders=DEVORDERS[:8],
        test_seeds=SEEDS, disjointness_asserted=True,
        selection="b.score on development split e.xd/e.yd; minimum mean dev score",
        splits="Original MNIST train/control/dev splits, official 10000-image test set",
        test_diagnostics="First 1000 test images for trajectories only; never feedback",
        limits=[
            "Train/test share ordering types: in-distribution for orderings, disjoint episode seeds and seeded digit permutations (permutations may coincide by chance).",
            "Saved regular-tuned fixed baseline is disadvantaged; extreme-dev retuned fixed is an extra fairness row.",
            "Two policy seeds, one network, 200 classifier updates, batch size 64.",
            "Test images used in earlier exploratory work; not a new test dataset.",
            "Retained SGD and other-family failure handling differ.",
            "Aggregate paired differences average orderings within seed before standard error and wins/20.",
        ]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "tune", "evaluate", "analyze"])
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--seed", type=int, choices=[338, 339])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.mode != "analyze" and not args.family:
        parser.error("--family is required")
    if args.mode == "train" and args.seed is None:
        parser.error("--seed is required")
    assert_disjoint()
    if args.smoke:
        OUT = OUT / "smoke"
        UPDATES = 2
        SEEDS = SEEDS[:1]
    OUT.mkdir(parents=True, exist_ok=True)
    design(args.smoke)
    if args.mode == "train":
        train(args.family, args.seed)
    elif args.mode == "tune":
        tune(args.family, args.smoke)
    elif args.mode == "evaluate":
        evaluate(args.family)
    else:
        analyze()
