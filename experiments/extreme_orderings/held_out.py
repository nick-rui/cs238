"""Leave one extreme-ordering type out of training and model selection."""

import argparse
import csv
import itertools
import json
import math
import statistics
import time

import run as extreme
import torch

f, b, e = extreme.f, extreme.b, extreme.e
ORDERS = tuple(extreme.ORDERS)
FAMILIES = ("SGD", *f.FAMILIES)
POLICY_SEEDS = (338, 339)
DEV_SEEDS = tuple(range(1940000, 1940016))
UPDATES = 500
BATCH = 8
OUT = e.ROOT / "results/extreme_held_out"
DEV = tuple(zip(DEV_SEEDS, ORDERS * 2))
KIND = "gradient"


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def fold_orders(held_out):
    return tuple(order for order in ORDERS if order != held_out)


def training_orders(held_out, update):
    remaining = fold_orders(held_out)
    # Each seven-update cycle presents each seen order four matched pairs.
    return [
        remaining[(update + offset) % len(remaining)] for offset in (0, 1, 3, 5)
    ] * 2


def train_seeds(policy_seed, update):
    first = 7000000 + policy_seed * 10000 + update * BATCH
    return list(range(first, first + BATCH))


def check_design():
    train_pools = [
        {s for u in range(UPDATES) for s in train_seeds(ps, u)} for ps in POLICY_SEEDS
    ]
    pools = train_pools + [
        set(DEV_SEEDS),
        set(extreme.SEEDS),
        set(b.DEVSEEDS),
        set(b.TESTSEEDS),
    ]
    assert all(a.isdisjoint(c) for a, c in itertools.combinations(pools, 2))
    for held_out in ORDERS:
        seen = set(fold_orders(held_out))
        assert {order for _, order in DEV if order != held_out} == seen
        cycle = [
            order for u in range(len(seen)) for order in training_orders(held_out, u)
        ]
        assert set(cycle) == seen and all(cycle.count(order) == BATCH for order in seen)
        for update in range(UPDATES):
            orders = training_orders(held_out, update)
            assert (
                len(orders) == BATCH
                and held_out not in orders
                and orders[:4] == orders[4:]
            )


def policy(family, seed):
    return e.policy(KIND, seed) if family == "SGD" else f.policy(family, KIND, seed)


def rollout(family, seeds, orders, pol=None, *, fixed=None, **kwargs):
    if family == "SGD":
        if fixed is not None:
            return (
                [
                    b.baseline(fixed, s, o, kwargs["evaluation"])[0]
                    for s, o in zip(seeds, orders, strict=True)
                ],
                [],
                None,
                None,
            )
        return e.rollout(pol, KIND, seeds, orders, **kwargs)[:4]
    return f.rollout(family, KIND, seeds, orders, pol, fixed=fixed, **kwargs)


def evaluate_episodes(family, episodes, evaluation, pol=None, fixed=None, batch=BATCH):
    rows = []
    for start in range(0, len(episodes), batch):
        seeds, orders = zip(*episodes[start : start + batch], strict=True)
        rows.extend(
            rollout(family, seeds, orders, pol, fixed=fixed, evaluation=evaluation)[0]
        )
    return rows


def fold_dir(root, family, held_out):
    return root / family / held_out


def checkpoint_updates(updates):
    return sorted(
        {u for u in (0, 25, 50, 100, 200, 300, 400, 500) if u <= updates} | {updates}
    )


def train(root, family, held_out, seed, smoke=False):
    folder = fold_dir(root, family, held_out) / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "training_summary.json").exists():
        return
    updates = 2 if smoke else UPDATES
    dev_episodes = [(s, o) for s, o in DEV if o != held_out]
    torch.manual_seed(seed)
    pol = policy(family, seed)
    opt = torch.optim.Adam(pol.parameters(), lr=0.003)
    development, training = [], []
    best, best_update = math.inf, None
    start = time.perf_counter()
    checkpoints = checkpoint_updates(updates)
    for update in range(updates + 1):
        if update in checkpoints:
            rows = evaluate_episodes(family, dev_episodes, (e.xd, e.yd), pol)
            value = statistics.mean(b.score(r) for r in rows)
            development.extend(dict(r, update=update) for r in rows)
            if value < best:
                best, best_update = value, update
                torch.save(
                    dict(
                        policy=pol.state_dict(),
                        update=update,
                        family=family,
                        held_out=held_out,
                        seed=seed,
                    ),
                    folder / "best.pt",
                )
            print(
                f"{family} {held_out} {seed} update={update} score={value:.6f} "
                f"seconds={time.perf_counter() - start:.1f}",
                flush=True,
            )
        if update == updates:
            break
        seeds = train_seeds(seed, update)
        rows, _, logp, rewards = rollout(
            family, seeds, training_orders(held_out, update), pol, training=True
        )
        returns = rewards.flip(1).cumsum(1).flip(1)
        adv = returns - returns.roll(BATCH // 2, 0)
        opt.zero_grad()
        (-(logp * adv.detach()).mean()).backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        training.extend(
            dict(
                update=update + 1,
                seed=r["seed"],
                order=r["order"],
                reward=r["return_value"],
                failed=r["failed"],
            )
            for r in rows
        )
    if best_update is None:
        raise RuntimeError(f"No valid checkpoint: {family} {held_out} {seed}")
    e.writecsv(folder / "development.csv", development)
    e.writecsv(folder / "training.csv", training)
    write_json(
        folder / "training_summary.json",
        dict(
            updates=updates,
            episodes=len(training),
            best_update=best_update,
            best_dev_score=best,
            failed=sum(r["failed"] for r in training),
            seconds=time.perf_counter() - start,
            training_orders={
                o: sum(row["order"] == o for row in training)
                for o in fold_orders(held_out)
            },
        ),
    )


def frozen_fixed(family):
    return json.loads((extreme.FAMILY_ROOT / f"{family}_baseline.json").read_text())[
        "config"
    ]


def configs(family):
    if family == "SGD":
        for lr, momentum, wd in itertools.product(
            (0.001, 0.003, 0.01, 0.03, 0.1, 0.3),
            (0.0, 0.25, 0.5, 0.75, 0.9, 0.95),
            (0.0, 1e-6, 1e-4, 0.01),
        ):
            yield dict(family=family, lr=lr, momentum=momentum, weight_decay=wd)
        return
    for lr, first, second, wd in itertools.product(
        (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03),
        (0.0, 0.5, 0.9, 0.95),
        (0.9, 0.99, 0.999, 0.9999),
        (0.0, 1e-4, 0.01),
    ):
        memory = (
            dict(momentum=first, alpha=second)
            if family == "RMSprop"
            else dict(beta1=first, beta2=second)
        )
        yield dict(family=family, lr=lr, weight_decay=wd, **memory)


def tune(root, family, smoke=False):
    folds = {o: fold_dir(root, family, o) for o in ORDERS}
    if all((folder / "fixed_retuned.json").exists() for folder in folds.values()):
        return
    candidates = list(configs(family))
    assert frozen_fixed(family) in candidates
    if smoke:
        candidates = [candidates[0], frozen_fixed(family)]
    records = {order: [] for order in ORDERS}
    for index, config in enumerate(candidates):
        # Independent development rollouts shared across folds; exclude the
        # held-out order before computing any fold's selection score.
        rows = evaluate_episodes(family, DEV, (e.xd, e.yd), fixed=config)
        assert [(r["seed"], r["order"]) for r in rows] == list(DEV)
        for held_out in ORDERS:
            seen_rows = [r for r in rows if r["order"] != held_out]
            records[held_out].append(
                dict(
                    candidate=index,
                    **config,
                    score=statistics.mean(b.score(r) for r in seen_rows),
                    failures=sum(r["failed"] for r in seen_rows),
                    episodes=len(seen_rows),
                )
            )
        if index % 24 == 0:
            print(f"{family} candidate {index + 1}/{len(candidates)}", flush=True)
    for held_out, folder in folds.items():
        folder.mkdir(parents=True, exist_ok=True)
        rows = records[held_out]
        best = min(rows, key=lambda row: row["score"])
        e.writecsv(folder / "fixed_tuning.csv", rows)
        write_json(
            folder / "fixed_retuned.json",
            dict(
                config=candidates[best["candidate"]],
                candidate=best["candidate"],
                dev_score=best["score"],
                candidates=len(rows),
                development_episodes=[(s, o) for s, o in DEV if o != held_out],
            ),
        )


VARIANTS = {
    "fixed_frozen": (-1,),
    "fixed_retuned": (-1,),
    "policy_frozen": POLICY_SEEDS,
    "policy_retrained": POLICY_SEEDS,
}


def load_policy(path, family, seed, held_out=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert (state.get("family", "SGD"), state["seed"], state.get("held_out")) == (
        family,
        seed,
        held_out,
    )
    pol = policy(family, seed)
    pol.load_state_dict(state["policy"])
    return pol.eval(), state["update"]


def frozen_checkpoint(family, seed):
    if family == "SGD":
        return OUT.parent / "recovery_study" / f"gradient_{seed}" / "best.pt"
    return extreme.FAMILY_ROOT / f"{family}_gradient_{seed}" / "best.pt"


def test_seeds(smoke):
    return extreme.SEEDS[:2] if smoke else extreme.SEEDS


def evaluate(root, family, held_out, smoke=False):
    folder = fold_dir(root, family, held_out)
    path = folder / "evaluation.csv"
    if path.exists():
        return
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    testset = (test.data.reshape(-1, 784).float() / 255, test.targets)
    episodes = [(s, held_out) for s in test_seeds(smoke)]
    data = []
    for variant, seeds in VARIANTS.items():
        for seed in seeds:
            pol, fixed, update = None, None, -1
            if variant == "fixed_frozen":
                fixed = frozen_fixed(family)
            elif variant == "fixed_retuned":
                fixed = json.loads((folder / "fixed_retuned.json").read_text())[
                    "config"
                ]
            elif variant == "policy_frozen":
                pol, update = load_policy(frozen_checkpoint(family, seed), family, seed)
            else:
                pol, update = load_policy(
                    folder / str(seed) / "best.pt", family, seed, held_out
                )
            data.extend(
                dict(
                    row,
                    family=family,
                    variant=variant,
                    policy_seed=seed,
                    checkpoint=update,
                )
                for row in evaluate_episodes(
                    family, episodes, testset, pol, fixed, batch=4
                )
            )
    temp = path.with_suffix(".tmp")
    e.writecsv(temp, data)
    temp.replace(path)
    print(f"{family} {held_out}: {len(data)} evaluated episodes", flush=True)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_fold(root, family, held_out, smoke):
    folder = fold_dir(root, family, held_out)
    expected_dev = {(s, o) for s, o in DEV if o != held_out}
    fixed = json.loads((folder / "fixed_retuned.json").read_text())
    assert list(map(tuple, fixed["development_episodes"])) == [
        (s, o) for s, o in DEV if o != held_out
    ]
    tuning = read_rows(folder / "fixed_tuning.csv")
    assert len(tuning) == fixed["candidates"]
    assert all(int(row["episodes"]) == len(expected_dev) for row in tuning)
    assert (
        int(min(tuning, key=lambda row: float(row["score"]))["candidate"])
        == fixed["candidate"]
    )
    chosen_updates = {}
    for seed in POLICY_SEEDS:
        policy_folder = folder / str(seed)
        training = json.loads((policy_folder / "training_summary.json").read_text())
        updates = 2 if smoke else UPDATES
        assert (
            training["updates"] == updates and training["episodes"] == updates * BATCH
        )
        planned = [
            order for u in range(updates) for order in training_orders(held_out, u)
        ]
        assert training["training_orders"] == {
            o: planned.count(o) for o in fold_orders(held_out)
        }
        development = read_rows(policy_folder / "development.csv")
        scores = {}
        for update, rows in itertools.groupby(
            development, key=lambda row: int(row["update"])
        ):
            rows = list(rows)
            actual = [(int(r["seed"]), r["order"]) for r in rows]
            assert len(actual) == len(expected_dev) and set(actual) == expected_dev
            scores[update] = statistics.mean(
                b.score(
                    dict(
                        eval_loss=float(r["eval_loss"]),
                        path_log_loss=float(r["path_log_loss"]),
                    )
                )
                for r in rows
            )
        assert list(scores) == checkpoint_updates(updates)
        chosen_updates[seed] = min(scores, key=scores.__getitem__)
        assert chosen_updates[seed] == training["best_update"]
        assert scores[chosen_updates[seed]] == training["best_dev_score"]

    expected = {
        (v, ps, seed)
        for v, seeds in VARIANTS.items()
        for ps in seeds
        for seed in test_seeds(smoke)
    }
    rows = read_rows(folder / "evaluation.csv")
    actual = {(r["variant"], int(r["policy_seed"]), int(r["seed"])) for r in rows}
    assert len(rows) == len(expected) and actual == expected
    for row in rows:
        assert (row["family"], row["order"]) == (family, held_out)
        row.update(
            seed=int(row["seed"]),
            policy_seed=int(row["policy_seed"]),
            accuracy=float(row["accuracy"]),
            failed=row["failed"] == "True",
        )
        assert 0 <= row["accuracy"] <= 1
        if row["variant"] == "policy_retrained":
            assert int(row["checkpoint"]) == chosen_updates[row["policy_seed"]]
    return rows


def analyze(root, smoke=False):
    summary, paired, seed_summary = [], [], []
    seeds = test_seeds(smoke)
    for family in FAMILIES:
        by_order = {order: read_fold(root, family, order, smoke) for order in ORDERS}
        scopes = {
            **{o: (o,) for o in ORDERS},
            "all": ORDERS,
            "biased": tuple(o for o in ORDERS if o != "mixed"),
        }
        for scope, orders in scopes.items():
            mean_accuracy = {}
            for variant, policy_seeds in VARIANTS.items():
                rows = [
                    r for o in orders for r in by_order[o] if r["variant"] == variant
                ]
                for ps in policy_seeds:
                    selected = [r for r in rows if r["policy_seed"] == ps]
                    seed_summary.append(
                        dict(
                            family=family,
                            scope=scope,
                            variant=variant,
                            policy_seed=ps,
                            accuracy=statistics.mean(r["accuracy"] for r in selected),
                            failures=sum(r["failed"] for r in selected),
                            episodes=len(selected),
                        )
                    )
                mean_accuracy[variant] = {
                    s: statistics.mean(r["accuracy"] for r in rows if r["seed"] == s)
                    for s in seeds
                }
                summary.append(
                    dict(
                        family=family,
                        scope=scope,
                        variant=variant,
                        accuracy=statistics.mean(mean_accuracy[variant].values()),
                        failures=sum(r["failed"] for r in rows),
                        episodes=len(rows),
                    )
                )
            comparisons = [
                ("policy_retrained", "policy_frozen"),
                ("policy_retrained", "fixed_frozen"),
                ("policy_retrained", "fixed_retuned"),
                ("fixed_retuned", "fixed_frozen"),
            ]
            for variant, comparator in comparisons:
                diffs = [
                    mean_accuracy[variant][s] - mean_accuracy[comparator][s]
                    for s in seeds
                ]
                paired.append(
                    dict(
                        family=family,
                        scope=scope,
                        variant=variant,
                        comparator=comparator,
                        mean_difference=statistics.mean(diffs),
                        std_err=statistics.stdev(diffs) / math.sqrt(len(diffs)),
                        wins=sum(d > 0 for d in diffs),
                        ties=sum(d == 0 for d in diffs),
                        pairs=len(diffs),
                    )
                )
    e.writecsv(root / "summary.csv", summary)
    e.writecsv(root / "policy_seed_summary.csv", seed_summary)
    e.writecsv(root / "paired_differences.csv", paired)
    write_json(
        root / "design.json",
        dict(
            smoke=smoke,
            orderings=extreme.DESCRIPTIONS,
            families=FAMILIES,
            policy_seeds=POLICY_SEEDS,
            updates=2 if smoke else UPDATES,
            episodes_per_update=BATCH,
            training_seed_formula="7000000 + policy_seed * 10000 + update * 8 + episode",
            training_schedule="Cyclic offsets (0,1,3,5) through the seven seen orderings, twice per update",
            development_episodes=DEV,
            test_seeds=seeds,
            torch_version=str(torch.__version__),
            torch_threads=torch.get_num_threads(),
            selection="Minimize mean log(final dev loss) + mean log(dev trajectory loss) on seen orders only",
            paired_unit="Test episode seed, averaged over orders in scope and both policy seeds before differencing",
            evaluation="Fresh paired replay of all variants, using original MNIST official test split",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "tune", "evaluate", "analyze"))
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--held-out", choices=ORDERS)
    parser.add_argument("--seed", type=int, choices=POLICY_SEEDS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.mode != "analyze" and args.family is None:
        parser.error("--family is required")
    if args.mode in ("train", "evaluate") and args.held_out is None:
        parser.error("--held-out is required")
    if args.mode == "train" and args.seed is None:
        parser.error("--seed is required")
    check_design()
    root = OUT / "smoke" if args.smoke else OUT
    if args.mode == "train":
        train(root, args.family, args.held_out, args.seed, args.smoke)
    elif args.mode == "tune":
        tune(root, args.family, args.smoke)
    elif args.mode == "evaluate":
        evaluate(root, args.family, args.held_out, args.smoke)
    else:
        analyze(root, args.smoke)


if __name__ == "__main__":
    main()
