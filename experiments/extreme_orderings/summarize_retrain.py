"""Summarize matched five-seed extreme-ordering evaluations."""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev

import retrain as r

CONTROLS = ("fixed", "fixed_retuned", "policy")
RETRAINED = "policy_retrained"
FROZEN_CACHE = "frozen_added_seeds.csv"
FROZEN_SOURCE = "https://github.com/nick-rui/cs238/tree/4e24b070613967eaf1b8e47d93ac25c20c560116/results/additional_seeds/extreme"
ROW_FIELDS = (
    "family",
    "variant",
    "policy_seed",
    "order",
    "seed",
    "accuracy",
    "failed",
    "checkpoint",
)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def index_rows(rows, family, variants, test_seeds, source):
    expected = {
        (variant, ps, order, seed)
        for variant, policy_seeds in variants.items()
        for ps, order, seed in itertools.product(policy_seeds, r.ORDERS, test_seeds)
    }
    indexed = {}
    for row in rows:
        key = (row["variant"], int(row["policy_seed"]), row["order"], int(row["seed"]))
        accuracy = float(row["accuracy"])
        if row["family"] != family or key not in expected or key in indexed:
            raise ValueError(
                f"{source}: unexpected or duplicate episode {row['family']} {key}"
            )
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"{source}: invalid accuracy for {key}")
        if row["failed"] not in ("True", "False"):
            raise ValueError(f"{source}: invalid failure flag for {key}")
        indexed[key] = (accuracy, row["failed"] == "True")
    if missing := expected - indexed.keys():
        raise ValueError(
            f"{source}: missing {len(missing)} episodes (e.g. {min(missing)})"
        )
    return indexed


def load_frozen(out, added_frozen_dir):
    if added_frozen_dir is None:
        rows = read_csv(out / FROZEN_CACHE)
    else:
        rows = []
        for family, ps in itertools.product(r.FAMILIES, r.ADDED_POLICY_SEEDS):
            path = added_frozen_dir / f"{family}_{ps}" / "evaluation.csv"
            batch = read_csv(path)
            index_rows(batch, family, {"policy": (ps,)}, r.SEEDS, path)
            rows.extend({field: row[field] for field in ROW_FIELDS} for row in batch)
    if unknown := {row["family"] for row in rows} - set(r.FAMILIES):
        raise ValueError(f"Unexpected families in frozen controls: {sorted(unknown)}")
    return rows


def load_episodes(family, out, frozen, test_seeds):
    episodes = {}
    for root, variants, seeds in (
        (r.extreme.OUT, {"fixed": (-1,), "policy": r.ORIGINAL_POLICY_SEEDS}, r.SEEDS),
        (r.OUT, {"fixed_retuned": (-1,), RETRAINED: r.ORIGINAL_POLICY_SEEDS}, r.SEEDS),
        (out, {RETRAINED: r.ADDED_POLICY_SEEDS}, test_seeds),
    ):
        path = root / f"{family}_evaluation.csv"
        episodes.update(index_rows(read_csv(path), family, variants, seeds, path))
    episodes.update(
        index_rows(
            [row for row in frozen if row["family"] == family],
            family,
            {"policy": r.ADDED_POLICY_SEEDS},
            r.SEEDS,
            FROZEN_CACHE,
        )
    )
    return episodes


def spread(values):
    return dict(
        mean=mean(values),
        sd=stdev(values) if len(values) > 1 else None,
        minimum=min(values),
        maximum=max(values),
    )


def win_counts(deltas):
    return dict(
        wins=sum(x > 0 for x in deltas),
        ties=sum(x == 0 for x in deltas),
        losses=sum(x < 0 for x in deltas),
        pairs=len(deltas),
    )


def summarize_family(family, episodes, test_seeds, outputs):
    scopes = [
        ("all", "", r.ORDERS),
        ("biased", "", [order for order in r.ORDERS if order != "mixed"]),
        *(("per_order", order, [order]) for order in r.ORDERS),
    ]
    for scope, order, orders in scopes:
        key = dict(family=family, scope=scope, order=order)
        scores = {}
        for variant in (*CONTROLS, RETRAINED):
            policy_seeds = r.POLICY_SEEDS if variant in ("policy", RETRAINED) else (-1,)
            failures = 0
            averages = []
            for ps in policy_seeds:
                scores[variant, ps] = [
                    mean(
                        episodes[variant, ps, condition, seed][0]
                        for condition in orders
                    )
                    for seed in test_seeds
                ]
                accuracy = mean(scores[variant, ps])
                failed = sum(
                    episodes[variant, ps, condition, seed][1]
                    for condition, seed in itertools.product(orders, test_seeds)
                )
                outputs["policy_seed_scores"].append(
                    dict(
                        key,
                        variant=variant,
                        policy_seed=ps,
                        accuracy=accuracy,
                        failed_episodes=failed,
                        episodes=len(orders) * len(test_seeds),
                    )
                )
                failures += failed
                averages.append(accuracy)
            outputs["summary"].append(
                dict(
                    key,
                    variant=variant,
                    policy_seeds=0 if policy_seeds == (-1,) else len(policy_seeds),
                    **spread(averages),
                    failed_episodes=failures,
                    episodes=len(policy_seeds) * len(orders) * len(test_seeds),
                )
            )

        for control in CONTROLS:
            deltas = []
            for ps in r.POLICY_SEEDS:
                ref = ps if control == "policy" else -1
                paired = [
                    a - b
                    for a, b in zip(
                        scores[RETRAINED, ps], scores[control, ref], strict=True
                    )
                ]
                delta = mean(paired)
                outputs["policy_seed_differences"].append(
                    dict(
                        key,
                        control=control,
                        policy_seed=ps,
                        mean_difference=delta,
                        **win_counts(paired),
                    )
                )
                deltas.append(delta)
            outputs["comparisons"].append(
                dict(
                    key,
                    control=control,
                    **spread(deltas),
                    **win_counts(deltas),
                )
            )


def summarize(out, added_frozen_dir, smoke):
    test_seeds = r.SEEDS[:1] if smoke else r.SEEDS
    frozen = load_frozen(out, added_frozen_dir)
    outputs = {
        name: []
        for name in (
            "policy_seed_scores",
            "summary",
            "policy_seed_differences",
            "comparisons",
        )
    }
    r.assert_disjoint()
    for family in r.FAMILIES:
        summarize_family(
            family, load_episodes(family, out, frozen, test_seeds), test_seeds, outputs
        )
    for name, rows in outputs.items():
        r.e.writecsv(out / f"{name}.csv", rows)
    if added_frozen_dir is not None:
        r.e.writecsv(out / FROZEN_CACHE, frozen)
    design = dict(
        smoke=smoke,
        orders=r.ORDERS,
        policy_seeds=r.POLICY_SEEDS,
        reused_retrain_seeds=r.ORIGINAL_POLICY_SEEDS,
        new_retrain_seeds=r.ADDED_POLICY_SEEDS,
        training_seed_formula="6000000 + policy_seed*10000 + update*8 + episode",
        new_training_updates=2 if smoke else r.UPDATES,
        episodes_per_update=8,
        development_seeds=r.DEVSEEDS,
        development_orders=r.DEVORDERS,
        tuning_seeds=r.DEVSEEDS[:8],
        tuning_orders=r.DEVORDERS[:8],
        test_seeds=test_seeds,
        disjointness_asserted=True,
        saved_fixed_and_original_frozen="results/extreme_orderings",
        retuned_fixed_and_original_retrain="results/extreme_retrained",
        additional_frozen_source=FROZEN_SOURCE,
        aggregation="Average orderings within test seed, then test seeds within policy seed.",
        paired_wins="Strictly positive mean delta over shared test seeds and orderings; match frozen policies by policy seed.",
        spread="Sample SD and min/max across policy seeds; no SD for fixed controls.",
    )
    (out / "design.json").write_text(json.dumps(design, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-results",
        type=Path,
        help="Directory containing the archived additional frozen evaluation directories",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summarize(
        r.EXTENDED_OUT / "smoke" if args.smoke else r.EXTENDED_OUT,
        args.frozen_results,
        args.smoke,
    )
