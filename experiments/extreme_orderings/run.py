"""Frozen-policy, frozen-baseline stress test; only training-example order changes."""

import argparse, csv, functools, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/extreme_orderings"
OUT.mkdir(parents=True, exist_ok=True)
FAMILY_ROOT = ROOT / "results/optimizer_families"
sys.path.insert(0, str(ROOT / "experiments/optimizer_families"))
import family_engine as f

b = f.base
e = f.e
torch.set_num_threads(1)
ORDERS = [
    "mixed",
    "single_max",
    "single_late",
    "pair_max",
    "half_block",
    "staircase",
    "class_sorted",
    "alternating_halves",
]
SEEDS = list(range(1920000, 1920020))
DESCRIPTIONS = {
    "mixed": "Fully shuffled balanced bag (reference).",
    "single_max": "One digit for 20 batches at [80,100); 100 mixed recovery batches.",
    "single_late": "One digit for 20 batches at [175,195); only 5 mixed recovery batches.",
    "pair_max": "Two digits for 40 batches at [140,180); 20 mixed recovery batches.",
    "half_block": "Five digits for 100 batches at [60,160); 40 mixed recovery batches.",
    "staircase": "Ten single-digit blocks of 16 batches at [20,180); 20 mixed batches before and after.",
    "class_sorted": "Ten single-digit blocks of 20 batches covering [0,200); no mixed recovery.",
    "alternating_halves": "Eight 25-batch blocks alternating two disjoint five-digit sets; no fully mixed segment.",
}
ORIGINAL_STREAM = b.stream


@functools.lru_cache(maxsize=256)
def stream(seed, order):
    rng = np.random.default_rng(seed + 700000)
    bags = [rng.choice(q, 1280, replace=True) for q in b.class_ids]
    digits = rng.permutation(10).tolist()
    events = []

    def event(start, length, targets):
        events.append(
            dict(start=start, end=start + length, digits=targets, profile="pure")
        )

    if order == "single_max":
        event(80, 20, digits[:1])
    elif order == "single_late":
        event(175, 20, digits[:1])
    elif order == "pair_max":
        event(140, 40, digits[:2])
    elif order == "half_block":
        event(60, 100, digits[:5])
    elif order == "staircase":
        for i, d in enumerate(digits):
            event(20 + 16 * i, 16, [d])
    elif order == "class_sorted":
        for i, d in enumerate(digits):
            event(20 * i, 20, [d])
    elif order == "alternating_halves":
        for i in range(8):
            event(25 * i, 25, digits[:5] if i % 2 == 0 else digits[5:])
    elif order != "mixed":
        raise ValueError(order)
    ids = np.full((200, 64), -1, dtype=np.int64)
    mask = np.zeros(200, dtype=bool)
    positions = [0] * 10
    for ev in events:
        start, end = ev["start"], ev["end"]
        mask[start:end] = True
        # Distribute targets across the entire block, avoiding per-batch rounding bias.
        labels = np.resize(np.array(ev["digits"]), (end - start) * 64)
        block = ids[start:end].reshape(-1)
        for d in ev["digits"]:
            where = np.flatnonzero(labels == d)
            lo = positions[d]
            hi = lo + len(where)
            assert hi <= len(bags[d]), (order, d, hi)
            block[where] = bags[d][lo:hi]
            positions[d] = hi
    remaining = np.concatenate([q[pos:] for q, pos in zip(bags, positions)])
    rng.shuffle(remaining)
    ids[ids < 0] = remaining
    for row in ids:
        rng.shuffle(row)
    return torch.from_numpy(ids), events, mask


def batches(seed, order):
    return stream(int(seed), order)[0]


b.stream = stream
b.batches = batches
e.batches = batches


def check():
    for seed in SEEDS:
        ref = ORIGINAL_STREAM(seed, "mixed")[0].flatten().sort().values
        for order in ORDERS:
            ids, events, _ = stream(seed, order)
            assert ids.shape == (200, 64) and torch.equal(
                ids.flatten().sort().values, ref
            )
            assert torch.equal(
                torch.bincount(e.y[ids.flatten()]), torch.full((10,), 1280)
            )
            for ev in events:
                assert torch.isin(
                    e.y[ids[ev["start"] : ev["end"]]], torch.tensor(ev["digits"])
                ).all()
    assert not set(SEEDS) & set(b.TESTSEEDS + b.DEVSEEDS)
    checks = dict(
        orders=ORDERS,
        seeds=SEEDS,
        exact_same_multiset=True,
        exact_class_balance=True,
        block_membership=True,
        new_episode_seeds=True,
    )
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2))
    print("STREAM CHECKS PASSED", flush=True)


def evaluate(family):
    path = OUT / f"{family}_evaluation.csv"
    if path.exists():
        raise SystemExit("Already completed " + family)
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    testset = (test.data.reshape(-1, 784).float() / 255, test.targets)
    config = json.loads((FAMILY_ROOT / f"{family}_baseline.json").read_text())["config"]
    rows = []
    traces = []
    start = time.time()
    for variant, ps in [("fixed", -1), ("policy", 338), ("policy", 339)]:
        pol = None
        checkpoint = -1
        if variant == "policy":
            folder = (
                OUT.parent / "recovery_study" / f"gradient_{ps}"
                if family == "SGD"
                else FAMILY_ROOT / f"{family}_gradient_{ps}"
            )
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
                if family == "SGD" and variant == "fixed":
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
                        fixed=config if variant == "fixed" else None,
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
    e.writecsv(OUT / f"{family}_traces.csv", traces)
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


def design():
    (OUT / "design.json").write_text(
        json.dumps(
            dict(
                orderings=DESCRIPTIONS,
                episode_seeds=SEEDS,
                policy_seeds=[338, 339],
                training="None: frozen previous development-selected checkpoints",
                baselines="Frozen previous development-selected configurations; not retuned on stress tests",
                matched="Same balanced 12800-example multiset, classifier initialization,200 updates and64 batch size per paired episode",
                test="10000 official test images final; first1000 for diagnostic trajectories; never feedback",
                limits=[
                    "Exploratory stronger distribution shifts, not optimized adversarial attacks.",
                    "Some conditions deliberately have no fully mixed recovery; report separately from recovery conditions.",
                    "Two policy-training seeds. New episode seeds, same MNIST test images.",
                    "Failure handling differs in retained SGD implementation; report failures.",
                    "Fixed baselines were tuned on the prior training mixture, not this harder test distribution.",
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "evaluate"])
    p.add_argument("--family", choices=["SGD", "Adam", "AdamW", "RMSprop"])
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.mode == "check":
        design()
        check()
    else:
        if not a.family:
            p.error("--family is required for evaluation")
        if a.force:
            (OUT / f"{a.family}_evaluation.csv").unlink(missing_ok=True)
        evaluate(a.family)
