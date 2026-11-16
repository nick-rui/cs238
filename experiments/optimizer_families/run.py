"""Train and evaluate family-specific optimizer controllers; preserve prior SGD study."""

import argparse, csv, gzip, itertools, json, math, time
from pathlib import Path
import numpy as np
import torch
import family_engine as f

b = f.base
e = f.e
OUT = f.e.ROOT / "results/optimizer_families"
OUT.mkdir(parents=True, exist_ok=True)
JOBS = [
    (family, kind, seed)
    for family in f.FAMILIES
    for kind in ["gradient"]
    for seed in [338, 339]
]


def train(family, kind, seed):
    folder = OUT / f"{family}_{kind}_{seed}"
    folder.mkdir(exist_ok=True)
    if (folder / "done.json").exists():
        return
    torch.manual_seed(seed)
    pol = f.policy(family, kind, seed)
    opt = torch.optim.Adam(pol.parameters(), lr=0.003)
    training = []
    dev = []
    best = math.inf
    start = time.perf_counter()
    for update in range(501):
        if update in [0, 25, 50, 100, 200, 300, 400, 500]:
            rr = []
            for st in range(0, 16, 8):
                rr += f.rollout(
                    family,
                    kind,
                    b.DEVSEEDS[st : st + 8],
                    b.DEVORDERS[st : st + 8],
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
        if update == 500:
            break
        seeds = [5000000 + seed * 10000 + update * 8 + i for i in range(8)]
        rr, _, logp, rewards = f.rollout(
            family, kind, seeds, b.TRAIN * 2, pol, training=True
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
                updates=500,
                episodes=4000,
                threads=torch.get_num_threads(),
                seconds=time.perf_counter() - start,
                failed=sum(r["failed"] for r in training),
            )
        )
    )


def configs(family):
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


def tune(family):
    path = OUT / f"{family}_baseline.json"
    if path.exists():
        return
    candidates = configs(family)
    records = []
    scores = []
    for ci, c in enumerate(candidates):
        rr = f.rollout(
            family,
            "gradient",
            b.DEVSEEDS[:8],
            b.DEVORDERS[:8],
            fixed=c,
            evaluation=(e.xd, e.yd),
        )[0]
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
        return
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    testset = (test.data.reshape(-1, 784).float() / 255, test.targets)
    c = json.loads((OUT / f"{family}_baseline.json").read_text())["config"]
    results = []
    traces = []
    jobs = [("fixed", -1)] + [
        (kind, seed) for kind in ["gradient"] for seed in [338, 339]
    ]
    for kind, seed in jobs:
        pol = None
        checkpoint = -1
        if kind != "fixed":
            saved = torch.load(
                OUT / f"{family}_{kind}_{seed}" / "best.pt", weights_only=True
            )
            pol = f.policy(family, kind, seed)
            pol.load_state_dict(saved["policy"])
            checkpoint = saved["update"]
        for order in b.ORDERS:
            for st in range(0, 20, 4):
                seeds = b.TESTSEEDS[st : st + 4]
                rr, tr, _, _ = f.rollout(
                    family,
                    kind,
                    seeds,
                    [order] * 4,
                    pol,
                    evaluation=testset,
                    trace=True,
                    fixed=c if kind == "fixed" else None,
                )
                for r in rr:
                    r.update(
                        family=family,
                        variant=kind,
                        policy_seed=seed,
                        checkpoint=checkpoint,
                    )
                for r in tr:
                    r.update(family=family, variant=kind, policy_seed=seed)
                results += rr
                traces += tr
        print("EVALUATED", family, kind, seed, checkpoint, flush=True)
    e.writecsv(path, results)
    e.writecsv(OUT / f"{family}_traces.csv", traces)


def import_sgd():
    root = OUT.parent / "recovery_study"
    mapping = {"Fixed SGD": "fixed", "Policy-controlled SGD": "gradient"}
    for src, dest in [
        ("evaluation.csv", "SGD_evaluation.csv"),
        ("traces.csv", "SGD_traces.csv"),
    ]:
        rows = []
        for r in csv.DictReader(
            (root / src).open()
            if (root / src).exists()
            else gzip.open(str(root / src) + ".gz", "rt")
        ):
            if r["method"] not in mapping:
                continue
            r["family"] = "SGD"
            r["variant"] = mapping[r.pop("method")]
            rows.append(r)
        e.writecsv(OUT / dest, rows)
    c = json.loads((root / "baseline_configs.json").read_text())["SGD"]
    (OUT / "SGD_baseline.json").write_text(
        json.dumps(
            dict(config=c, reused=True, candidates=144, episodes_per_candidate=8),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "tune", "evaluate", "import"])
    p.add_argument("--family", choices=f.FAMILIES)
    p.add_argument("--job")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--force",
        action="store_true",
        help="Replace completed outputs for the selected operation",
    )
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.mode == "import":
        import_sgd()
    elif a.mode == "train":
        for fam, kind, seed in JOBS:
            if (not a.family or fam == a.family) and (
                not a.job or a.job == f"{fam}_{kind}_{seed}"
            ):
                if a.force:
                    (OUT / f"{fam}_{kind}_{seed}" / "done.json").unlink(missing_ok=True)
                train(fam, kind, seed)
    elif a.mode == "tune":
        for fam in [a.family] if a.family else f.FAMILIES:
            if a.force:
                (OUT / f"{fam}_baseline.json").unlink(missing_ok=True)
            tune(fam)
    else:
        for fam in [a.family] if a.family else f.FAMILIES:
            if a.force:
                (OUT / f"{fam}_evaluation.csv").unlink(missing_ok=True)
            evaluate(fam)
