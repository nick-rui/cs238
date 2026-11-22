"""Recovery from biased training streams, with predetermined train/dev/test separation."""

import argparse, csv, json, math, time, functools
from pathlib import Path
import numpy as np
import torch
from torch import nn
import engine as e

OUT = e.ROOT / "results/recovery_study"
OUT.mkdir(parents=True, exist_ok=True)
ORDERS = [
    "mixed",
    "single_early",
    "single_middle",
    "single_late",
    "pair_long",
    "two_shocks",
    "repeat_same",
    "three_shocks",
    "ramp",
    "rapid",
]
TRAIN = ["mixed", "single_train", "pair_train", "double_train"]
JOBS = [("gradient", 338), ("gradient", 339)]
DEVSEEDS = list(range(1810000, 1810016))
DEVORDERS = TRAIN * 4
TESTSEEDS = list(range(1900000, 1900020))
class_ids = [torch.where(e.y == d)[0].numpy() for d in range(10)]


@functools.lru_cache(maxsize=1024)
def stream(seed, order):
    rng = np.random.default_rng(seed + 700000)
    # Identical balanced bag in every order: 1280 draws per digit, with replacement.
    bags = [rng.choice(q, 1280, replace=True) for q in class_ids]
    digits = rng.permutation(10).tolist()
    d = digits[0]
    pair = digits[:2]
    events = []

    def event(start, length, targets, profile="pure"):
        events.append(
            dict(
                start=int(start),
                end=int(start + length),
                digits=list(targets),
                profile=profile,
            )
        )

    if order == "single_train":
        event(rng.integers(25, 140), rng.integers(8, 17), [d])
    elif order == "pair_train":
        event(rng.integers(25, 125), rng.integers(18, 33), pair)
    elif order == "double_train":
        event(rng.integers(25, 65), rng.integers(6, 13), [d])
        event(rng.integers(105, 150), rng.integers(6, 13), [digits[1]])
    elif order == "single_early":
        event(20, 16, [d])
    elif order == "single_middle":
        event(80, 16, [d])
    elif order == "single_late":
        event(150, 16, [d])
    elif order == "pair_long":
        event(75, 32, pair)
    elif order == "two_shocks":
        event(40, 12, [d])
        event(115, 12, [digits[1]])
    elif order == "repeat_same":
        event(45, 8, [d])
        event(120, 8, [d])
    elif order == "three_shocks":
        for start, target in zip([35, 85, 135], digits[:3]):
            event(start, 10, [target])
    elif order == "ramp":
        event(70, 40, pair, "ramp")
    elif order == "rapid":
        for k, start in enumerate([40, 55, 70, 100, 115, 130]):
            event(start, 3, [digits[k % 2]])
    elif order != "mixed":
        raise ValueError(order)
    ids = np.full((200, 64), -1, dtype=np.int64)
    mask = np.zeros(200, dtype=bool)
    positions = [0] * 10
    for ev in events:
        for step in range(ev["start"], ev["end"]):
            mask[step] = True
            frac = (
                1.0
                if ev["profile"] == "pure"
                else 0.2
                + 0.7
                * (
                    1
                    - abs(
                        2 * (step - ev["start"] + 0.5) / (ev["end"] - ev["start"]) - 1
                    )
                )
            )
            count = int(round(64 * frac))
            targets = ev["digits"]
            for slot in range(count):
                target = targets[slot % len(targets)]
                pos = positions[target]
                assert pos < len(bags[target]), (seed, order, events)
                ids[step, slot] = bags[target][pos]
                positions[target] += 1
    remaining = np.concatenate([bag[pos:] for bag, pos in zip(bags, positions)])
    rng.shuffle(remaining)
    ids[ids < 0] = remaining
    # Shuffle within minibatches, preserving class composition.
    for row in ids:
        rng.shuffle(row)
    return torch.from_numpy(ids), events, mask


def batches(seed, order):
    return stream(int(seed), order)[0]


e.batches = batches


def score(r):
    return math.log(max(r["eval_loss"], 1e-8)) + r["path_log_loss"]


def annotate(trace):
    for r in trace:
        _, events, mask = stream(int(r["seed"]), r["order"])
        start = int(r["step"]) - 5
        r["biased_fraction"] = float(mask[start : start + 5].mean())
    return trace


def check():
    for seed in [1810000, 1900000, 1900001]:
        reference = batches(seed, "mixed").flatten().sort().values
        for order in ORDERS + TRAIN:
            ids, events, mask = stream(seed, order)
            assert ids.shape == (200, 64) and torch.equal(
                ids.flatten().sort().values, reference
            )
            assert torch.equal(
                torch.bincount(e.y[ids.flatten()]), torch.full((10,), 1280)
            )
            for ev in events:
                if ev["profile"] == "pure":
                    assert torch.isin(
                        e.y[ids[ev["start"] : ev["end"]]], torch.tensor(ev["digits"])
                    ).all()
    pol = e.policy("gradient", 338)
    rr, tr, lp, rw, _ = e.rollout(
        pol,
        "gradient",
        [1810000, 1810001],
        ["mixed", "single_middle"],
        training=True,
        evaluation=(e.xd, e.yd),
        trace=True,
    )
    assert lp.shape == rw.shape == (2, 40) and len(tr) == 80
    (-lp.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in pol.parameters())
    (OUT / "checks.json").write_text(
        json.dumps(
            dict(
                same_bag_all_orders=True,
                exact_class_balance=True,
                pure_blocks_correct=True,
                forty_decisions=True,
                finite_policy_gradients=True,
            ),
            indent=2,
        )
    )
    print("CHECKS PASSED", flush=True)


def train(kind, seed, updates=500):
    folder = OUT / f"{kind}_{seed}"
    folder.mkdir(exist_ok=True)
    if (folder / "done.json").exists():
        return
    torch.manual_seed(seed)
    pol = e.policy(kind, seed)
    opt = torch.optim.Adam(pol.parameters(), lr=0.003)
    training = []
    dev = []
    best = math.inf
    start = time.perf_counter()
    for update in range(updates + 1):
        if update in [0, 25, 50, 100, 200, 300, 400, 500] or update == updates:
            rr = []
            for st in range(0, 16, 8):
                chunk, *_ = e.rollout(
                    pol,
                    kind,
                    DEVSEEDS[st : st + 8],
                    DEVORDERS[st : st + 8],
                    evaluation=(e.xd, e.yd),
                )
                rr += chunk
            value = float(np.mean([score(r) for r in rr]))
            for r in rr:
                r.update(update=update, kind=kind, policy_seed=seed)
            dev += rr
            state = dict(policy=pol.state_dict(), update=update, kind=kind, seed=seed)
            torch.save(state, folder / f"checkpoint_{update}.pt")
            if value < best:
                best = value
                torch.save(state, folder / "best.pt")
            e.writecsv(folder / "development.csv", dev)
            print(
                json.dumps(
                    dict(
                        job=f"{kind}_{seed}",
                        update=update,
                        score=value,
                        accuracy=float(np.mean([r["accuracy"] for r in rr])),
                        seconds=time.perf_counter() - start,
                    )
                ),
                flush=True,
            )
        if update == updates:
            break
        # Two runs of each regime per update; leave-one-out baseline within regime.
        ss = [5000000 + seed * 10000 + update * 8 + i for i in range(8)]
        rr, _, logp, rewards, _ = e.rollout(pol, kind, ss, TRAIN * 2, training=True)
        returns = rewards.flip(1).cumsum(1).flip(1)
        adv = returns - returns[torch.tensor([4, 5, 6, 7, 0, 1, 2, 3])]
        opt.zero_grad()
        (-(logp * adv.detach()).mean()).backward()
        assert all(torch.isfinite(q.grad).all() for q in pol.parameters())
        nn.utils.clip_grad_norm_(pol.parameters(), 1)
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
                updates=updates,
                episodes=updates * 8,
                seconds=time.perf_counter() - start,
                failed=sum(r["failed"] for r in training),
            )
        )
    )


def baseline(c, seed, order, evaluation, trace=False):
    p = e.parameters([seed])
    net = nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
    for q, r in zip(net.parameters(), p):
        q.data.copy_(r[0])
    args = dict(lr=c["lr"], weight_decay=c["weight_decay"])
    if c["family"] == "SGD":
        opt = torch.optim.SGD(net.parameters(), momentum=c["momentum"], **args)
    elif c["family"] in ["Adam", "AdamW"]:
        opt = getattr(torch.optim, c["family"])(
            net.parameters(), betas=(c["beta1"], 0.999), **args
        )
    else:
        opt = torch.optim.RMSprop(
            net.parameters(), momentum=c["momentum"], alpha=0.99, **args
        )
    probe = torch.randperm(
        len(e.xc), generator=torch.Generator().manual_seed(seed + 200000)
    )[:512]
    ex, ey = evaluation
    path = []
    tr = []
    failed = False
    for step, ids in enumerate(batches(seed, order)):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(e.x[ids]), e.y[ids])
        if not torch.isfinite(loss):
            failed = True
            break
        loss.backward()
        opt.step()
        if (step + 1) % 5 == 0:
            with torch.no_grad():
                ez = net(ex[:1000])
                el = float(nn.functional.cross_entropy(ez, ey[:1000]))
                ea = float((ez.argmax(1) == ey[:1000]).float().mean())
                path.append(
                    math.log(max(el, 1e-8)) if math.isfinite(el) else math.log(100.0)
                )
                if trace:
                    cz = net(e.xc[probe])
                    cl = float(nn.functional.cross_entropy(cz, e.yc[probe]))
                    ca = float((cz.argmax(1) == e.yc[probe]).float().mean())
            if trace:
                tr.append(
                    dict(
                        seed=seed,
                        order=order,
                        step=step + 1,
                        lr=c["lr"],
                        momentum=c.get("momentum", c.get("beta1")),
                        weight_decay=c["weight_decay"],
                        control_loss=cl,
                        control_accuracy=ca,
                        eval_probe_loss=el,
                        eval_probe_accuracy=ea,
                        active=not failed,
                    )
                )
    with torch.no_grad():
        z = net(ex)
        el = float(nn.functional.cross_entropy(z, ey))
        acc = float((z.argmax(1) == ey).float().mean())
    if failed or not math.isfinite(el):
        el = 100.0
        acc = 0.0
        path = path + [math.log(100.0)] * (40 - len(path))
    return dict(
        seed=seed,
        order=order,
        eval_loss=el,
        accuracy=acc,
        path_log_loss=float(np.mean(path)),
        failed=failed or not math.isfinite(el),
    ), annotate(tr)


def sgd_configs(grid="default"):
    import itertools

    lr = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    momentum = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
    wd = [0.0, 1e-6, 1e-4, 0.01]
    if grid == "wide":
        lr = [.0003, .001, .002, .003, .006, .01, .02, .03, .06, .1, .2, .3]
        momentum = [0.0, .25, .5, .75, .9, .95, .99]
        wd = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, .01]
    return [
        dict(family="SGD", lr=l, momentum=m, weight_decay=w)
        for l, m, w in itertools.product(lr, momentum, wd)
    ]


def tune(grid="default", threads=1):
    torch.set_num_threads(threads)
    start = time.perf_counter()
    grids = {"SGD": sgd_configs(grid)}
    configs = {}
    records = []
    for family in ["SGD"]:
        scores = []
        for ci, c in enumerate(grids[family]):
            rr = []
            # Same eight development episodes for each candidate, two per training regime.
            for seed, order in zip(DEVSEEDS[:8], DEVORDERS[:8]):
                r, _ = baseline(c, seed, order, (e.xd, e.yd))
                r.update(family=family, config_id=ci)
                rr.append(r)
            records += rr
            scores.append(float(np.mean([score(r) for r in rr])))
            if ci % 24 == 0:
                print("TUNE", family, ci, flush=True)
        index = int(np.argmin(scores))
        configs[family] = grids[family][index]
        print("SELECTED", family, configs[family], scores[index], flush=True)
    e.writecsv(OUT / "baseline_tuning.csv", records)
    (OUT / "baseline_configs.json").write_text(json.dumps(configs, indent=2))
    if grid == "wide":
        (OUT / "SGD_baseline.json").write_text(json.dumps(dict(
            config=configs["SGD"], score=scores[index], grid=grid,
            candidates=len(grids["SGD"]), episodes_per_candidate=8,
            episodes=len(records), seeds=DEVSEEDS[:8], orders=DEVORDERS[:8],
            seconds=time.perf_counter() - start,
        ), indent=2))


def evaluate(config_path=None, fixed_only=False):
    start = time.perf_counter()
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    testset = (test.data.reshape(-1, 784).float() / 255, test.targets)
    configs = json.loads((Path(config_path) if config_path else OUT / "baseline_configs.json").read_text())
    if "config" in configs:
        configs = {"SGD": configs["config"]}
    results = []
    traces = []
    for family, c in configs.items():
        for order in ORDERS:
            for seed in TESTSEEDS:
                r, tr = baseline(c, seed, order, testset, True)
                r.update(method="Fixed " + family, policy_seed=-1)
                results.append(r)
                for t in tr:
                    t.update(method="Fixed " + family, policy_seed=-1)
                traces += tr
        print("EVALUATED", family, flush=True)
    for kind, seed in ([] if fixed_only else JOBS):
        saved = torch.load(OUT / f"{kind}_{seed}" / "best.pt", weights_only=True)
        pol = e.policy(kind, seed)
        pol.load_state_dict(saved["policy"])
        method = "Policy-controlled SGD"
        for order in ORDERS:
            for st in range(0, 20, 4):
                ss = TESTSEEDS[st : st + 4]
                rr, tr, *_ = e.rollout(
                    pol, kind, ss, [order] * 4, evaluation=testset, trace=True
                )
                for r in rr:
                    r.update(
                        method=method, policy_seed=seed, checkpoint=saved["update"]
                    )
                for r in tr:
                    r.update(method=method, policy_seed=seed)
                results += rr
                traces += annotate(tr)
        print("EVALUATED", kind, seed, saved["update"], flush=True)
    e.writecsv(OUT / ("SGD_evaluation.csv" if fixed_only else "evaluation.csv"), results)
    e.writecsv(OUT / ("SGD_traces.csv" if fixed_only else "traces.csv"), traces)
    if fixed_only:
        (OUT / "SGD_done.json").write_text(json.dumps(dict(
            rows=len(results), seconds=time.perf_counter() - start,
            failures=sum(r["failed"] for r in results),
        ), indent=2))
    (OUT / "test_streams.json").write_text(
        json.dumps(
            [
                dict(seed=s, order=o, events=stream(s, o)[1])
                for s in TESTSEEDS
                for o in ORDERS
            ],
            indent=2,
        )
    )
    print("EVALUATION COMPLETE", len(results), len(traces), flush=True)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("mode", choices=["check", "train", "tune", "evaluate"])
    arg.add_argument("--job")
    arg.add_argument("--force", action="store_true")
    arg.add_argument("--grid", choices=["default", "wide"], default="default")
    arg.add_argument("--output-dir", type=Path)
    arg.add_argument("--config-path", type=Path)
    arg.add_argument("--fixed-only", action="store_true")
    arg.add_argument("--threads", type=int)
    a = arg.parse_args()
    if a.grid == "wide" and not a.output_dir:
        arg.error("--grid wide requires --output-dir")
    if a.mode == "evaluate" and (a.output_dir or a.config_path or a.fixed_only):
        if not (a.output_dir and a.config_path and a.fixed_only):
            arg.error("isolated evaluation requires --output-dir, --config-path and --fixed-only")
    if a.output_dir:
        if a.mode not in ["tune", "evaluate"]:
            arg.error("--output-dir is supported for tune/evaluate only")
        OUT = a.output_dir
        OUT.mkdir(parents=True, exist_ok=True)
    if a.threads is not None:
        torch.set_num_threads(a.threads)
    if a.mode == "check":
        check()
    elif a.mode == "train":
        for kind, seed in JOBS:
            if a.job is None or a.job == f"{kind}_{seed}":
                if a.force:
                    (OUT / f"{kind}_{seed}" / "done.json").unlink(missing_ok=True)
                train(kind, seed)
    elif a.mode == "tune":
        tune(a.grid, a.threads or 1)
    else:
        evaluate(a.config_path, a.fixed_only)
