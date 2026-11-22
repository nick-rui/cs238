"""CNN on CIFAR-10 with the regular-ordering protocol, batched on one GPU.

Reuses the MLP engines with CS238_MODEL=cnn and CS238_DEVICE=cuda; only the
data (CIFAR-10, 40k/2k/2k split as in scripts/time_cifar10_episode.py) and the
output folder differ. Train/tune loops are the existing ones; evaluation mirrors
study.evaluate / optimizer_families.run.evaluate on the official test set.
"""

import argparse, csv, gzip, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("CS238_MODEL", "cnn")
os.environ.setdefault("CS238_DEVICE", "cuda")
os.environ["CS238_DATASET"] = "mnist"  # engine import default; data replaced below
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/optimizer_families"))
import run as runner  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.datasets import CIFAR10  # noqa: E402

e, b, f = runner.e, runner.b, runner.f
FAMILIES = ["SGD", "Adam", "AdamW", "RMSprop"]
SEEDS = [338, 339]
OUT = ROOT / "results/cnn_cifar10"


def flat(images):
    return torch.as_tensor(np.asarray(images)).reshape(len(images), -1).float() / 255


def load_cifar10():
    train = CIFAR10(str(e.DATA_DIR), train=True, download=True)
    test = CIFAR10(str(e.DATA_DIR), train=False, download=True)
    xx, yy = flat(train.data), torch.as_tensor(train.targets).long()
    ix = torch.randperm(len(train), generator=torch.Generator().manual_seed(238))
    d = e.DEVICE
    e.x, e.y = xx[ix[:40000]].to(d), yy[ix[:40000]].to(d)
    e.xc, e.yc = xx[ix[40000:42000]].to(d), yy[ix[40000:42000]].to(d)
    e.xd, e.yd = xx[ix[42000:44000]].to(d), yy[ix[42000:44000]].to(d)
    e.ink = e.x.sum(1)
    b.class_ids = [torch.where(e.y == c)[0].cpu().numpy() for c in range(10)]
    b.stream.cache_clear()
    return flat(test.data).to(d), torch.as_tensor(test.targets).long().to(d)


def set_out(path):
    global OUT
    OUT = path
    OUT.mkdir(parents=True, exist_ok=True)
    b.OUT = runner.OUT = OUT


def train(family, seed, updates):
    if family == "SGD":
        b.train("gradient", seed, updates)
    else:
        runner.train(family, "gradient", seed, updates)


def tune(family):
    if family == "SGD":
        if not (OUT / "baseline_configs.json").exists():
            b.tune("default", 1)
    else:
        runner.tune(family)


def config(family):
    if family == "SGD":
        return json.loads((OUT / "baseline_configs.json").read_text())["SGD"]
    return json.loads((OUT / f"{family}_baseline.json").read_text())["config"]


def policy_folder(family, seed):
    return OUT / (f"gradient_{seed}" if family == "SGD" else f"{family}_gradient_{seed}")


def evaluate(family, testset, seeds=b.TESTSEEDS, chunk=4):
    path = OUT / f"{family}_evaluation.csv"
    if path.exists():
        return
    start = time.perf_counter()
    c = config(family)
    results, traces = [], []
    for ps in [-1] + SEEDS:
        pol, checkpoint = None, -1
        if ps != -1:
            saved = torch.load(policy_folder(family, ps) / "best.pt", weights_only=True)
            pol = e.policy("gradient", ps) if family == "SGD" else f.policy(family, "gradient", ps)
            pol.load_state_dict(saved["policy"])
            checkpoint = saved["update"]
        for order in b.ORDERS:
            if ps == -1 and family == "SGD":
                for seed in seeds:
                    r, tr = b.baseline(c, seed, order, testset, True)
                    results.append(r)
                    traces += [dict(t, policy_seed=-1) for t in tr]
                continue
            for st in range(0, len(seeds), chunk):
                ss = seeds[st : st + chunk]
                if family == "SGD":
                    rr, tr, *_ = e.rollout(pol, "gradient", ss, [order] * len(ss),
                                           evaluation=testset, trace=True)
                    tr = b.annotate(tr)
                else:
                    rr, tr, _, _ = f.rollout(family, "gradient", ss, [order] * len(ss), pol,
                                             evaluation=testset, trace=True,
                                             fixed=c if ps == -1 else None)
                results += rr
                traces += [dict(t, policy_seed=ps) for t in tr]
        for r in results:
            r.setdefault("policy_seed", ps)
            r.setdefault("checkpoint", checkpoint)
        print("EVALUATED", family, ps, checkpoint, round(time.perf_counter() - start), flush=True)
    for r in results:
        r.update(family=family, variant="fixed" if r["policy_seed"] == -1 else "gradient")
        r.pop("per_class_accuracy", None)
    e.writecsv(path, results)
    with gzip.open(OUT / f"{family}_traces.csv.gz", "wt") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for t in traces for k in t)))
        w.writeheader()
        w.writerows(traces)
    (OUT / f"{family}_evaluation_done.json").write_text(json.dumps(dict(
        rows=len(results), seconds=time.perf_counter() - start,
        failures=sum(bool(r["failed"]) for r in results)), indent=2))


def time_update(family, seed=338, repeats=3):
    """Time training updates and one 16-episode development evaluation."""
    pol = e.policy("gradient", seed) if family == "SGD" else f.policy(family, "gradient", seed)
    opt = torch.optim.Adam(pol.parameters(), lr=0.003)

    def rollout(*args, **kw):
        if family == "SGD":
            return e.rollout(pol, "gradient", *args, **kw)
        return f.rollout(family, "gradient", *args, pol=pol, **kw)

    seconds = []
    for k in range(repeats + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ss = [5000000 + seed * 10000 + k * 8 + i for i in range(8)]
        rr, _, logp, rewards, *_ = rollout(ss, b.TRAIN * 2, training=True)
        returns = rewards.flip(1).cumsum(1).flip(1)
        adv = returns - returns[torch.tensor([4, 5, 6, 7, 0, 1, 2, 3])]
        opt.zero_grad()
        (-(logp * adv.detach()).mean()).backward()
        opt.step()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    for st in range(0, 16, 8):
        rollout(b.DEVSEEDS[st : st + 8], b.DEVORDERS[st : st + 8], evaluation=(e.xd, e.yd))
    torch.cuda.synchronize()
    dev = time.perf_counter() - t0
    return dict(family=family, update_seconds=seconds[1:], warmup_seconds=seconds[0],
                dev16_seconds=dev)


def summarize(families):
    rows = []
    for family in families:
        path = OUT / f"{family}_evaluation.csv"
        if path.exists():
            rows += list(csv.DictReader(path.open()))
    lines = []
    out = []
    for family in families:
        rr = [r for r in rows if r["family"] == family]
        if not rr:
            continue
        fixed = {(r["order"], r["seed"]): 100 * float(r["accuracy"])
                 for r in rr if r["variant"] == "fixed"}
        for ps in SEEDS:
            pol = {(r["order"], r["seed"]): 100 * float(r["accuracy"])
                   for r in rr if r["variant"] == "gradient" and int(r["policy_seed"]) == ps}
            if not pol:
                continue
            ckpt = next(r["checkpoint"] for r in rr if r["variant"] == "gradient"
                        and int(r["policy_seed"]) == ps)
            for label, orders in [("shuffled", b.ORDERS[:1]), ("biased", b.ORDERS[1:])]:
                seeds = sorted({s for o, s in fixed})
                # Average orderings within test seed, then pair across 20 test seeds.
                fx = np.array([np.mean([fixed[o, s] for o in orders]) for s in seeds])
                pl = np.array([np.mean([pol[o, s] for o in orders]) for s in seeds])
                d = pl - fx
                out.append(dict(family=family, policy_seed=ps, checkpoint=int(ckpt),
                                ordering=label, fixed_accuracy=fx.mean(),
                                policy_accuracy=pl.mean(), difference_pp=d.mean(),
                                se_pp=d.std(ddof=1) / math.sqrt(len(d)),
                                policy_wins=int((d > 0).sum()), episodes=len(d) * len(orders),
                                fixed_failed=sum(r["failed"] == "True" for r in rr
                                                 if r["variant"] == "fixed" and r["order"] in orders),
                                policy_failed=sum(r["failed"] == "True" for r in rr
                                                  if r["variant"] == "gradient" and r["order"] in orders
                                                  and int(r["policy_seed"]) == ps)))
    if out:
        e.writecsv(OUT / "summary.csv", out)
        lines.append("| Family | Policy seed (ckpt) | Ordering | Fixed acc. | Policy acc. | Diff. (pp) ± SE | Wins /20 |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in out:
            lines.append(f"| {r['family']} | {r['policy_seed']} ({r['checkpoint']}) | {r['ordering']} | "
                         f"{r['fixed_accuracy']:.2f} | {r['policy_accuracy']:.2f} | "
                         f"{r['difference_pp']:+.2f} ± {r['se_pp']:.2f} | {r['policy_wins']} |")
        (OUT / "summary.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    # Per-ordering table, policy seeds averaged.
    per = []
    for family in families:
        rr = [r for r in rows if r["family"] == family]
        for order in b.ORDERS:
            fx = [100 * float(r["accuracy"]) for r in rr if r["order"] == order and r["variant"] == "fixed"]
            pl = [100 * float(r["accuracy"]) for r in rr if r["order"] == order and r["variant"] == "gradient"]
            if fx and pl:
                per.append(dict(family=family, order=order, fixed_accuracy=np.mean(fx),
                                policy_accuracy=np.mean(pl), difference_pp=np.mean(pl) - np.mean(fx)))
    if per:
        e.writecsv(OUT / "summary_by_ordering.csv", per)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "tune", "evaluate", "time", "smoke", "summarize"])
    p.add_argument("--family", choices=FAMILIES, required=False)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--updates", type=int, default=500)
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.mode == "summarize":
        set_out(OUT)
        summarize(FAMILIES)
        sys.exit()
    testset = load_cifar10()
    if a.mode == "time":
        set_out(OUT)
        records = [time_update(fam) for fam in ([a.family] if a.family else ["SGD", "Adam"])]
        print(json.dumps(records, indent=2))
        (OUT / "timing.json").write_text(json.dumps(dict(
            device=torch.cuda.get_device_name(), timings=records), indent=2) + "\n")
    elif a.mode == "smoke":
        # Two training updates, one tuning candidate, one test seed per ordering.
        set_out(OUT / "smoke")
        b.sgd_configs = lambda grid="default": [dict(family="SGD", lr=0.01, momentum=0.9, weight_decay=1e-4)]
        families = [a.family] if a.family else ["SGD", "Adam"]
        original = runner.configs
        runner.configs = lambda family, grid="default": original(family, grid)[50:51]
        for fam in families:
            for seed in SEEDS:
                train(fam, seed, 2)
            tune(fam)
            evaluate(fam, testset, seeds=b.TESTSEEDS[:1])
        summarize(families)
    else:
        set_out(OUT)
        if a.mode == "train":
            train(a.family, a.seed, a.updates)
        elif a.mode == "tune":
            tune(a.family)
        else:
            evaluate(a.family, testset)
