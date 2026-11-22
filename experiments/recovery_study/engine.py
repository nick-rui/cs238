"""Batched rollout engine for recovery from biased streams.
CPU batched independent classifiers; no gradient through optimizer steps.
Run study.py to train, tune and evaluate this experiment.
"""

import argparse, csv, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST, FashionMNIST

ROOT = Path(__file__).resolve().parents[2]
DATASET = os.environ.get("CS238_DATASET", "mnist")
if DATASET not in ["mnist", "fashion_mnist"]:
    raise ValueError(f"Unknown CS238_DATASET: {DATASET!r}")
RESULTS = ROOT / "results"
if DATASET == "fashion_mnist":
    MNIST = FashionMNIST
    RESULTS = RESULTS / "fashion_mnist"
OUT = RESULTS / "recovery_study"
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(os.environ.get("MNIST_DATA_DIR", ROOT / "data"))
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
data = MNIST(str(DATA_DIR), train=True, download=True)
ix = torch.randperm(60000, generator=torch.Generator().manual_seed(238))
X = data.data.reshape(-1, 784).float() / 255
Y = data.targets
x, y = X[ix[:50000]], Y[ix[:50000]]
ink = x.sum(1)
xc, yc = X[ix[50000:52000]], Y[ix[50000:52000]]
xd, yd = X[ix[52000:54000]], Y[ix[52000:54000]]
STEPS = 200
INTERVAL = 5
H = 40
UPDATES = 500
CHECKS = [0, 25, 50, 100, 200, 300, 400, 500]


def policy(kind, seed, warm=None):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        inputs = 6 if kind == "no_feedback" else 9
        outputs = {"lr_only": 1, "fixed_memory": 2}.get(kind, 3)
        p = nn.Sequential(nn.Linear(inputs, 16), nn.Tanh(), nn.Linear(16, outputs))
        nn.init.zeros_(p[-1].weight)
        nn.init.zeros_(p[-1].bias)
        if warm is not None:
            _, mom, wd = warm_settings(warm)
            # Relative lr keeps bias zero; clipped inverse tanh handles boundaries.
            targets = [2 * mom / 0.95 - 1, (math.log10(wd) + 6) / 2 - 1]
            with torch.no_grad():
                for i, target in enumerate(targets, 1):
                    p[-1].bias[i] = math.atanh(max(-math.tanh(3), min(math.tanh(3), target)))
    return p


def warm_settings(config):
    return (max(0.001, min(0.3, config["lr"])),
            max(0.0, min(0.95, config["momentum"])),
            max(1e-6, min(0.01, config["weight_decay"])))


def batches(seed, order):
    gen = torch.Generator().manual_seed(seed + 100000)
    ids = torch.randint(len(x), (STEPS * 64,), generator=gen)
    if order in ["ink_ascending", "ink_descending"]:
        return ids[
            torch.argsort(ink[ids], descending=order == "ink_descending", stable=True)
        ].reshape(STEPS, 64)
    if order == "roundrobin":
        queues = [ids[y[ids] == c] for c in range(10)]
        parts = []
        for start in range(0, max(len(q) for q in queues), 64):
            for q in queues:
                if start < len(q):
                    parts.append(q[start : start + 64])
        return torch.cat(parts).reshape(STEPS, 64)
    if order == "bursty":
        queues = [ids[y[ids] == c] for c in range(10)]
        positions = [0] * 10
        parts = []
        previous = -1
        while any(positions[c] < len(queues[c]) for c in range(10)):
            choices = [
                c for c in range(10) if positions[c] < len(queues[c]) and c != previous
            ]
            if not choices:
                choices = [c for c in range(10) if positions[c] < len(queues[c])]
            c = choices[int(torch.randint(len(choices), (1,), generator=gen))]
            size = int(torch.randint(1, 25, (1,), generator=gen)) * 64
            chunk = queues[c][positions[c] : positions[c] + size]
            parts.append(chunk)
            positions[c] += len(chunk)
            previous = c
        return torch.cat(parts).reshape(STEPS, 64)
    if order != "shuffled":
        presets = {
            "ascending": list(range(10)),
            "descending": list(range(9, -1, -1)),
            "evenodd": [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
            "zigzag": [0, 9, 1, 8, 2, 7, 3, 6, 4, 5],
        }
        permutation = (
            torch.tensor(presets[order])
            if order in presets
            else torch.randperm(10, generator=gen)
        )
        rank = torch.empty(10, dtype=torch.long)
        rank[permutation] = torch.arange(10)
        ids = ids[torch.argsort(rank[y[ids]], stable=True)]
    return ids.reshape(STEPS, 64)


def parameters(seeds):
    nets = []
    for seed in seeds:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            nets.append(nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10)))
    return [
        torch.stack([list(n.parameters())[i].detach() for n in nets]).requires_grad_()
        for i in range(4)
    ]


def forward(p, xx):
    z = torch.bmm(xx, p[0].transpose(1, 2)) + p[1][:, None, :]
    return torch.bmm(z.relu(), p[2].transpose(1, 2)) + p[3][:, None, :]


def losses(p, xx, yy):
    z = forward(p, xx)
    b, n, _ = z.shape
    return (
        nn.functional.cross_entropy(z.reshape(-1, 10), yy.reshape(-1), reduction="none")
        .reshape(b, n)
        .mean(1)
    )


def dot(a, b):
    return sum((u.flatten(1) * v.flatten(1)).sum(1) for u, v in zip(a, b))


def cosine(a, b):
    return dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt() + 1e-12)


def rollout(
    pol,
    kind,
    seeds,
    orders,
    training=False,
    evaluation=None,
    schedules=None,
    trace=False,
    warm=None,
    held=None,
    controller=None,
):
    # controller(t, obs) -> (lr, momentum, weight_decay): optional hand-written
    # feedback rule using the policy's observation vector; None keeps all paths.
    if controller is not None and (schedules is not None or training or kind != "gradient"):
        raise ValueError("A controller replaces the policy and needs the gradient observations")
    n = len(seeds)
    p = parameters(seeds)
    v = [torch.zeros_like(q) for q in p]
    prevgrad = [q.clone() for q in v]
    ids = torch.stack([batches(s, o) for s, o in zip(seeds, orders)])
    probeids = torch.stack(
        [
            torch.randperm(
                len(xc), generator=torch.Generator().manual_seed(s + 200000)
            )[:512]
            for s in seeds
        ]
    )
    px, py = xc[probeids], yc[probeids]
    lr = torch.full((n,), 0.1)
    mom = torch.full((n,), 0.475)
    wd = torch.full((n,), 1e-4)
    if warm is not None:
        for state, value in zip((lr, mom, wd), warm_settings(warm)):
            state.fill_(value)
    if held is not None and kind in ["lr_only", "fixed_memory"]:
        mom.fill_(held["momentum"])
        if kind == "lr_only":
            wd.fill_(held["weight_decay"])
    with torch.no_grad():
        initial = losses(p, px, py)
        prev = initial.clone()
    trend = torch.zeros(n)
    active = torch.ones(n, dtype=torch.bool)
    rewards = []
    logs = []
    traces = []
    classhistory = []
    t0 = time.perf_counter()
    evalpath = []
    for t in range(H):
        # Compute upcoming batch's gradient now, then reuse it as the first update.
        bx = x[ids[:, t * INTERVAL]]
        by = y[ids[:, t * INTERVAL]]
        g = torch.autograd.grad(losses(p, bx, by).sum(), p)
        obs = torch.stack(
            [
                torch.log(lr / 0.1),
                mom / 0.95,
                (torch.log10(wd) + 4) / 2,
                torch.full((n,), t / (H - 1)),
                torch.log((prev + 1e-8) / (initial + 1e-8)),
                trend,
            ],
            1,
        )
        if kind in ["gradient", "lr_only", "fixed_memory"]:
            extra = torch.stack(
                [
                    torch.log(dot(g, g).sqrt() + 1e-8) / 10,
                    cosine(g, v),
                    cosine(g, prevgrad),
                ],
                1,
            )
            obs = torch.cat([obs, extra], 1)
        if controller is not None:
            safe_obs = torch.nan_to_num(obs.detach(), nan=0.0, posinf=20.0, neginf=-20.0)
            lr, mom, wd = [torch.as_tensor(v, dtype=torch.float32).expand(n).clone()
                           for v in controller(t, safe_obs)]
        elif schedules is None:
            if warm is not None or kind == "lr_only":
                obs = torch.nan_to_num(obs, nan=0.0, posinf=20.0, neginf=-20.0)
            mu = pol(obs.detach())
            dist = torch.distributions.Normal(mu, torch.full_like(mu, 0.5))
            raw = dist.sample() if training else mu.detach()
            logs.append(dist.log_prob(raw).sum(1))
            a = raw.detach().tanh()
            lr = (lr * 2 ** a[:, 0]).clamp(0.001, 0.3)
            if kind not in ["lr_only", "fixed_memory"]:
                mom = 0.95 * (a[:, 1] + 1) / 2
            if kind != "lr_only":
                wd = 10 ** (-6 + 2 * (a[:, 1 if kind == "fixed_memory" else 2] + 1))
        else:
            settings = torch.tensor([schedules[o][t] for o in orders])
            lr, mom, wd = settings[:, 0], settings[:, 1], settings[:, 2]
        for j in range(INTERVAL):
            if j:
                g = torch.autograd.grad(
                    losses(
                        p, x[ids[:, t * INTERVAL + j]], y[ids[:, t * INTERVAL + j]]
                    ).sum(),
                    p,
                )
            with torch.no_grad():
                active &= torch.stack(
                    [torch.isfinite(q).flatten(1).all(1) for q in g]
                ).all(0)
                new = []
                for k, (q, gg) in enumerate(zip(p, g)):
                    shape = [n] + [1] * (q.ndim - 1)
                    grad = torch.nan_to_num(gg) * active.reshape(shape)
                    v[k] = mom.reshape(shape) * v[k] + grad + wd.reshape(shape) * q
                    value = q - lr.reshape(shape) * v[k]
                    value = torch.where(
                        active.reshape(shape), value, torch.zeros_like(value)
                    )
                    new.append(value.detach().requires_grad_())
                prevgrad = [torch.nan_to_num(q.detach()) for q in g]
                p = new
        with torch.no_grad():
            current = losses(p, px, py)
            valid = torch.isfinite(current) & (current < 100) & active
            r = torch.where(
                valid,
                torch.log(prev + 1e-8) - torch.log(current + 1e-8),
                torch.where(
                    active, torch.full_like(prev, -10.0), torch.zeros_like(prev)
                ),
            )
            active &= valid
            current = torch.nan_to_num(
                current, nan=100.0, posinf=100.0, neginf=100.0
            ).clamp(max=100)
            rewards.append(r - torch.log(current.clamp(min=1e-8)) / H)
            prev = current
            trend = r
            if evaluation is not None:
                ex, ey = evaluation
                ex = ex[:1000]
                ey = ey[:1000]
                ez = forward(p, ex.unsqueeze(0).expand(n, -1, -1))
                ep_loss = (
                    nn.functional.cross_entropy(
                        ez.reshape(-1, 10),
                        ey.unsqueeze(0).expand(n, -1).reshape(-1),
                        reduction="none",
                    )
                    .reshape(n, -1)
                    .mean(1)
                )
                ep_acc = (ez.argmax(2) == ey).float().mean(1)
                evalpath.append(ep_loss)
            if trace:
                pred = forward(p, px).argmax(2)
                ca = torch.stack(
                    [
                        ((pred == py) & (py == c)).sum(1)
                        / (py == c).sum(1).clamp(min=1)
                        for c in range(10)
                    ],
                    1,
                )
                classhistory.append(ca)
                for i, s in enumerate(seeds):
                    traces.append(
                        dict(
                            seed=s,
                            order=orders[i],
                            step=(t + 1) * INTERVAL,
                            lr=float(lr[i]),
                            momentum=float(mom[i]),
                            weight_decay=float(wd[i]),
                            control_loss=float(current[i]),
                            control_accuracy=float((pred[i] == py[i]).float().mean()),
                            eval_probe_loss=(
                                float(ep_loss[i]) if evaluation is not None else None
                            ),
                            eval_probe_accuracy=(
                                float(ep_acc[i]) if evaluation is not None else None
                            ),
                            active=bool(active[i]),
                            obs_log_relative_loss=float(obs[i, 4]),
                            obs_recent_improvement=float(obs[i, 5]),
                            obs_log_grad_norm=(
                                float(obs[i, 6] * 10)
                                if kind in ["gradient", "lr_only", "fixed_memory"]
                                else None
                            ),
                            obs_grad_momentum_cosine=(
                                float(obs[i, 7])
                                if kind in ["gradient", "lr_only", "fixed_memory"]
                                else None
                            ),
                            obs_grad_previous_cosine=(
                                float(obs[i, 8])
                                if kind in ["gradient", "lr_only", "fixed_memory"]
                                else None
                            ),
                        )
                    )
    rs = torch.stack(rewards, 1)
    good = active
    # Reward includes both final loss reduction and mean log control loss.
    evloss = evacc = None
    if evaluation is not None:
        ex, ey = evaluation
        with torch.no_grad():
            total = torch.zeros(n)
            correct = torch.zeros(n)
            counts = torch.zeros(n, 10)
            classn = torch.bincount(ey, minlength=10)
            for start in range(0, len(ex), 500):
                xx = ex[start : start + 500].unsqueeze(0).expand(n, -1, -1)
                yy = ey[start : start + 500].unsqueeze(0).expand(n, -1)
                z = forward(p, xx)
                total += (
                    nn.functional.cross_entropy(
                        z.reshape(-1, 10), yy.reshape(-1), reduction="none"
                    )
                    .reshape(n, -1)
                    .sum(1)
                )
                hit = z.argmax(2) == yy
                correct += hit.sum(1)
                for c in range(10):
                    counts[:, c] += (hit & (yy == c)).sum(1)
            evloss = (total / len(ex)).tolist()
            evacc = (correct / len(ex)).tolist()
            perclass = (counts / classn.clamp(min=1)).tolist()
    rows = [
        dict(
            seed=s,
            order=orders[i],
            control_loss=float(prev[i]),
            return_value=float(rs[i].sum()),
            failed=not bool(active[i]),
            eval_loss=evloss[i] if evloss else None,
            accuracy=evacc[i] if evacc else None,
            per_class_accuracy=perclass[i] if evloss else None,
            path_log_loss=(
                float(torch.stack(evalpath).clamp(min=1e-8).log().mean(0)[i])
                if evalpath
                else None
            ),
        )
        for i, s in enumerate(seeds)
    ]
    return rows, traces, torch.stack(logs, 1) if logs else None, rs, classhistory


def writecsv(path, rows):
    with path.open("w") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
