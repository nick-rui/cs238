"""Batched policy-controlled Adam, AdamW and RMSprop with persistent memory."""

import sys, math
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recovery_study"))
import study as base

e = base.e
FAMILIES = ["Adam", "AdamW", "RMSprop"]
INIT = {
    "Adam": (0.003, 0.9, 0.999),
    "AdamW": (0.003, 0.9, 0.999),
    "RMSprop": (0.001, 0.5, 0.99),
}
EPS = 1e-8


def policy(family, kind, seed, warm=None):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        inputs = 7 if kind == "no_feedback" else 10
        outputs = {"lr_only": 1, "fixed_memory": 2}.get(kind, 4)
        pol = nn.Sequential(nn.Linear(inputs, 16), nn.Tanh(), nn.Linear(16, outputs))
        nn.init.zeros_(pol[-1].weight)
        nn.init.zeros_(pol[-1].bias)
        _, moment, decay = INIT[family]
        with torch.no_grad():
            if kind in ["lr_only", "fixed_memory"]:
                return pol
            pol[-1].bias[1] = math.atanh(2 * moment / 0.95 - 1)
            pol[-1].bias[2] = math.atanh(2 * (-math.log10(1 - decay) - 1) / 3 - 1)
            if warm is not None:
                _, moment, decay, wd = warm_settings(warm)
                targets = [2 * moment / 0.95 - 1,
                           2 * (-math.log10(1 - decay) - 1) / 3 - 1,
                           2 * math.log1p(wd / 1e-6) / math.log(10001) - 1]
                # Clip inverse tanh to finite logits; boundary targets use +/-3.
                for i, target in enumerate(targets, 1):
                    pol[-1].bias[i] = math.atanh(max(-math.tanh(3), min(math.tanh(3), target)))
    return pol


def warm_settings(config):
    return (max(1e-4, min(0.03, config["lr"])),
            max(0.0, min(0.95, config.get("beta1", config.get("momentum")))),
            max(0.9, min(0.9999, config.get("beta2", config.get("alpha")))),
            max(0.0, min(0.01, config["weight_decay"])))


class Memory:
    def __init__(self, p):
        self.first = [torch.zeros_like(q) for q in p]
        self.second = [torch.zeros_like(q) for q in p]
        self.b1prod = torch.ones(len(p[0]), dtype=p[0].dtype)
        self.b2prod = self.b1prod.clone()

    def update(self, p, g, family, lr, mom, decay, wd):
        n = len(p[0])
        new = []
        if family in ["Adam", "AdamW"]:
            self.b1prod *= mom
            self.b2prod *= decay
        for k, (q, gg) in enumerate(zip(p, g)):
            shape = [n] + [1] * (q.ndim - 1)
            eta = lr.reshape(shape)
            mu = mom.reshape(shape)
            rho = decay.reshape(shape)
            lam = wd.reshape(shape)
            grad = gg if family == "AdamW" else gg + lam * q
            if family in ["Adam", "AdamW"]:
                self.first[k] = mu * self.first[k] + (1 - mu) * grad
                self.second[k] = rho * self.second[k] + (1 - rho) * grad.square()
                mhat = self.first[k] / (1 - self.b1prod).reshape(shape)
                vhat = self.second[k] / (1 - self.b2prod).reshape(shape)
                value = q * (1 - eta * lam) if family == "AdamW" else q
                value = value - eta * mhat / (vhat.sqrt() + EPS)
            else:
                self.second[k] = rho * self.second[k] + (1 - rho) * grad.square()
                self.first[k] = mu * self.first[k] + grad / (
                    self.second[k].sqrt() + EPS
                )
                value = q - eta * self.first[k]
            new.append(value.detach().requires_grad_())
        return new


def rollout(
    family,
    kind,
    seeds,
    orders,
    pol=None,
    training=False,
    evaluation=None,
    trace=False,
    fixed=None,
    warm=None,
    held=None,
    schedule=None,
):
    if schedule is not None:
        if training or fixed is not None:
            raise ValueError("Schedule replay cannot train or use a fixed configuration")
        schedule = torch.as_tensor(schedule, dtype=torch.float32)
        if schedule.shape != (40, 4) or not torch.isfinite(schedule).all():
            raise ValueError("Expected 40 finite (lr, momentum, second_decay, wd) rows")
    n = len(seeds)
    p = e.parameters(seeds)
    mem = Memory(p)
    prevgrad = [torch.zeros_like(q) for q in p]
    ids = torch.stack([base.batches(s, o) for s, o in zip(seeds, orders)])
    probeids = torch.stack(
        [
            torch.randperm(
                len(e.xc), generator=torch.Generator().manual_seed(s + 200000)
            )[:512]
            for s in seeds
        ]
    )
    px, py = e.xc[probeids], e.yc[probeids]
    initial_lr, initial_mom, initial_decay = INIT[family]
    lr = torch.full((n,), initial_lr)
    mom = torch.full((n,), initial_mom)
    decay = torch.full((n,), initial_decay)
    wd = torch.full((n,), 1e-4)
    if warm is not None:
        for state, value in zip((lr, mom, decay, wd), warm_settings(warm)):
            state.fill_(value)
    if held is not None and kind in ["lr_only", "fixed_memory"]:
        mom.fill_(held.get("beta1", held.get("momentum")))
        decay.fill_(held.get("beta2", held.get("alpha")))
        if kind == "lr_only":
            wd.fill_(held["weight_decay"])
    if fixed is not None:
        lr.fill_(fixed["lr"])
        mom.fill_(fixed.get("beta1", fixed.get("momentum")))
        decay.fill_(fixed.get("beta2", fixed.get("alpha")))
        wd.fill_(fixed["weight_decay"])
    with torch.no_grad():
        initial = e.losses(p, px, py)
        prev = initial.clone()
    trend = torch.zeros(n)
    active = torch.ones(n, dtype=torch.bool)
    rewards = []
    logs = []
    traces = []
    evalpath = []
    for t in range(40):
        g = torch.autograd.grad(
            e.losses(p, e.x[ids[:, t * 5]], e.y[ids[:, t * 5]]).sum(), p
        )
        if schedule is not None:
            lr, mom, decay, wd = [value.expand(n) for value in schedule[t]]
        elif fixed is None:
            obs = torch.stack(
                [
                    torch.log(lr / initial_lr),
                    mom / 0.95,
                    (torch.log10((1 - decay).clamp(min=1e-6)) + 2.5) / 1.5,
                    2 * torch.log1p(wd / 1e-6) / math.log(10001) - 1,
                    torch.full((n,), t / 39),
                    torch.log((prev + 1e-8) / (initial + 1e-8)),
                    trend,
                ],
                1,
            )
            if kind in ["gradient", "lr_only", "fixed_memory"]:
                obs = torch.cat(
                    [
                        obs,
                        torch.stack(
                            [
                                torch.log(e.dot(g, g).sqrt() + 1e-8) / 10,
                                e.cosine(g, mem.first),
                                e.cosine(g, prevgrad),
                            ],
                            1,
                        ),
                    ],
                    1,
                )
            obs = torch.nan_to_num(obs.detach(), nan=0.0, posinf=20.0, neginf=-20.0)
            mu = pol(obs)
            dist = torch.distributions.Normal(mu, torch.full_like(mu, 0.5))
            raw = dist.sample() if training else mu.detach()
            logs.append(dist.log_prob(raw).sum(1))
            a = raw.detach().tanh()
            lr = (lr * 2 ** a[:, 0]).clamp(0.0001, 0.03)
            if kind not in ["lr_only", "fixed_memory"]:
                mom = 0.95 * (a[:, 1] + 1) / 2
                decay = 1 - 10 ** (-1 - 3 * (a[:, 2] + 1) / 2)
            if kind != "lr_only":
                wd = 1e-6 * torch.expm1((a[:, 1 if kind == "fixed_memory" else 3] + 1) / 2 * math.log(10001))
        was_active = active.clone()
        for j in range(5):
            if j:
                g = torch.autograd.grad(
                    e.losses(p, e.x[ids[:, t * 5 + j]], e.y[ids[:, t * 5 + j]]).sum(), p
                )
            with torch.no_grad():
                active &= torch.stack(
                    [torch.isfinite(q).flatten(1).all(1) for q in g]
                ).all(0)
                safe = [
                    torch.nan_to_num(q.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                    for q in g
                ]
                p = mem.update(p, safe, family, lr, mom, decay, wd)
                active &= torch.stack(
                    [torch.isfinite(q).flatten(1).all(1) for q in p]
                ).all(0)
                for k, q in enumerate(p):
                    shape = [n] + [1] * (q.ndim - 1)
                    p[k] = (
                        torch.where(active.reshape(shape), q, torch.zeros_like(q))
                        .detach()
                        .requires_grad_()
                    )
                    mem.first[k] = torch.where(
                        active.reshape(shape), mem.first[k], torch.zeros_like(q)
                    )
                    mem.second[k] = torch.where(
                        active.reshape(shape), mem.second[k], torch.zeros_like(q)
                    )
                prevgrad = safe
        with torch.no_grad():
            current = e.losses(p, px, py)
            active &= torch.isfinite(current) & (current < 100)
            # Failed episodes remain failed and receive loss100 thereafter; no dropped runs.
            current = torch.where(active, current, torch.full_like(current, 100.0))
            improvement = torch.where(
                active,
                torch.log(prev + 1e-8) - torch.log(current + 1e-8),
                torch.where(
                    was_active, torch.full_like(prev, -10.0), torch.zeros_like(prev)
                ),
            )
            rewards.append(improvement - torch.log(current.clamp(min=1e-8)) / 40)
            prev = current
            trend = improvement
            if evaluation is not None:
                ex, ey = evaluation
                ex = ex[:1000]
                ey = ey[:1000]
                ez = e.forward(p, ex.unsqueeze(0).expand(n, -1, -1))
                el = (
                    nn.functional.cross_entropy(
                        ez.reshape(-1, 10),
                        ey.unsqueeze(0).expand(n, -1).reshape(-1),
                        reduction="none",
                    )
                    .reshape(n, -1)
                    .mean(1)
                )
                ea = (ez.argmax(2) == ey).float().mean(1)
                el = torch.where(active, el, torch.full_like(el, 100.0))
                ea = torch.where(active, ea, torch.zeros_like(ea))
                evalpath.append(el)
            if trace:
                ca = (e.forward(p, px).argmax(2) == py).float().mean(1)
                for i, s in enumerate(seeds):
                    traces.append(
                        dict(
                            seed=s,
                            order=orders[i],
                            step=(t + 1) * 5,
                            lr=float(lr[i]),
                            momentum=float(mom[i]),
                            second_decay=float(decay[i]),
                            weight_decay=float(wd[i]),
                            control_loss=float(current[i]),
                            control_accuracy=float(ca[i]) if active[i] else 0.0,
                            eval_probe_loss=(
                                float(el[i]) if evaluation is not None else None
                            ),
                            eval_probe_accuracy=(
                                float(ea[i]) if evaluation is not None else None
                            ),
                            active=bool(active[i]),
                        )
                    )
    rs = torch.stack(rewards, 1)
    evloss = evacc = None
    if evaluation is not None:
        ex, ey = evaluation
        with torch.no_grad():
            total = torch.zeros(n)
            correct = torch.zeros(n)
            for start in range(0, len(ex), 500):
                xx = ex[start : start + 500].unsqueeze(0).expand(n, -1, -1)
                yy = ey[start : start + 500].unsqueeze(0).expand(n, -1)
                z = e.forward(p, xx)
                total += (
                    nn.functional.cross_entropy(
                        z.reshape(-1, 10), yy.reshape(-1), reduction="none"
                    )
                    .reshape(n, -1)
                    .sum(1)
                )
                correct += (z.argmax(2) == yy).sum(1)
            evloss = torch.where(
                active, total / len(ex), torch.full_like(total, 100.0)
            ).tolist()
            evacc = torch.where(
                active, correct / len(ex), torch.zeros_like(correct)
            ).tolist()
    rows = [
        dict(
            seed=s,
            order=orders[i],
            control_loss=float(prev[i]),
            return_value=float(rs[i].sum()),
            failed=not bool(active[i]),
            eval_loss=evloss[i] if evloss else None,
            accuracy=evacc[i] if evacc else None,
            path_log_loss=(
                float(torch.stack(evalpath).clamp(min=1e-8).log().mean(0)[i])
                if evalpath
                else None
            ),
        )
        for i, s in enumerate(seeds)
    ]
    return (
        rows,
        base.annotate(traces) if trace else [],
        torch.stack(logs, 1) if logs else None,
        rs,
    )
