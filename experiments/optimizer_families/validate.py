"""Check constant and changing optimizer controls against independent references."""

import json, math
from pathlib import Path
import torch
import family_engine as f

OUT = f.e.ROOT / "results/optimizer_families"
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(452)
checks = {}
for family in f.FAMILIES:
    for wdvalue in [0.0, 0.007]:
        for changing in [False, True]:
            p = [
                torch.randn(2, 3, 4, dtype=torch.float64).requires_grad_(),
                torch.randn(2, 4, dtype=torch.float64).requires_grad_(),
            ]
            mem = f.Memory(p)
            pp = [
                [torch.nn.Parameter(q[i].detach().clone()) for q in p] for i in range(2)
            ]
            opts = []
            for params in pp:
                if family in ["Adam", "AdamW"]:
                    opts.append(
                        getattr(torch.optim, family)(
                            params,
                            lr=0.003,
                            betas=(0.8, 0.99),
                            weight_decay=wdvalue,
                            eps=f.EPS,
                        )
                    )
                else:
                    opts.append(
                        torch.optim.RMSprop(
                            params,
                            lr=0.003,
                            momentum=0.8,
                            alpha=0.99,
                            weight_decay=wdvalue,
                            eps=f.EPS,
                        )
                    )
            for step in range(12):
                lrvalue = 0.003 * (1 + step / 10) if changing else 0.003
                momvalue = (
                    0.8
                    if family in ["Adam", "AdamW"] or not changing
                    else 0.2 + 0.05 * step
                )
                wdv = wdvalue * (1 + step / 10) if changing else wdvalue
                g = [torch.randn_like(q) for q in p]
                with torch.no_grad():
                    p = mem.update(
                        p,
                        g,
                        family,
                        *[
                            torch.full((2,), v, dtype=torch.float64)
                            for v in [lrvalue, momvalue, 0.99, wdv]
                        ]
                    )
                for i, opt in enumerate(opts):
                    opt.param_groups[0].update(lr=lrvalue, weight_decay=wdv)
                    if family == "RMSprop":
                        opt.param_groups[0]["momentum"] = momvalue
                    for q, gg in zip(pp[i], g):
                        q.grad = gg[i].clone()
                    opt.step()
                    for q, r in zip(pp[i], p):
                        assert torch.allclose(q, r[i], atol=1e-11, rtol=1e-10), (
                            family,
                            changing,
                            (q - r[i]).abs().max(),
                        )
    checks[family + "_native_constant_and_dynamic_lr_decay"] = True
# Independently expand moment sums under changing betas, rather than reusing the recurrence.
for family in ["Adam", "AdamW"]:
    p = [torch.tensor([[0.7, -0.2]], dtype=torch.float64, requires_grad=True)]
    mem = f.Memory(p)
    expected = p[0].detach().clone()
    gs = []
    b1s = []
    b2s = []
    for step, (b1, b2) in enumerate(
        zip([0.9, 0.1, 0.8, 0.3, 0.95, 0.5], [0.99, 0.9, 0.999, 0.95, 0.9999, 0.99])
    ):
        g = torch.tensor([[0.2 + step * 0.1, (-1) ** step * 0.3]], dtype=torch.float64)
        gs.append(g)
        b1s.append(b1)
        b2s.append(b2)
        mh = sum(
            (1 - b1s[i]) * math.prod(b1s[i + 1 :]) * gs[i] for i in range(step + 1)
        ) / (1 - math.prod(b1s))
        vh = sum(
            (1 - b2s[i]) * math.prod(b2s[i + 1 :]) * gs[i].square()
            for i in range(step + 1)
        ) / (1 - math.prod(b2s))
        expected = expected - 0.003 * mh / (vh.sqrt() + f.EPS)
        with torch.no_grad():
            p = mem.update(
                p,
                [g],
                family,
                *[torch.tensor([v], dtype=torch.float64) for v in [0.003, b1, b2, 0.0]]
            )
        assert torch.allclose(p[0], expected, atol=1e-12, rtol=1e-10)
    checks[family + "_varying_beta_explicit_weighted_sum"] = True
# Vary RMSprop alpha as well as momentum, independent scalar recurrence.
p = [torch.tensor([[0.7, -0.2]], dtype=torch.float64, requires_grad=True)]
mem = f.Memory(p)
expected = p[0].detach().clone()
variance = torch.zeros_like(expected)
buffer = torch.zeros_like(expected)
for t in range(6):
    rho = [0.9, 0.99, 0.999, 0.95, 0.9999, 0.99][t]
    mu = [0.5, 0.7, 0.2, 0.9, 0.3, 0.6][t]
    g = torch.tensor([[0.2 + t * 0.1, (-1) ** t * 0.3]], dtype=torch.float64)
    variance = rho * variance + (1 - rho) * g.square()
    buffer = mu * buffer + g / (variance.sqrt() + f.EPS)
    expected -= 0.003 * buffer
    with torch.no_grad():
        p = mem.update(
            p,
            [g],
            "RMSprop",
            *[torch.tensor([v], dtype=torch.float64) for v in [0.003, mu, rho, 0.0]]
        )
    assert torch.allclose(p[0], expected, atol=1e-12, rtol=1e-10)
checks["RMSprop_varying_alpha_momentum"] = True
for family in f.FAMILIES:
    for kind in ["gradient"]:
        pol = f.policy(family, kind, 338)
        rows, tr, lp, rw = f.rollout(
            family,
            kind,
            [1810000, 1810001],
            ["mixed", "single_middle"],
            pol,
            training=True,
            evaluation=(f.e.xd, f.e.yd),
            trace=True,
        )
        assert lp.shape == rw.shape == (2, 40) and len(tr) == 80
        (-lp.mean()).backward()
        assert all(torch.isfinite(q.grad).all() for q in pol.parameters())
        assert all(0 <= r["accuracy"] <= 1 for r in rows)
    checks[family + "_rollout_and_policy_gradients"] = True
# Float32 comparisons use identical represented coefficients on both sides.
errors = {}
torch.manual_seed(678)
lr, mu, rho, wd = [torch.tensor([v]) for v in [0.003, 0.9, 0.9999, 0.0]]
for family in f.FAMILIES:
    p = [torch.randn(1, 4, 3, requires_grad=True)]
    mem = f.Memory(p)
    q = torch.nn.Parameter(p[0][0].detach().clone())
    if family == "RMSprop":
        opt = torch.optim.RMSprop(
            [q], lr=float(lr[0]), momentum=float(mu[0]), alpha=float(rho[0])
        )
    else:
        opt = getattr(torch.optim, family)(
            [q], lr=float(lr[0]), betas=(float(mu[0]), float(rho[0])), weight_decay=0
        )
    mx = 0.0
    for step in range(100):
        g = torch.randn_like(p[0])
        q.grad = g[0].clone()
        opt.step()
        with torch.no_grad():
            p = mem.update(p, [g], family, lr, mu, rho, wd)
        mx = max(mx, float((p[0][0] - q).abs().max().detach()))
    errors[family] = mx
    assert mx < 1e-5, (family, mx)
checks["float32_100step_native_same_represented_controls_max_abs_error"] = errors
checks["precision_note"] = (
    "Native comparison uses the same float32-representable coefficients, important near beta2/alpha=1."
)
(OUT / "checks.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
