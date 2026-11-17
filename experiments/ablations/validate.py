"""Check ablation dimensions, initialization, fixed controls and gradients."""

import json
import math
import torch
import run as a

torch.set_num_threads(1)
for family in ["SGD", "Adam"]:
    baseline = json.loads((a.SOURCE / f"{family}_baseline.json").read_text())["config"]
    for kind in a.KINDS:
        pol = a.policy(family, kind, 338)
        full_inputs, full_outputs = (9, 3) if family == "SGD" else (10, 4)
        assert pol[0].in_features == full_inputs - (3 if kind == "no_feedback" else 0)
        assert pol[-1].out_features == {"lr_only": 1, "fixed_memory": 2}.get(kind, full_outputs)
        assert not pol[-1].weight.count_nonzero()
        if kind != "no_feedback":
            assert not pol[-1].bias.count_nonzero()
        observations = []
        hook = pol.register_forward_pre_hook(lambda module, args: observations.append(args[0].clone()))
        rows, traces, lp, rewards = a.rollout(family, kind, [1810000, 1810001], ["mixed", "single_middle"], pol, training=True, trace=True, evaluation=(a.e.xd, a.e.yd))
        hook.remove()
        assert lp.shape == rewards.shape == (2, 40)
        assert len(rows) == 2 and len(traces) == 80
        assert all(torch.isfinite(obs).all() for obs in observations)
        (-(lp * rewards.detach()).mean()).backward()
        assert all(torch.isfinite(p.grad).all() for p in pol.parameters())
        for index, r in enumerate(traces):
            if family == "SGD":
                for field, column, scale in [
                    ("obs_log_grad_norm", 6, 10),
                    ("obs_grad_momentum_cosine", 7, 1),
                    ("obs_grad_previous_cosine", 8, 1),
                ]:
                    if kind == "no_feedback":
                        assert r[field] is None
                    else:
                        expected = float(observations[index // 2][index % 2, column] * scale)
                        assert r[field] == expected
            if kind in ["lr_only", "fixed_memory"]:
                expected = baseline["momentum" if family == "SGD" else "beta1"]
                assert math.isclose(r["momentum"], expected, abs_tol=1e-7)
                if family != "SGD":
                    assert math.isclose(r["second_decay"], baseline["beta2"], abs_tol=1e-7)
            if kind == "lr_only":
                assert r["weight_decay"] == baseline["weight_decay"]
        print("PASSED", family, kind, "dimensions, initialization, observations, held controls, finite gradients", flush=True)
print("ABLATION CHECKS PASSED")
