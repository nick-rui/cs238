"""Replay representative saved evaluations after source packaging changes."""

import argparse, csv, gzip, importlib.util, json, math, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/optimizer_families"))
import family_engine as f

b = f.base
e = f.e
torch.set_num_threads(1)
p = argparse.ArgumentParser()
p.add_argument("--suite", choices=["regular", "extreme"], required=True)
args = p.parse_args()
if args.suite == "extreme":
    spec = importlib.util.spec_from_file_location(
        "stress_streams", ROOT / "experiments/extreme_orderings/run.py"
    )
    stress = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stress)
    seed, order = 1920000, "single_late"
    folder = "extreme_orderings"
    policy_variant = "policy"
else:
    seed, order = 1900000, "single_middle"
    folder = "optimizer_families"
    policy_variant = "gradient"
test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
testset = (test.data.reshape(-1, 784).float() / 255, test.targets)
checks = []
for family in ["SGD", "Adam", "AdamW", "RMSprop"]:
    with (ROOT / "results" / folder / f"{family}_evaluation.csv").open() as file:
        expected = list(csv.DictReader(file))
    trace_path = ROOT / "results" / folder / f"{family}_traces.csv"
    with (
        trace_path.open()
        if trace_path.exists()
        else gzip.open(str(trace_path) + ".gz", "rt")
    ) as file:
        expected_traces = list(csv.DictReader(file))
    config = json.loads(
        (ROOT / "results/optimizer_families" / f"{family}_baseline.json").read_text()
    )["config"]
    for ps in [-1, 338, 339]:
        variant = "fixed" if ps == -1 else policy_variant
        ref = next(
            r
            for r in expected
            if r["order"] == order
            and int(r["seed"]) == seed
            and r["variant"] == variant
            and int(r["policy_seed"]) == ps
        )
        if ps == -1 and family == "SGD":
            row, tr = b.baseline(config, seed, order, testset, True)
        else:
            pol = None
            if ps != -1:
                ckpt = (
                    ROOT
                    / "results"
                    / ("recovery_study" if family == "SGD" else "optimizer_families")
                    / (
                        f"gradient_{ps}"
                        if family == "SGD"
                        else f"{family}_gradient_{ps}"
                    )
                    / "best.pt"
                )
                saved = torch.load(ckpt, weights_only=True)
                pol = (
                    e.policy("gradient", ps)
                    if family == "SGD"
                    else f.policy(family, "gradient", ps)
                )
                pol.load_state_dict(saved["policy"])
                pol.eval()
            if family == "SGD":
                rr, tr, *_ = e.rollout(
                    pol, "gradient", [seed], [order], evaluation=testset, trace=True
                )
            else:
                rr, tr, *_ = f.rollout(
                    family,
                    "gradient",
                    [seed],
                    [order],
                    pol=pol,
                    fixed=config if ps == -1 else None,
                    evaluation=testset,
                    trace=True,
                )
            row = rr[0]
        errors = {
            k: abs(float(row[k]) - float(ref[k]))
            for k in ["accuracy", "eval_loss", "path_log_loss"]
        }
        assert errors["accuracy"] < 1e-6, (family, ps, errors)
        assert max(errors.values()) < 3e-5, (family, ps, errors)
        assert str(row["failed"]) == ref["failed"]
        for trace in tr:
            et = next(
                r
                for r in expected_traces
                if r["order"] == order
                and int(r["seed"]) == seed
                and r["variant"] == variant
                and int(r["policy_seed"]) == ps
                and int(r["step"]) == int(trace["step"])
            )
            for k in ["lr", "momentum", "weight_decay"]:
                assert math.isclose(
                    float(trace[k]), float(et[k]), rel_tol=2e-4, abs_tol=3e-6
                ), (family, ps, trace["step"], k, trace[k], et[k])
        checks.append(
            dict(
                family=family,
                policy_seed=ps,
                metric_absolute_errors=errors,
                action_traces_match=True,
            )
        )
print(json.dumps(dict(suite=args.suite, checks=checks), indent=2))
