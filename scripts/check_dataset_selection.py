"""Check dataset isolation and smoke-test regular Fashion-MNIST/CIFAR-10 policies."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECK = r'''
import math, sys
sys.path.insert(0, "experiments/optimizer_families")
import run as r
e, b, f = r.e, r.b, r.f
expected = sys.argv[1]
assert e.DATASET == expected
assert type(e.data).__name__ == {"mnist": "MNIST", "fashion_mnist": "FashionMNIST", "cifar10": "CIFAR10"}[expected]
root = e.ROOT / "results"
if expected != "mnist":
    root /= expected
assert e.OUT == b.OUT == root / "recovery_study"
assert r.OUT == root / "optimizer_families"
assert r.OUT.parent / "recovery_study" == b.OUT
features = 3072 if expected == "cifar10" else 784
assert (len(e.x), len(e.xc), len(e.xd)) == ((40000 if expected == "cifar10" else 50000), 2000, 2000)
assert e.FEATURES == e.x.shape[1] == features and e.parameters([0])[0].shape == (1, 64, features)
assert 0 <= float(e.x.min()) and float(e.x.max()) <= 1 and e.y.dtype == e.torch.long
# Train/control/dev indices are disjoint slices of one fixed permutation.
assert len(set(e.ix[:len(e.x) + 4000].tolist())) == len(e.x) + 4000
evaluation = e.test_set()
assert evaluation[0].shape == (10000, features) and len(evaluation[1]) == 10000
print(expected, type(e.data).__name__, "split/test/output isolation OK", flush=True)
if expected != "mnist":
    e.torch.set_num_threads(1)
    rows, *_ = e.rollout(e.policy("gradient", 338), "gradient", [1900000], ["mixed"], evaluation=evaluation)
    for family in ["SGD"] + f.FAMILIES:
        if family != "SGD":
            rows, *_ = f.rollout(family, "gradient", [1900000], ["mixed"], pol=f.policy(family, "gradient", 338), evaluation=evaluation)
        row = rows[0]
        assert not row["failed"] and math.isfinite(row["eval_loss"])
        assert 0 <= row["accuracy"] <= 1
        print(f"{family}: loss={row['eval_loss']:.6f} accuracy={row['accuracy']:.4f} failed={row['failed']}", flush=True)
'''

for selection in [None, "mnist", "fashion_mnist", "cifar10", "", "unknown"]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("CS238_DATASET", None)
    if selection is not None:
        env["CS238_DATASET"] = selection
    result = subprocess.run(
        [sys.executable, "-c", CHECK, selection or "mnist"],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    if selection in ["", "unknown"]:
        assert result.returncode != 0 and "Unknown CS238_DATASET:" in result.stderr
        print(f"Invalid selection {selection!r}: rejected")
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        print(result.stdout, end="")
print("Dataset selection checks passed")
