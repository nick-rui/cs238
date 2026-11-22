"""CPU timings using the regular experiment engines, without changing artifacts."""

import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CS238_DATASET"] = "mnist"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/optimizer_families"))
import run as runner
import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST

e, b, f = runner.e, runner.b, runner.f
original_parameters = e.parameters
original_nn = b.nn
results = []


def parameters(seeds):
    nets = []
    for seed in seeds:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            nets.append(nn.Sequential(nn.Linear(e.x.shape[1], 64), nn.ReLU(), nn.Linear(64, 10)))
    return [
        torch.stack([list(net.parameters())[i].detach() for net in nets]).requires_grad_()
        for i in range(4)
    ]


def timed(dataset, family, operation, classifiers, function):
    start = time.perf_counter()
    rows = function()
    seconds = time.perf_counter() - start
    assert len(rows) == classifiers
    record = dict(dataset=dataset, family=family, operation=operation,
                  classifiers=classifiers, threads=torch.get_num_threads(),
                  seconds=seconds, seconds_per_episode=seconds / classifiers,
                  failed_episodes=sum(row["failed"] for row in rows))
    results.append(record)
    print(json.dumps(record), flush=True)


for name, dataset_class in [("mnist", MNIST), ("fashion_mnist", FashionMNIST), ("cifar10", CIFAR10)]:
    train = dataset_class(str(e.DATA_DIR), train=True, download=True)
    test = dataset_class(str(e.DATA_DIR), train=False, download=True)
    xx = torch.as_tensor(np.asarray(train.data)).reshape(len(train), -1).float() / 255
    yy = torch.as_tensor(np.asarray(train.targets)).long()
    # CIFAR-10 has 50k rather than 60k training examples; reserve the same probes.
    ntrain = 50000 if len(train) == 60000 else 40000
    ix = torch.randperm(len(train), generator=torch.Generator().manual_seed(238))
    e.x, e.y = xx[ix[:ntrain]], yy[ix[:ntrain]]
    e.xc, e.yc = xx[ix[ntrain:ntrain + 2000]], yy[ix[ntrain:ntrain + 2000]]
    e.xd, e.yd = xx[ix[ntrain + 2000:ntrain + 4000]], yy[ix[ntrain + 2000:ntrain + 4000]]
    e.ink = e.x.sum(1)
    evaluation = (torch.as_tensor(np.asarray(test.data)).reshape(len(test), -1).float() / 255,
                  torch.as_tensor(np.asarray(test.targets)).long())
    del xx, yy, train, test
    e.parameters = parameters
    b.nn = SimpleNamespace(
        Sequential=nn.Sequential, ReLU=nn.ReLU, functional=nn.functional,
        Linear=lambda inputs, outputs: nn.Linear(e.x.shape[1] if inputs == 784 else inputs, outputs),
    )
    b.class_ids = [torch.where(e.y == digit)[0].numpy() for digit in range(10)]
    b.stream.cache_clear()
    for family in ["SGD", "Adam"]:
        torch.set_num_threads(2 if family == "SGD" else 1)
        pol = e.policy("gradient", 338) if family == "SGD" else f.policy(family, "gradient", 338)
        opt = torch.optim.Adam(pol.parameters(), lr=0.003)

        def rollout(seeds, orders, training=False, evaluation=None, fixed=None, trace=False):
            if family == "SGD":
                return e.rollout(pol, "gradient", seeds, orders, training=training,
                                 evaluation=evaluation, trace=trace)
            return f.rollout(family, "gradient", seeds, orders, pol, training=training,
                             evaluation=evaluation, fixed=fixed, trace=trace)

        def training():
            seeds = [5000000 + 338 * 10000 + i for i in range(8)]
            rows, _, logp, rewards, *_ = rollout(seeds, b.TRAIN * 2, training=True)
            returns = rewards.flip(1).cumsum(1).flip(1)
            adv = returns - returns[torch.tensor([4, 5, 6, 7, 0, 1, 2, 3])]
            opt.zero_grad()
            (-(logp * adv.detach()).mean()).backward()
            assert all(torch.isfinite(q.grad).all() for q in pol.parameters())
            nn.utils.clip_grad_norm_(pol.parameters(), 1)
            opt.step()
            return rows

        timed(name, family, "train", 8, training)
        timed(name, family, "dev_policy", 8,
              lambda: rollout(b.DEVSEEDS[:8], b.DEVORDERS[:8], evaluation=(e.xd, e.yd))[0])
        config = (dict(family="SGD", lr=0.1, momentum=0.5, weight_decay=0.0001)
                  if family == "SGD" else dict(lr=0.003, beta1=0.9, beta2=0.999, weight_decay=0.0001))
        torch.set_num_threads(1)
        timed(name, family, "tune", 8,
              lambda: [b.baseline(config, seed, order, (e.xd, e.yd))[0]
                       for seed, order in zip(b.DEVSEEDS[:8], b.DEVORDERS[:8])]
              if family == "SGD" else rollout(b.DEVSEEDS[:8], b.DEVORDERS[:8],
                                              evaluation=(e.xd, e.yd), fixed=config)[0])
        torch.set_num_threads(2 if family == "SGD" else 1)
        timed(name, family, "eval_policy", 4,
              lambda: rollout(b.TESTSEEDS[:4], ["single_middle"] * 4,
                              evaluation=evaluation, trace=True)[0])
        timed(name, family, "eval_fixed", 1 if family == "SGD" else 4,
              lambda: [b.baseline(config, b.TESTSEEDS[0], "single_middle", evaluation, True)[0]]
              if family == "SGD" else rollout(b.TESTSEEDS[:4], ["single_middle"] * 4,
                                              evaluation=evaluation, fixed=config, trace=True)[0])

e.parameters, b.nn = original_parameters, original_nn


def seconds(dataset, family, operation):
    return next(r["seconds"] for r in results
                if (r["dataset"], r["family"], r["operation"]) == (dataset, family, operation))


projections = {}
for dataset in ["mnist", "fashion_mnist", "cifar10"]:
    families = {}
    for family in ["SGD", "Adam", "AdamW", "RMSprop"]:
        proxy = "SGD" if family == "SGD" else "Adam"
        grid = 6 * 6 * 4 if family == "SGD" else len(runner.configs(family))
        families[family] = dict(
            grid_candidates=grid,
            tune_seconds=grid * seconds(dataset, proxy, "tune"),
            train_seconds=1000 * seconds(dataset, proxy, "train") + 32 * seconds(dataset, proxy, "dev_policy"),
            eval_seconds=100 * seconds(dataset, proxy, "eval_policy")
            + (200 if family == "SGD" else 50) * seconds(dataset, proxy, "eval_fixed"),
        )
    # Match the planned driver: tunes parallel; six family seeds parallel with
    # a sequential two-seed SGD worker; evaluations parallel. No contention model.
    stage_seconds = dict(
        tune=max(v["tune_seconds"] for v in families.values()),
        train=max(families["SGD"]["train_seconds"],
                  *(families[family]["train_seconds"] / 2 for family in f.FAMILIES)),
        evaluate=max(v["eval_seconds"] for v in families.values()),
    )
    projections[dataset] = dict(families=families, stage_seconds=stage_seconds,
                                wall_hours=sum(stage_seconds.values()) / 3600)
output = dict(timings=results, projections=projections,
              caveats="One sample per operation; no warmup. Adam proxies AdamW/RMSprop. "
              "Representative stable fixed config, not full grid. No concurrency contention, "
              "download/import/analysis/serialization costs. CIFAR-10 split 40k/2k/2k; raw pixels /255.")
path = ROOT / "results/cifar10_timing.json"
assert not path.exists(), f"Refusing to overwrite {path}"
path.write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(dict(projections=projections), indent=2), flush=True)
