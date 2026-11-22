"""GPU-path checks: MLP on CUDA vs CPU (MNIST, saved policies), and the batched
grouped-convolution CNN vs independent nn.Module CNNs. Writes gpu_checks.json."""

import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cnn_cifar10"
# Most episodes agree to float rounding (median loss rel. error ~1e-7); a few
# trajectories amplify rounding over 200 steps, as CPU-vs-CPU replays on another
# VM do (this VM's CPU differs from committed rows by up to 0.14 pp). Hence a tight
# median bound plus a loose worst-case bound.
TOL = dict(accuracy_abs=2.5e-3, loss_rel=0.02, return_abs=0.05, median_loss_rel=1e-5,
           logits_abs=1e-4)


def worker():
    sys.path.insert(0, str(ROOT / "experiments/optimizer_families"))
    import run as runner
    import torch

    e, b, f = runner.e, runner.b, runner.f
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    testset = ((test.data.reshape(-1, 784).float() / 255).to(e.DEVICE), test.targets.to(e.DEVICE))
    seeds, order = b.TESTSEEDS[:4], "single_middle"
    rows = []
    sgd = json.loads((ROOT / "results/recovery_study/baseline_configs.json").read_text())["SGD"]
    for s in seeds:
        r, _ = b.baseline(sgd, s, order, testset)
        rows.append(dict(r, family="SGD", variant="fixed"))
    for ps in [338, 339]:
        saved = torch.load(ROOT / f"results/recovery_study/gradient_{ps}/best.pt", weights_only=True)
        pol = e.policy("gradient", ps)
        pol.load_state_dict(saved["policy"])
        rr, *_ = e.rollout(pol, "gradient", seeds, [order] * 4, evaluation=testset)
        rows += [dict(r, family="SGD", variant="gradient", policy_seed=ps) for r in rr]
        saved = torch.load(ROOT / f"results/optimizer_families/Adam_gradient_{ps}/best.pt", weights_only=True)
        pol = f.policy("Adam", "gradient", ps)
        pol.load_state_dict(saved["policy"])
        rr, *_ = f.rollout("Adam", "gradient", seeds, [order] * 4, pol, evaluation=testset)
        rows += [dict(r, family="Adam", variant="gradient", policy_seed=ps) for r in rr]
    adam = json.loads((ROOT / "results/optimizer_families/Adam_baseline.json").read_text())["config"]
    rr, *_ = f.rollout("Adam", "gradient", seeds, [order] * 4, fixed=adam, evaluation=testset)
    rows += [dict(r, family="Adam", variant="fixed") for r in rr]
    for r in rows:
        r.pop("per_class_accuracy", None)
    print("ROWS" + json.dumps(rows))


def cnn_check():
    sys.path.insert(0, str(ROOT / "experiments/recovery_study"))
    import engine as e
    import torch

    seeds = [11, 12, 13]
    p = e.parameters(seeds)
    xx = torch.rand(3, 50, 3072, device=e.DEVICE)
    batched = e.forward(p, xx)
    nets = []
    for s in seeds:
        torch.manual_seed(s)
        nets.append(e.CNN().to(e.DEVICE))
    single = torch.stack([net(xx[i]) for i, net in enumerate(nets)])
    g = torch.autograd.grad(e.losses(p, xx, torch.zeros(3, 50, dtype=torch.long, device=e.DEVICE)).sum(), p)
    gs = []
    for i, net in enumerate(nets):
        loss = torch.nn.functional.cross_entropy(net(xx[i]), torch.zeros(50, dtype=torch.long, device=e.DEVICE))
        gs.append(torch.autograd.grad(loss, list(net.parameters())))
    grad_err = max(float((g[k][i] - gs[i][k]).abs().max()) for i in range(3) for k in range(8))
    return dict(cnn_parameters=sum(q[0].numel() for q in p),
                logits_max_abs_error=float((batched - single).abs().max()),
                grad_max_abs_error=grad_err)


def run(device, model="mlp"):
    env = dict(os.environ, CS238_DEVICE=device, CS238_MODEL=model, CS238_DATASET="mnist")
    out = subprocess.run([sys.executable, __file__, "worker"], env=env, check=True,
                         capture_output=True, text=True).stdout
    return json.loads(next(l for l in out.splitlines() if l.startswith("ROWS"))[4:])


if __name__ == "__main__":
    if sys.argv[1:] == ["worker"]:
        worker()
        sys.exit()
    if sys.argv[1:] == ["cnn"]:
        print("CNN" + json.dumps(cnn_check()))
        sys.exit()
    import csv
    cpu, gpu = run("cpu"), run("cuda")
    diffs = []
    for c, g in zip(cpu, gpu):
        assert (c["seed"], c["family"], c["variant"]) == (g["seed"], g["family"], g["variant"])
        diffs.append(dict(family=c["family"], variant=c["variant"], policy_seed=c.get("policy_seed", -1),
                          seed=c["seed"], cpu_accuracy=c["accuracy"], gpu_accuracy=g["accuracy"],
                          accuracy_abs=abs(c["accuracy"] - g["accuracy"]),
                          loss_rel=abs(c["eval_loss"] - g["eval_loss"]) / c["eval_loss"],
                          return_abs=(abs(c["return_value"] - g["return_value"])
                                      if "return_value" in c else None)))
    # Committed CPU evaluation rows for the same episodes (replay check on CPU and GPU).
    committed = []
    for family in ["SGD", "Adam"]:
        for r in csv.DictReader((ROOT / f"results/optimizer_families/{family}_evaluation.csv").open()):
            if r["order"] == "single_middle" and int(r["seed"]) in range(1900000, 1900004):
                committed.append(r)
    replay = []
    for c, g in zip(cpu, gpu):
        ps = c.get("policy_seed", -1)
        ref = next(r for r in committed if r["family"] == c["family"] and r["variant"] == c["variant"]
                   and int(r["policy_seed"]) == ps and int(r["seed"]) == c["seed"])
        replay.append(dict(family=c["family"], variant=c["variant"], policy_seed=ps, seed=c["seed"],
                           committed=float(ref["accuracy"]), cpu=c["accuracy"], gpu=g["accuracy"]))
    env = dict(os.environ, CS238_DEVICE="cuda", CS238_MODEL="cnn")
    out = subprocess.run([sys.executable, __file__, "cnn"], env=env, check=True,
                         capture_output=True, text=True).stdout
    cnn = json.loads(next(l for l in out.splitlines() if l.startswith("CNN"))[3:])
    worst = dict(accuracy_abs=max(d["accuracy_abs"] for d in diffs),
                 loss_rel=max(d["loss_rel"] for d in diffs),
                 median_loss_rel=sorted(d["loss_rel"] for d in diffs)[len(diffs) // 2],
                 return_abs=max(d["return_abs"] for d in diffs if d["return_abs"] is not None),
                 replay_gpu_vs_committed_abs=max(abs(r["gpu"] - r["committed"]) for r in replay),
                 replay_cpu_vs_committed_abs=max(abs(r["cpu"] - r["committed"]) for r in replay))
    passed = (worst["accuracy_abs"] <= TOL["accuracy_abs"] and worst["loss_rel"] <= TOL["loss_rel"]
              and worst["return_abs"] <= TOL["return_abs"]
              and worst["median_loss_rel"] <= TOL["median_loss_rel"]
              and cnn["logits_max_abs_error"] <= TOL["logits_abs"])
    result = dict(passed=passed, tolerances=TOL, worst=worst, cnn_batched_vs_module=cnn,
                  mlp_cpu_vs_gpu=diffs, replay=replay,
                  episodes="MNIST MLP, single_middle, test seeds 1900000-1900003, full 10k test set")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gpu_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=passed, worst=worst, cnn=cnn), indent=2))
