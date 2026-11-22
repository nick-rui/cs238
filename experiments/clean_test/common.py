"""Shared helpers for evaluation-only controls on saved policies.

Nothing here trains or selects anything. The helpers

* load the saved checkpoints, fixed configurations and open-loop schedules;
* run the original engines with the original episode seeds, orderings, rollout
  batch size (four episodes) and thread counts;
* capture each rollout's final classifier parameters (by wrapping the engine's
  ``forward`` for the duration of the call) so the same trained classifiers are
  scored on both the official 10,000-image test split and the 6,000 unused
  training-split images, without touching the engines' code;
* wrap a policy so its observations can be recorded or have their feedback
  columns replaced (the scrambled-feedback control).

Importing this module for the extreme suite executes
``experiments/extreme_orderings/run.py``, which swaps the stream generator
process-wide. Use one suite per process.
"""

import csv
import gzip
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "experiments/optimizer_families"))
import family_engine as f  # noqa: E402

b, e = f.base, f.e
FAMILIES = ["SGD", "Adam", "AdamW", "RMSprop"]
POLICY_SEEDS = [338, 339, 340, 341, 342]
ORIGINAL_SEEDS = [338, 339]
CLEAN_START, CLEAN_END = 54000, 60000
# Observation columns that carry feedback about the realized trajectory:
# relative control loss, recent improvement, log gradient norm, and the two
# gradient cosines. Settings and elapsed time stay real.
FEEDBACK_COLUMNS = {"SGD": [4, 5, 6, 7, 8], "adaptive": [5, 6, 7, 8, 9]}
FEEDBACK_NAMES = ["log_relative_loss", "recent_improvement", "log_grad_norm",
                  "grad_memory_cosine", "grad_previous_cosine"]
METHODS = ["fixed", "fixed_retuned", "policy", "warm", "retrained", "open_loop"]


def feedback_columns(family):
    return FEEDBACK_COLUMNS["SGD" if family == "SGD" else "adaptive"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def suite_design(suite):
    """Orders and episode seeds; installs the extreme stream generator if needed."""
    if suite == "regular":
        return list(b.ORDERS), list(b.TESTSEEDS)
    stress = load_module("stress_streams", "experiments/extreme_orderings/run.py")
    return list(stress.ORDERS), list(stress.SEEDS)


def official_testset():
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    return test.data.reshape(-1, 784).float() / 255, test.targets


def clean_indices():
    return e.ix[CLEAN_START:CLEAN_END]


def clean_testset():
    idx = clean_indices()
    return e.X[idx], e.Y[idx]


def threads_for(suite, method, family, seed):
    """Thread counts used by the runs that produced the saved records."""
    if suite == "regular" and family == "SGD":
        if method in ["fixed", "warm"] or (method == "policy" and seed in ORIGINAL_SEEDS):
            return 2
    return 1


def units():
    rows = []
    for family in FAMILIES:
        rows.append(("regular", "fixed", family, -1))
        rows += [("regular", "policy", family, s) for s in POLICY_SEEDS]
        rows += [("regular", "warm", family, s) for s in POLICY_SEEDS]
        rows += [("regular", "open_loop", family, s) for s in ORIGINAL_SEEDS]
        rows.append(("extreme", "fixed", family, -1))
        rows.append(("extreme", "fixed_retuned", family, -1))
        rows += [("extreme", "policy", family, s) for s in POLICY_SEEDS]
        rows += [("extreme", "retrained", family, s) for s in POLICY_SEEDS]
        rows += [("extreme", "warm", family, s) for s in POLICY_SEEDS]
    return [dict(suite=s, method=m, family=fam, seed=ps, threads=threads_for(s, m, fam, ps))
            for s, m, fam, ps in rows]


def unit_name(suite, method, family, seed):
    return f"{suite}/{method}/{family}_{seed}" if seed >= 0 else f"{suite}/{method}/{family}"


def fixed_config(family, method):
    if method == "fixed_retuned":
        return json.loads((RESULTS / f"extreme_retrained/{family}_baseline.json").read_text())["config"]
    if family == "SGD":
        return json.loads((RESULTS / "recovery_study/baseline_configs.json").read_text())["SGD"]
    return json.loads((RESULTS / f"optimizer_families/{family}_baseline.json").read_text())["config"]


def checkpoint_path(method, family, seed):
    if method == "policy":
        if seed in ORIGINAL_SEEDS:
            return RESULTS / ("recovery_study" if family == "SGD" else "optimizer_families") / (
                f"gradient_{seed}" if family == "SGD" else f"{family}_gradient_{seed}") / "best.pt"
        return RESULTS / f"additional_seeds/training/{family}_gradient_{seed}/best.pt"
    if method == "warm":
        return RESULTS / "warm_start" / (f"SGD/gradient_{seed}" if family == "SGD" else f"{family}_gradient_{seed}") / "best.pt"
    if method == "retrained":
        folder = "extreme_retrained" if seed in ORIGINAL_SEEDS else "extreme_retrained_five_seeds"
        return RESULTS / folder / f"{family}_gradient_{seed}" / "best.pt"
    raise ValueError(method)


def load_policy(method, family, seed):
    saved = torch.load(checkpoint_path(method, family, seed), weights_only=True)
    pol = e.policy("gradient", seed) if family == "SGD" else f.policy(family, "gradient", seed)
    pol.load_state_dict(saved["policy"])
    pol.eval()
    return pol, int(saved["update"])


def open_loop_schedule(family, seed):
    path = RESULTS / f"open_loop/regular/{family}_gradient_{seed}/schedule.csv"
    keys = ["lr", "momentum"] + ([] if family == "SGD" else ["second_decay"]) + ["weight_decay"]
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["step"]) for r in rows] == list(range(5, 201, 5))
    return [[float(r[k]) for k in keys] for r in rows]


class FeedbackPolicy(nn.Module):
    """Pass-through policy wrapper that records, and optionally replaces, feedback.

    ``start(n, donor)`` must be called before every rollout. ``donor`` is None
    (real feedback) or a float tensor [n, 40, 5] whose values replace the
    feedback columns at each decision. The real observation is always recorded.
    """

    def __init__(self, pol, family):
        super().__init__()
        self.pol = pol
        self.columns = feedback_columns(family)
        self.donor = None
        self.records = []

    def start(self, donor=None):
        self.donor = donor
        self.records = []

    def forward(self, obs):
        t = len(self.records)
        self.records.append(obs.detach().clone())
        if self.donor is not None:
            obs = obs.clone()
            obs[:, self.columns] = self.donor[:, t].to(obs.dtype)
        return self.pol(obs)

    def observations(self):
        """Real observations fed during the last rollout, as [n, 40, inputs]."""
        assert len(self.records) == 40, len(self.records)
        return torch.stack(self.records, 1)


class FinalParameters:
    """Capture the parameters seen by the engine's last ``forward`` call.

    Both engines' rollouts end with a loop over the evaluation images that calls
    ``engine.forward(p, ...)`` with the final classifier parameters, so the last
    captured ``p`` is the final classifier batch.
    """

    def __enter__(self):
        self.original = e.forward
        self.p = None

        def forward(p, xx):
            self.p = p
            return self.original(p, xx)

        e.forward = forward
        return self

    def __exit__(self, *exc):
        e.forward = self.original
        return False


def score_parameters(p, testset, failed, mask_failures):
    """Final loss/accuracy with the engine's own chunking and failure convention."""
    ex, ey = testset
    n = len(p[0])
    with torch.no_grad():
        total = torch.zeros(n)
        correct = torch.zeros(n)
        for start in range(0, len(ex), 500):
            xx = ex[start:start + 500].unsqueeze(0).expand(n, -1, -1)
            yy = ey[start:start + 500].unsqueeze(0).expand(n, -1)
            z = e.forward(p, xx)
            total += nn.functional.cross_entropy(z.reshape(-1, 10), yy.reshape(-1), reduction="none").reshape(n, -1).sum(1)
            correct += (z.argmax(2) == yy).sum(1)
        loss = total / len(ex)
        acc = correct / len(ex)
        if mask_failures:
            # Family engine: failed episodes are scored as loss 100, accuracy 0.
            active = ~torch.tensor(failed)
            loss = torch.where(active, loss, torch.full_like(loss, 100.0))
            acc = torch.where(active, acc, torch.zeros_like(acc))
    return loss.tolist(), acc.tolist()


def evaluate_batch(unit, seeds, orders, official, clean, pol=None, config=None, schedule=None, warm=None):
    """Run one original rollout batch and score it on both test sets.

    Returns rows with the engine's own official metrics, an independent
    recomputation of the official metrics from the captured parameters, and
    the clean-split metrics.
    """
    family = unit["family"]
    if family == "SGD" and config is not None:
        rows = []
        for seed, order in zip(seeds, orders):
            r_off, _ = b.baseline(config, seed, order, official, True)
            r_clean, _ = b.baseline(config, seed, order, clean, True)
            assert r_off["failed"] == r_clean["failed"], (seed, order)
            rows.append(dict(seed=seed, order=order, failed=r_off["failed"],
                             official_accuracy=r_off["accuracy"], official_eval_loss=r_off["eval_loss"],
                             official_path_log_loss=r_off["path_log_loss"],
                             clean_accuracy=r_clean["accuracy"], clean_eval_loss=r_clean["eval_loss"],
                             official_recomputed_accuracy=r_off["accuracy"]))
        return rows
    with FinalParameters() as capture:
        if family == "SGD":
            rr, *_ = e.rollout(pol, "gradient", seeds, orders, evaluation=official, trace=True, warm=warm,
                               schedules=None if schedule is None else {o: schedule for o in orders})
        else:
            rr, *_ = f.rollout(family, "gradient", seeds, orders, pol, evaluation=official, trace=True,
                               fixed=config, warm=warm, schedule=schedule)
    failed = [bool(r["failed"]) for r in rr]
    mask = family != "SGD"
    off_loss, off_acc = score_parameters(capture.p, official, failed, mask)
    clean_loss, clean_acc = score_parameters(capture.p, clean, failed, mask)
    rows = []
    for i, r in enumerate(rr):
        # The capture must have seen the final parameters: recomputed official
        # accuracy has to equal the engine's own value.
        assert abs(off_acc[i] - r["accuracy"]) < 1e-6, (unit, r["seed"], off_acc[i], r["accuracy"])
        assert abs(off_loss[i] - r["eval_loss"]) < 1e-4 * max(1.0, abs(r["eval_loss"])), (unit, r["seed"])
        rows.append(dict(seed=r["seed"], order=r["order"], failed=r["failed"],
                         official_accuracy=r["accuracy"], official_eval_loss=r["eval_loss"],
                         official_path_log_loss=r["path_log_loss"],
                         clean_accuracy=clean_acc[i], clean_eval_loss=clean_loss[i],
                         official_recomputed_accuracy=off_acc[i]))
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_csv(path):
    path = Path(path)
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        path = path.with_name(path.name + ".gz")
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
        return list(csv.DictReader(handle))


def is_true(value):
    return str(value).lower() in ["true", "1"]


def committed_reference(suite, method, family, seed):
    """Committed official-test rows for this unit, keyed by (order, seed)."""
    if method == "open_loop":
        rows = read_csv(RESULTS / f"open_loop/regular/{family}_gradient_{seed}/evaluation.csv")
    elif method == "warm":
        rows = [r for r in read_csv(RESULTS / f"warm_start/{family}_{suite}_evaluation.csv")
                if r["variant"] == "warm" and int(r["policy_seed"]) == seed]
    elif method == "fixed_retuned":
        rows = [r for r in read_csv(RESULTS / f"extreme_retrained/{family}_evaluation.csv") if r["variant"] == "fixed_retuned"]
    elif method == "retrained":
        folder = "extreme_retrained" if seed in ORIGINAL_SEEDS else "extreme_retrained_five_seeds"
        rows = [r for r in read_csv(RESULTS / f"{folder}/{family}_evaluation.csv")
                if r["variant"] == "policy_retrained" and int(r["policy_seed"]) == seed]
    elif method in ["fixed", "policy"] and (method == "fixed" or seed in ORIGINAL_SEEDS):
        source = "optimizer_families" if suite == "regular" else "extreme_orderings"
        variants = ["fixed"] if method == "fixed" else ["gradient", "policy"]
        rows = [r for r in read_csv(RESULTS / f"{source}/{family}_evaluation.csv")
                if r["variant"] in variants and int(r["policy_seed"]) == seed]
    else:
        rows = read_csv(RESULTS / f"additional_seeds/{suite}/{family}_{seed}/evaluation.csv")
    return {(r["order"], int(r["seed"])): r for r in rows}


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def sample_sd(values):
    values = list(values)
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


class Slots:
    """CPU-thread budget; a job takes all the slots it needs at once (no deadlock)."""

    def __init__(self, capacity):
        import threading
        self.free = capacity
        self.capacity = capacity
        self.cond = threading.Condition()

    def acquire(self, n):
        assert n <= self.capacity
        with self.cond:
            self.cond.wait_for(lambda: self.free >= n)
            self.free -= n

    def release(self, n):
        with self.cond:
            self.free += n
            self.cond.notify_all()
