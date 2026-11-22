"""Checks for the prospective-schedule and reactive-rule controls.

Run: MNIST_DATA_DIR=... python -m unittest experiments/reactive_rule/test_controls.py
"""

import math
import sys
import unittest
from pathlib import Path

import importlib.util

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/prospective_schedules"))
import common as c  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schedules = load("prospective_schedules_run", "experiments/prospective_schedules/run.py")
rule = load("reactive_rule_run", "experiments/reactive_rule/run.py")

c.setup("regular")
SEEDS, ORDERS = c.b.DEVSEEDS[:2], c.b.DEVORDERS[:2]


class Schedules(unittest.TestCase):
    def test_shapes_and_budget(self):
        self.assertEqual(schedules.lr_factor("cosine", 0), 1.0)
        self.assertEqual(schedules.lr_factor("step", 19), 1.0)
        self.assertAlmostEqual(schedules.lr_factor("step", 20), 0.1)
        self.assertAlmostEqual(schedules.lr_factor("step", 39), 0.01)
        self.assertAlmostEqual(schedules.lr_factor("warmup_cosine", 0), 0.25)
        self.assertEqual(schedules.lr_factor("warmup_cosine", 4), 1.0)
        self.assertTrue(all(0 < schedules.lr_factor(k, t) <= 1 for k in schedules.KINDS for t in range(40)))
        sgd = sum(len(schedules.candidates("SGD", k)) for k in schedules.KINDS)
        adaptive = sum(len(schedules.candidates("Adam", k)) for k in schedules.KINDS)
        self.assertLessEqual(8 * sgd, 4032)
        self.assertEqual(8 * adaptive, 3840)
        # Deterministic, distinct samples on the fixed grid axes.
        a, b = schedules.candidates("Adam", "step"), schedules.candidates("Adam", "step")
        self.assertEqual(a, b)
        self.assertEqual(len({tuple(x[k] for k in c.columns("Adam")) for x in a}), 160)

    def test_constant_schedule_matches_fixed_path(self):
        data = (c.e.xd, c.e.yd)
        for family in ["Adam", "RMSprop"]:
            fixed = c.fixed_config(family)
            ref, *_ = c.f.rollout(family, "gradient", SEEDS, ORDERS, fixed=fixed, evaluation=data)
            sched = schedules.make_schedule(family, "constant", dict(zip(c.columns(family), c.config_values(family, fixed))))
            got, _ = c.rollout(family, SEEDS, ORDERS, data, schedule=sched)
            self.assertEqual([r["eval_loss"] for r in ref], [r["eval_loss"] for r in got])


class ReactiveRule(unittest.TestCase):
    def test_rule_dynamics(self):
        r = rule.Rule("SGD", (0.3, 0.75, 0.0), tau=0.1, kappa=0.5, rho=0.5, gamma=1.0)
        obs = torch.zeros(2, 9)
        lr, mom, wd = r(0, obs)
        self.assertTrue(torch.equal(mom, torch.tensor([0.75, 0.75])))
        obs[:, 7] = torch.tensor([-0.5, 0.5])
        lr, mom, wd = r(1, obs)
        self.assertTrue(torch.allclose(mom, torch.tensor([0.375, 0.75])))
        self.assertTrue(torch.allclose(lr, torch.tensor([0.15, 0.3])))
        obs[:, 7] = 0.5
        lr, mom, wd = r(2, obs)
        self.assertTrue(torch.allclose(mom, torch.tensor([0.5625, 0.75])))
        self.assertEqual(wd, 0.0)

    def test_off_rule_equals_fixed(self):
        data = (c.e.xd, c.e.yd)
        for family in c.FAMILIES:
            base = rule.base(family)
            sched = schedules.make_schedule(family, "constant", dict(zip(c.columns(family), base)))
            want, _ = c.rollout(family, SEEDS, ORDERS, data, schedule=sched)
            got, tr = c.rollout(family, SEEDS, ORDERS, data, controller=rule.Rule(family, base, **rule.OFF), trace=True)
            self.assertEqual([r["eval_loss"] for r in want], [r["eval_loss"] for r in got], family)
            self.assertTrue(all(math.isclose(float(t["momentum"]), base[1], rel_tol=1e-6) for t in tr))

    def test_rule_fires_and_uses_memory_cosine(self):
        family = "SGD"
        base = rule.base(family)
        _, tr = c.rollout(family, SEEDS, ORDERS, (c.e.xd, c.e.yd),
                          controller=rule.Rule(family, base, tau=0.5, kappa=0.5, rho=0.5, gamma=0.0), trace=True)
        self.assertTrue(any(float(t["momentum"]) < base[1] - 1e-6 for t in tr))
        self.assertEqual(len(rule.candidates()), rule.DESIGN["candidates"])
        self.assertLessEqual(rule.DESIGN["budget_episodes"], 3840)


if __name__ == "__main__":
    unittest.main()
