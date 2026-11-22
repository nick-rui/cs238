"""Checks for the random baselines (python -m unittest experiments/extreme_baselines/test_baselines.py)."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

# Load under another name: experiments/extreme_orderings/retrain.py imports its own `run`.
_spec = importlib.util.spec_from_file_location("extreme_baselines_run", Path(__file__).resolve().parent / "run.py")
xb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xb)


class RandomBaselineTest(unittest.TestCase):
    def test_draws_depend_only_on_episode_and_seed(self):
        a = xb.RandomActions("Adam", [1920000, 1920001], ["mixed", "staircase"], 3)
        b = xb.RandomActions("Adam", [1920001], ["staircase"], 3)
        torch.testing.assert_close(a.a[:, 1], b.a[:, 0], rtol=0, atol=0)
        other = xb.RandomActions("Adam", [1920001], ["staircase"], 4)
        self.assertFalse(torch.equal(other.a, b.a))
        self.assertEqual(tuple(a.a.shape), (40, 2, 4))
        self.assertLess(a.a.abs().max().item(), 1)
        out = torch.cat([a(torch.zeros(2, 10)) for _ in range(40)])
        torch.testing.assert_close(out.tanh(), a.a.reshape(-1, 4), rtol=0, atol=1e-5)

    def test_uniform_moments(self):
        a = xb.RandomActions("SGD", list(range(1920000, 1920020)), ["mixed"] * 20, 0).a.numpy().ravel()
        self.assertAlmostEqual(a.mean(), 0, delta=0.05)
        self.assertAlmostEqual(a.var(), 1 / 3, delta=0.03)

    def test_sgd_random_schedule_uses_policy_mapping(self):
        pol = xb.RandomActions("SGD", [1920000], ["half_block"], 2)
        draws = pol.a[:, 0].clone()
        _, tr = xb.rollout("SGD", [1920000], ["half_block"], None, pol=pol, trace=True)
        self.assertEqual(pol.t, 40)
        mom = np.array([float(r["momentum"]) for r in tr])
        wd = np.array([float(r["weight_decay"]) for r in tr])
        lr = np.array([float(r["lr"]) for r in tr])
        np.testing.assert_allclose(mom, 0.95 * (draws[:, 1].numpy() + 1) / 2, atol=1e-5)
        np.testing.assert_allclose(np.log10(wd), -6 + 2 * (draws[:, 2].numpy() + 1), atol=1e-4)
        expect, cur = [], 0.1  # cold start
        for a0 in draws[:, 0].numpy():
            cur = min(0.3, max(0.001, cur * 2 ** a0))
            expect.append(cur)
        np.testing.assert_allclose(lr, expect, rtol=1e-4)

    def test_random_fixed_is_deterministic_grid_point(self):
        for family in xb.FAMILIES:
            grid = xb.r.configs(family)
            picks = [xb.random_fixed_config(family, k) for k in xb.RANDOM_SEEDS]
            self.assertEqual(picks, [xb.random_fixed_config(family, k) for k in xb.RANDOM_SEEDS])
            for index, config in picks:
                self.assertEqual(grid[index], config)

    def test_untrained_policy_ignores_seed_and_observation(self):
        for family in xb.FAMILIES:
            for warm in (None, xb.extreme_config(family)):
                obs = torch.randn(3, 9 if family == "SGD" else 10)
                outs = [xb.untrained_policy(family, s, warm)(obs) for s in xb.POLICY_SEEDS]
                for o in outs:
                    torch.testing.assert_close(o, outs[0].expand_as(o), rtol=0, atol=0)
                    torch.testing.assert_close(o, o[:1].expand_as(o), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
