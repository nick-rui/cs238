"""Small initialization checks; full numeric replay uses check_reproduction.py."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as w


class WarmInitializationTest(unittest.TestCase):
    def test_default_is_unchanged_when_warm_omitted(self):
        for family in w.FAMILIES:
            a, b = w.policy(family, 338), w.policy(family, 338, None)
            for left, right in zip(a.parameters(), b.parameters()):
                self.assertTrue(torch.equal(left, right))

    def test_warm_mean_actions(self):
        for family in w.FAMILIES:
            config = w.config(family)
            pol = w.policy(family, 338, config)
            self.assertTrue(torch.equal(pol[-1].weight, torch.zeros_like(pol[-1].weight)))
            self.assertEqual(float(pol[-1].bias[0]), 0)
            self.assertTrue(torch.isfinite(pol[-1].bias).all())
            self.assertTrue((pol[-1].bias.abs() <= 3).all())
            a = pol[-1].bias.detach().tanh()
            target = config.get("momentum", config.get("beta1"))
            self.assertAlmostEqual(float(.95 * (a[1] + 1) / 2), target, delta=.0024)
            if family != "SGD":
                decay = 1 - 10 ** (-1 - 3 * float(a[2] + 1) / 2)
                self.assertAlmostEqual(decay, config.get("alpha", config.get("beta2")), delta=.002)

    def test_clamps_and_invalid_config(self):
        self.assertEqual(w.e.warm_settings(dict(lr=1., momentum=-1., weight_decay=0.)), (.3, 0., 1e-6))
        self.assertEqual(w.f.warm_settings(dict(lr=0., beta1=2., beta2=1., weight_decay=1.)), (1e-4, .95, .9999, .01))
        for family in w.FAMILIES:
            with self.assertRaises(KeyError):
                w.policy(family, 338, {})


if __name__ == "__main__":
    unittest.main()
