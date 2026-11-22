"""Checks for the extreme warm-start runner (no training)."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as x  # noqa: E402


class ExtremeWarmStartTest(unittest.TestCase):
    def test_initialized_from_extreme_retuned_config(self):
        for family in x.FAMILIES:
            config = x.extreme_config(family)
            self.assertNotEqual(config, x.regular_config(family))
            captured = {}
            original = x.r.train

            def fake_train(fam, seed):
                captured["policy"] = x.r.make_policy(fam, seed)
                captured["rollout"] = x.r.rollout
                raise RuntimeError("stop")

            x.r.train = fake_train
            try:
                with self.assertRaises(RuntimeError):
                    x.train(family, 338, x.OUT / "unused")
            finally:
                x.r.train = original
            expected = (x.e.policy("gradient", 338, warm=config) if family == "SGD"
                        else x.f.policy(family, "gradient", 338, warm=config))
            for a, b in zip(captured["policy"].parameters(), expected.parameters()):
                self.assertTrue(torch.equal(a, b))
            # Patches are restored after training.
            self.assertIsNot(x.r.rollout, captured["rollout"])
            self.assertEqual(x.r.make_policy.__name__, "make_policy")

    def test_training_design_matches_cold_retraining(self):
        self.assertEqual(x.r.UPDATES, 500)
        self.assertEqual(x.r.train_seed(338, 0, 0), 6000000 + 3380000)
        self.assertEqual(x.r.DEVSEEDS, list(range(1930000, 1930016)))
        self.assertEqual(x.TEST_SEEDS, tuple(range(1920000, 1920020)))
        self.assertEqual(set(x.SEEDS), set(x.r.POLICY_SEEDS))
        x.r.assert_disjoint()

    def test_comparator_checkpoints_exist(self):
        for family in x.FAMILIES:
            for variant in ("policy_retrained", "warm_regular"):
                for seed in x.SEEDS:
                    self.assertTrue(x.checkpoint_path(family, variant, seed).is_file())


if __name__ == "__main__":
    unittest.main()
