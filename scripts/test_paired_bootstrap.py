"""Synthetic tests for saved-result pairing and hierarchical resampling."""

import unittest

import numpy as np

from paired_bootstrap import bootstrap, paired


def row(policy, episode, accuracy, order="mixed"):
    return dict(policy_seed=str(policy), seed=str(episode), accuracy=str(accuracy), order=order)


class PairedBootstrapTest(unittest.TestCase):
    def test_pairing_and_known_mean(self):
        a = [row(s, e, 0.5 + s * 0.01 + e * 0.001) for s in (1, 2) for e in (10, 20)]
        b = [row(-1, e, 0.5 + e * 0.001) for e in (20, 10)]
        matrices = paired(list(reversed(a)), b)
        np.testing.assert_allclose(matrices["mixed"][1], [[1, 1], [2, 2]])
        result = bootstrap(matrices)
        self.assertAlmostEqual(result["mean_diff_pp"], 1.5)
        self.assertLessEqual(result["ci_low_pp"], 1.5)
        self.assertGreaterEqual(result["ci_high_pp"], 1.5)
        self.assertEqual(result["episode_win_rate"], 1)
        self.assertEqual(result["seeds_beating"], 2)
        self.assertEqual(result, bootstrap(matrices))

    def test_seed_pairing_ties_and_joint_aggregate(self):
        a = [row(s, e, 0.7) for s in (338, 339) for e in (1, 2)]
        b = [row(s, e, 0.7) for s in (339, 338) for e in (2, 1)]
        result = bootstrap(paired(a, b))
        self.assertEqual(result["episode_win_rate"], 0.5)
        self.assertEqual(result["p_le_0"], 1)
        self.assertEqual(result["seeds_beating"], 0)
        d = np.array([[1., -1.], [2., -2.]])
        result = bootstrap({"a": ((1, 2), d), "b": ((1, 2), -d)})
        self.assertEqual(result["ci_low_pp"], 0)
        self.assertEqual(result["ci_high_pp"], 0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            paired([], [])
        with self.assertRaises(ValueError):
            bootstrap({})
        for a, b in [([row(1, 1, .5)] * 2, [row(-1, 1, .5)]),
                     ([row(1, 1, .5)], [row(-1, 2, .5)]),
                     ([row(1, 1, .5)], [row(2, 1, .5)]),
                     ([row(1, 1, float("nan"))], [row(-1, 1, .5)]),
                     ([row(1, 1, .5), row(2, 2, .5)],
                      [row(-1, 1, .5), row(-1, 2, .5)])]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                paired(a, b)


if __name__ == "__main__":
    unittest.main()
