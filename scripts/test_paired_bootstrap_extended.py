"""Checks for scripts/paired_bootstrap_extended.py.

Synthetic: metric selection and cross-fold pairing. Saved data: the extended
script's mean paired differences reproduce each experiment's own committed
summary (tolerance 1e-9 pp), so the CIs are attached to the published numbers.
"""

import csv
import unittest

import numpy as np

import paired_bootstrap_extended as x
from paired_bootstrap import bootstrap, paired


def published(path, **filters):
    with (x.ROOT / path).open() as handle:
        return [r for r in csv.DictReader(handle) if all(r[k] == v for k, v in filters.items())]


class ExtendedBootstrapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boot = x.Runner()
        inp = x.Inputs()
        for group in ("ablations_start", "retrained_five", "warm", "held_out"):
            getattr(x, group)(inp, cls.boot)

    def row(self, group, comparison, family, suite, condition):
        rows = [r for r in self.boot.rows if (r["group"], r["comparison"], r["family"], r["suite"], r["condition"])
                == (group, comparison, family, suite, condition)]
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_cross_fold_pairing_shares_policy_seeds(self):
        # Two folds, each with its own policies labelled by the same seeds.
        a = [dict(policy_seed=str(s), seed=str(e), order=o, accuracy=str(0.5 + 0.01 * s))
             for o in ("f1", "f2") for s in (1, 2) for e in (10, 11)]
        b = [dict(policy_seed="-1", seed=str(e), order=o, accuracy="0.5") for o in ("f1", "f2") for e in (10, 11)]
        result = bootstrap(paired(a, b))
        self.assertAlmostEqual(result["mean_diff_pp"], 1.5)
        self.assertEqual(result["n_policy_seeds"], 2)

    def test_retrained_matches_published(self):
        for family in x.FAMILIES:
            for comparator, name in (("fixed_retuned", "retrained_vs_retuned_fixed"), ("fixed", "retrained_vs_saved_fixed"),
                                     ("policy", "retrained_vs_frozen")):
                ref = published("results/extreme_retrained_five_seeds/comparisons.csv", family=family, scope="biased",
                                control=comparator)
                self.assertEqual(len(ref), 1)
                got = self.row("retrained_five", name, family, "extreme", "biased_mean")["mean_diff_pp"]
                self.assertAlmostEqual(got, 100 * float(ref[0]["mean"]), delta=1e-9)

    def test_warm_matches_published(self):
        for family in x.FAMILIES:
            for suite, group, condition in (("regular", "biased", "biased_mean"), ("extreme", "extreme", "all")):
                ref = published("results/warm_start/summary.csv", family=family, suite=suite, group=group)[0]
                got = self.row("warm", "warm_vs_fixed", family, suite, condition)["mean_diff_pp"]
                self.assertAlmostEqual(got, 100 * float(ref["warm_minus_fixed"]), delta=1e-9)
                got = self.row("warm", "warm_vs_full", family, suite, condition)["mean_diff_pp"]
                self.assertAlmostEqual(got, 100 * float(ref["warm_minus_full"]), delta=1e-9)

    def test_held_out_matches_published(self):
        for family in x.FAMILIES:
            ref = published("results/extreme_held_out/paired_differences.csv", family=family, scope="all",
                            variant="policy_retrained", comparator="fixed_retuned")[0]
            got = self.row("held_out", "retrained_vs_fold_retuned_fixed", family, "extreme_held_out", "all")
            self.assertAlmostEqual(got["mean_diff_pp"], 100 * float(ref["mean_difference"]), delta=1e-9)

    def test_start_held_matches_summary(self):
        # Published summary.md rounds to 0.01 pp.
        expected = {("SGD", "lr_only"): 0.08, ("SGD", "fixed_memory"): -0.60,
                    ("Adam", "lr_only"): -1.05, ("Adam", "fixed_memory"): -0.88}
        for (family, variant), value in expected.items():
            got = self.row("ablations_start", f"start_{variant}_vs_full_policy", family, "regular", "all")
            self.assertAlmostEqual(got["mean_diff_pp"], value, delta=0.005 + 1e-9)

    def test_intervals_are_ordered_and_reproducible(self):
        for r in self.boot.rows:
            self.assertLessEqual(r["ci_low_pp"], r["ci_high_pp"])
            self.assertTrue(np.isfinite(r["mean_diff_pp"]))
        again = x.Runner()
        x.warm(x.Inputs(), again)
        self.assertEqual(again.rows, [r for r in self.boot.rows if r["group"] == "warm"])


if __name__ == "__main__":
    unittest.main()
