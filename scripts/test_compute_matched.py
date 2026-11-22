"""Focused grid and isolated-evaluation regression tests."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from analyze_compute_matched import paired_accuracy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/optimizer_families"))
spec = importlib.util.spec_from_file_location("family_runner", ROOT / "experiments/optimizer_families/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
b = runner.b


class SearchTests(unittest.TestCase):
    def test_paired_records(self):
        rows = [dict(seed="1", order="mixed", policy_seed="-1", accuracy="0.9")]
        self.assertEqual(paired_accuracy(rows, [1], ["mixed"]), 90)
        for invalid in [[], rows * 2, [dict(rows[0], seed="2")]]:
            with self.assertRaises(ValueError):
                paired_accuracy(invalid, [1], ["mixed"])
        with self.assertRaises(ValueError):
            paired_accuracy([], [], [])
        policy = [dict(rows[0], policy_seed=str(ps)) for ps in [338, 339]]
        self.assertEqual(paired_accuracy(policy, [1], ["mixed"], policy=True), 90)

    def test_grids(self):
        for family in ["SGD", "Adam", "AdamW", "RMSprop"]:
            configs = b.sgd_configs if family == "SGD" else lambda grid="default": runner.configs(family, grid)
            old, wide = configs(), configs("wide")
            key = lambda c: tuple(sorted(c.items()))
            self.assertEqual(len(old), 144 if family == "SGD" else 288)
            self.assertEqual(len(wide), 504 if family == "SGD" else 480)
            self.assertEqual(len({key(c) for c in wide}), len(wide))
            self.assertTrue({key(c) for c in old} <= {key(c) for c in wide})
        self.assertEqual(b.DEVSEEDS[:8], list(range(1810000, 1810008)))
        self.assertEqual(b.DEVORDERS[:8], b.TRAIN * 2)

    def test_unsafe_cli_rejected(self):
        for script in ["recovery_study/study.py", "optimizer_families/run.py", "extreme_orderings/run.py"]:
            result = subprocess.run(
                [sys.executable, str(ROOT / "experiments" / script), "evaluate", "--fixed-only"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("isolated evaluation requires", result.stderr)
        for script in ["recovery_study/study.py", "optimizer_families/run.py"]:
            result = subprocess.run(
                [sys.executable, str(ROOT / "experiments" / script), "tune", "--grid", "wide"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--grid wide requires --output-dir", result.stderr)


if __name__ == "__main__":
    unittest.main()
