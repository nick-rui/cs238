"""Regression checks for interrupted ablation evaluation outputs."""

import csv
import gzip
import math
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import torch
import run as a


class StartHeldTests(unittest.TestCase):
    def test_unknown_held_source_is_rejected(self):
        job = ("SGD", "lr_only", 338)
        with self.assertRaisesRegex(ValueError, "unknown held_source"):
            a.rollout(*job[:2], seeds=[], orders=[], pol=None, held_source="typo")
        with self.assertRaisesRegex(ValueError, "unknown held_source"):
            a.train(*job, updates=1, root=None, held_source="typo")
        with patch.object(a.e, "MNIST") as mnist:
            with self.assertRaisesRegex(ValueError, "unknown held_source"):
                a.evaluate([job], held_source="typo")
            mnist.assert_not_called()

    def test_removed_controls_keep_full_rollout_start(self):
        torch.set_num_threads(1)
        for family, wd_column in [("SGD", 2), ("Adam", 3)]:
            seeds, orders = [1810000], ["mixed"]
            full = a.policy(family, "gradient", 338)
            full_obs = []
            hook = full.register_forward_pre_hook(lambda _, args: full_obs.append(args[0]))
            _, full_trace, _, _ = a.rollout(
                family, "gradient", seeds, orders, full, held_source="start", trace=True
            )
            hook.remove()
            for kind in ["lr_only", "fixed_memory"]:
                with self.subTest(family=family, kind=kind), patch.object(
                    a, "SOURCE", None
                ):
                    pol = a.policy(family, kind, 338)
                    obs = []
                    hook = pol.register_forward_pre_hook(lambda _, args: obs.append(args[0]))
                    _, trace, _, _ = a.rollout(
                        family, kind, seeds, orders, pol, held_source="start", trace=True
                    )
                    hook.remove()
                    torch.testing.assert_close(obs[0], full_obs[0])
                    for row in trace:
                        self.assertAlmostEqual(row["momentum"], full_trace[0]["momentum"], places=6)
                        if family == "Adam":
                            self.assertAlmostEqual(row["second_decay"], full_trace[0]["second_decay"], places=6)
                        if kind == "lr_only":
                            self.assertAlmostEqual(row["weight_decay"], 1e-4)
                            self.assertTrue(math.isfinite(float(obs[0][0, wd_column])))


class EvaluationReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "SGD_lr_only_338"
        self.folder.mkdir()

    def write_outputs(self):
        a.e.writecsv(self.folder / "evaluation.csv", [dict(seed=i) for i in range(200)])
        with gzip.open(self.folder / "traces.csv.gz", "wt") as stream:
            writer = csv.DictWriter(stream, fieldnames=["step"])
            writer.writeheader()
            writer.writerows(dict(step=i) for i in range(8000))

    def test_complete(self):
        self.write_outputs()
        self.assertTrue(a.evaluation_complete(self.folder))

    def test_missing_and_empty(self):
        self.assertFalse(a.evaluation_complete(self.folder))
        self.write_outputs()
        (self.folder / "traces.csv.gz").unlink()
        self.assertFalse(a.evaluation_complete(self.folder))
        (self.folder / "evaluation.csv").write_text("")
        self.assertFalse(a.evaluation_complete(self.folder))

    def test_truncated_outputs(self):
        self.write_outputs()
        trace = self.folder / "traces.csv.gz"
        trace.write_bytes(trace.read_bytes()[:-8])
        self.assertFalse(a.evaluation_complete(self.folder))
        self.write_outputs()
        a.e.writecsv(self.folder / "evaluation.csv", [dict(seed=1)])
        self.assertFalse(a.evaluation_complete(self.folder))
        self.write_outputs()
        with gzip.open(trace, "wt") as stream:
            stream.write("step\n1\n")
        self.assertFalse(a.evaluation_complete(self.folder))

    def test_corrupt_compressed_data(self):
        self.write_outputs()
        damaged = bytearray(gzip.compress(b"step\n1\n", mtime=0))
        damaged[10] = (damaged[10] & 0xF8) | 0x07
        with self.assertRaises(zlib.error):
            gzip.decompress(damaged)
        (self.folder / "traces.csv.gz").write_bytes(damaged)
        self.assertFalse(a.evaluation_complete(self.folder))

    def test_crash_before_evaluation_publish_and_retry(self):
        pol = a.policy("SGD", "lr_only", 338)
        torch.save(dict(policy=pol.state_dict(), update=0), self.folder / "best.pt")
        test = type("Test", (), dict(data=torch.zeros(1, 784), targets=torch.zeros(1, dtype=torch.long)))()

        def rollout(family, kind, seeds, orders, policy, **kwargs):
            rows = [dict(seed=s, order=o) for s, o in zip(seeds, orders)]
            traces = [dict(seed=s, order=o, step=t) for t in range(a.e.H) for s, o in zip(seeds, orders)]
            return rows, traces, None, None

        with patch.object(a, "OUT", self.root), patch.object(a.e, "MNIST", return_value=test), patch.object(a, "rollout", side_effect=rollout) as run:
            with patch.object(a.e, "writecsv", side_effect=OSError("interrupted CSV write")):
                with self.assertRaisesRegex(OSError, "interrupted CSV write"):
                    a.evaluate([("SGD", "lr_only", 338)])
            self.assertTrue((self.folder / "traces.csv.gz").exists())
            self.assertFalse((self.folder / "evaluation.csv").exists())
            self.assertFalse(a.evaluation_complete(self.folder))
            a.evaluate([("SGD", "lr_only", 338)])
            self.assertTrue(a.evaluation_complete(self.folder))
            run.reset_mock()
            a.evaluate([("SGD", "lr_only", 338)])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
