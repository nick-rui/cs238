"""Regression checks for interrupted ablation evaluation outputs."""

import csv
import gzip
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import torch
import run as a


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
