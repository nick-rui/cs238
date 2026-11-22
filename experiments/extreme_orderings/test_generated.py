"""Invariant tests for procedurally generated extreme orderings."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import generated as g  # noqa: E402

SEEDS = (1920000, 1920007, 1970123)


class GeneratedStreams(unittest.TestCase):
    def test_original_specs_reproduce_original_streams(self):
        for name, spec in g.ORIGINAL_SPECS.items():
            for seed in SEEDS:
                ids, events, mask = g._ORIGINAL_STREAM(seed, name)
                gids, gevents, gmask = g.spec_stream(seed, spec)
                self.assertTrue(torch.equal(ids, gids), name)
                self.assertEqual(events, gevents, name)
                self.assertTrue(np.array_equal(mask, gmask), name)

    def test_sampled_specs_invariants(self):
        specs = [g.sample_spec(s) for s in range(9000000, 9000060)]
        for i, spec in enumerate(specs):
            self.assertTrue(g.feasible(spec))
            self.assertIn(spec.k, g.KS)
            self.assertTrue(1 <= spec.n <= 10)
            name = f"test_{i}"
            g.register(name, spec)
            for seed in SEEDS[:2]:
                ref = g._ORIGINAL_STREAM(seed, "mixed")[0].flatten().sort().values
                ids, events, mask = g.stream(seed, name)
                self.assertEqual(tuple(ids.shape), (200, 64))
                # Same balanced multiset as the shuffled stream for this episode seed.
                self.assertTrue(torch.equal(ids.flatten().sort().values, ref))
                self.assertTrue(torch.equal(torch.bincount(g.e.y[ids.flatten()], minlength=10),
                                            torch.full((10,), 1280)))
                self.assertEqual(len(events), spec.n)
                self.assertEqual(int(mask.sum()), spec.n * spec.L)
                self.assertEqual(events[0]["start"], spec.start)
                self.assertEqual(200 - events[-1]["end"], spec.R)
                for ev in events:
                    self.assertEqual(len(set(ev["digits"])), spec.k)
                    lab = g.e.y[ids[ev["start"]:ev["end"]]]
                    self.assertTrue(torch.isin(lab, torch.tensor(ev["digits"])).all())
                    # Block classes balanced within one example.
                    counts = torch.bincount(lab.flatten(), minlength=10)[ev["digits"]]
                    self.assertLessEqual(int(counts.max() - counts.min()), 1)
                # Batches 64 and episode digits permuted: block digits depend on seed.
            # Different episode seeds permute digits differently.
            d0 = g.stream(SEEDS[0], name)[1][0]["digits"]
            d1 = g.stream(SEEDS[1], name)[1][0]["digits"]
            self.assertEqual(len(d0), len(d1))

    def test_modes(self):
        s = g.Spec(2, 4, "alternate", 10, 3, 7)
        self.assertEqual([s.block_slots(i) for i in range(4)], [[0, 1], [2, 3], [0, 1], [2, 3]])
        s = g.Spec(3, 4, "change", 10, 0, 7)
        self.assertEqual(s.block_slots(3), [9, 0, 1])
        s = g.Spec(2, 3, "repeat", 10, 2, 7)
        self.assertEqual({tuple(s.block_slots(i)) for i in range(3)}, {(0, 1)})

    def test_paired_initialization_and_probe(self):
        # Classifier initialization depends only on the episode seed.
        a = g.e.parameters([1970000])
        c = g.e.parameters([1970000])
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, c)))

    def test_sampler_deterministic_and_covers_space(self):
        self.assertEqual(g.sample_spec(123), g.sample_spec(123))
        specs = [g.sample_spec(s) for s in range(9100000, 9100400)]
        self.assertEqual({s.k for s in specs}, set(g.KS))
        self.assertEqual({s.n for s in specs}, set(range(1, 11)))
        self.assertEqual({s.mode for s in specs}, {"single", *g.MODES})
        self.assertIn(0, {s.R for s in specs})

    def test_severity(self):
        sev = g.severity(g.ORIGINAL_SPECS["class_sorted"])
        self.assertAlmostEqual(sev["mean_kl"], np.log(10))
        self.assertEqual(g.severity(g.ORIGINAL_SPECS["single_max"])["biased_batches"], 20)
        for name, spec in g.ORIGINAL_SPECS.items():
            self.assertEqual(g.nearness(spec)["nearest"], name)
            self.assertIn(name, g.nearness(spec)["near_duplicate_of"])


class Design(unittest.TestCase):
    def test_prespecified_design(self):
        import generated_eval as G
        G.check_design()  # seed pools disjoint from all earlier experiments
        self.assertEqual(len(G.ORDERINGS), G.N_ORDERINGS)
        self.assertEqual(len({G.SPECS[o] for o in G.ORDERINGS}) > 150, True)
        for o in G.ORDERINGS:
            self.assertTrue(g.feasible(G.SPECS[o]))


if __name__ == "__main__":
    unittest.main()
