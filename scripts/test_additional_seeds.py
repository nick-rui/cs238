"""Fast tests for isolated output handling and across-policy-seed summaries."""

import argparse
import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('additional', Path(__file__).with_name('additional_seeds.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class AdditionalSeedsTests(unittest.TestCase):
    def test_summary_averages_episode_means_across_five_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_root = runner.ROOT
            runner.ROOT = root
            try:
                out = root / 'results/additional_seeds'
                out.mkdir(parents=True)
                for suite, source in [('regular', 'optimizer_families'), ('extreme', 'extreme_orderings')]:
                    for family in runner.FAMILIES:
                        rows = []
                        for seed, accuracy in [(-1, .5), (338, .4), (339, .5), (340, .6), (341, .7), (342, .8)]:
                            rr = [dict(order=o, policy_seed=seed, accuracy=accuracy, failed=False)
                                  for o in ['mixed', 'biased'] for _ in range(2)]
                            if seed <= 339:
                                rows.extend(rr)
                            else:
                                folder = out / suite / f'{family}_{seed}'
                                folder.mkdir(parents=True)
                                runner.write_csv(folder / 'evaluation.csv', rr)
                        folder = root / 'results' / source
                        folder.mkdir(parents=True, exist_ok=True)
                        runner.write_csv(folder / f'{family}_evaluation.csv', rows)
                runner.summarize(argparse.Namespace(out_dir=out, seeds=[340,341,342]))
                rows = runner.read_csv(out / 'summary.csv')
                self.assertEqual(len(rows), 24)
                for r in rows:
                    self.assertAlmostEqual(float(r['policy_mean_pct']), 60)
                    self.assertAlmostEqual(float(r['fixed_accuracy_pct']), 50)
                    self.assertAlmostEqual(float(r['delta_pp']), 10)
                    self.assertAlmostEqual(float(r['policy_sd_pct']), 15.811388300841896)
                    self.assertEqual(int(r['seeds_beating_fixed']), 3)
            finally:
                runner.ROOT = old_root

    def test_empty_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'empty.csv'
            path.write_text('accuracy\n')
            with self.assertRaisesRegex(ValueError, 'Empty evaluation file'):
                runner.read_csv(path)

    def test_protected_original_output_is_rejected(self):
        result = subprocess.run([sys.executable, str(Path(runner.__file__)), 'pipeline',
                                 '--out-dir', str(runner.ROOT / 'results/optimizer_families')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('original studies are protected', result.stderr)


if __name__ == '__main__':
    unittest.main()
