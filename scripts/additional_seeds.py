"""Isolated, restartable extra-seed training and paired evaluation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/additional_seeds'
FAMILIES = ['SGD', 'Adam', 'AdamW', 'RMSprop']


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + '.partial')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(temporary, 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def engines(suite='regular'):
    sys.path.insert(0, str(ROOT / 'experiments/optimizer_families'))
    import family_engine as f
    import run as runner
    stress = None
    if suite == 'extreme':
        spec = importlib.util.spec_from_file_location(
            'stress', ROOT / 'experiments/extreme_orderings/run.py')
        stress = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stress)
    import torch
    torch.set_num_threads(1)
    return f, runner, stress


def train(args):
    f, runner, _ = engines()
    out = args.out_dir / 'training'
    out.mkdir(parents=True, exist_ok=True)
    folder = out / f'{args.family}_gradient_{args.seed}'
    if args.family == 'SGD':
        f.base.OUT = out
        original = out / f'gradient_{args.seed}'
        if folder.exists() and not original.exists():
            folder.rename(original)
        try:
            f.base.train('gradient', args.seed, updates=args.updates)
        finally:
            if original.exists():
                original.rename(folder)
    else:
        runner.OUT = out
        runner.train(args.family, 'gradient', args.seed, updates=args.updates)


def checkpoint(args):
    return args.out_dir / 'training' / f'{args.family}_gradient_{args.seed}' / 'best.pt'


def evaluate(args):
    import torch
    f, _, stress = engines(args.suite)
    b, e = f.base, f.e
    out = args.out_dir / args.suite / f'{args.family}_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'done.json').exists():
        return
    start = time.perf_counter()
    saved = torch.load(checkpoint(args), weights_only=True)
    pol = e.policy('gradient', args.seed) if args.family == 'SGD' else f.policy(args.family, 'gradient', args.seed)
    pol.load_state_dict(saved['policy'])
    pol.eval()
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    testset = (test.data.reshape(-1, 784).float() / 255, test.targets)
    seeds = stress.SEEDS if stress else b.TESTSEEDS
    orders = stress.ORDERS if stress else b.ORDERS
    if args.smoke:
        seeds, orders = seeds[:4], orders[:1]
    rows, traces = [], []
    for order in orders:
        for st in range(0, len(seeds), 4):
            ss = seeds[st:st + 4]
            if args.family == 'SGD':
                rr, tr, *_ = e.rollout(pol, 'gradient', ss, [order] * len(ss), evaluation=testset, trace=True)
            else:
                rr, tr, *_ = f.rollout(args.family, 'gradient', ss, [order] * len(ss), pol, evaluation=testset, trace=True)
            if args.family == 'SGD' and not stress:
                tr = b.annotate(tr)
            for r in rr:
                r.update(family=args.family, variant='policy', policy_seed=args.seed, checkpoint=saved['update'])
            for r in tr:
                r.update(family=args.family, variant='policy', policy_seed=args.seed)
            rows.extend(rr)
            traces.extend(tr)
        print(json.dumps(dict(suite=args.suite, family=args.family, seed=args.seed, order=order, seconds=time.perf_counter()-start)), flush=True)
    write_csv(out / 'evaluation.csv', rows)
    write_csv(out / 'traces.csv.gz', traces)
    write_json(out / 'done.json', dict(rows=len(rows), traces=len(traces), seconds=time.perf_counter()-start,
                                      failures=sum(r['failed'] for r in rows), checkpoint=saved['update']))


def read_csv(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Empty evaluation file: {path}')
    return rows


def replay(args):
    import torch
    f, _, stress = engines(args.suite)
    e = f.e
    folder = args.out_dir / args.suite / f'{args.family}_{args.seed}'
    order = 'single_late' if stress else 'single_middle'
    episode_seed = 1920000 if stress else 1900000
    expected = next(r for r in read_csv(folder / 'evaluation.csv')
                    if r['order'] == order and int(r['seed']) == episode_seed)
    saved = torch.load(checkpoint(args), weights_only=True)
    pol = e.policy('gradient', args.seed) if args.family == 'SGD' else f.policy(args.family, 'gradient', args.seed)
    pol.load_state_dict(saved['policy'])
    test = e.MNIST(str(e.DATA_DIR), train=False, download=False)
    testset = (test.data.reshape(-1,784).float()/255, test.targets)
    if args.family == 'SGD':
        rr, tr, *_ = e.rollout(pol, 'gradient', [episode_seed], [order], evaluation=testset, trace=True)
    else:
        rr, tr, *_ = f.rollout(args.family, 'gradient', [episode_seed], [order], pol, evaluation=testset, trace=True)
    errors = {k: abs(float(rr[0][k])-float(expected[k])) for k in ['accuracy','eval_loss','path_log_loss']}
    assert errors['accuracy'] < 1e-6 and max(errors.values()) < 3e-5, errors
    assert str(rr[0]['failed']) == expected['failed']
    with gzip.open(folder / 'traces.csv.gz', 'rt') as handle:
        expected_traces = {int(r['step']): r for r in csv.DictReader(handle)
                           if r['order'] == order and int(r['seed']) == episode_seed}
    for t in tr:
        for k in ['lr','momentum','weight_decay','second_decay']:
            if k in t:
                assert math.isclose(float(t[k]), float(expected_traces[t['step']][k]), rel_tol=2e-4, abs_tol=3e-6), (t['step'], k)
    write_json(folder / 'reproduction.json', dict(metric_absolute_errors=errors, action_traces_match=True,
                                                order=order, episode_seed=episode_seed))
    print('REPLAY PASSED', args.suite, args.family, args.seed, errors, flush=True)


def summarize(args):
    summary, seed_rows = [], []
    for suite, source in [('regular', 'optimizer_families'), ('extreme', 'extreme_orderings')]:
        for family in FAMILIES:
            rows = read_csv(ROOT / 'results' / source / f'{family}_evaluation.csv')
            for seed in args.seeds:
                rows.extend(read_csv(args.out_dir / suite / f'{family}_{seed}' / 'evaluation.csv'))
            orders = sorted({r['order'] for r in rows})
            for condition in orders + ['biased_mean']:
                chosen = [r for r in rows if (r['order'] != 'mixed' if condition == 'biased_mean' else r['order'] == condition)]
                fixed = statistics.mean(float(r['accuracy']) for r in chosen if int(r['policy_seed']) == -1)
                per_seed = []
                for seed in [338, 339] + args.seeds:
                    rr = [r for r in chosen if int(r['policy_seed']) == seed]
                    value = statistics.mean(float(r['accuracy']) for r in rr)
                    per_seed.append(value)
                    seed_rows.append(dict(suite=suite, family=family, condition=condition, policy_seed=seed,
                                          accuracy_pct=100*value, fixed_accuracy_pct=100*fixed,
                                          delta_pp=100*(value-fixed), failed=sum(r['failed'] in ['True', 'true', '1'] for r in rr)))
                summary.append(dict(suite=suite, family=family, condition=condition, policy_seeds=len(per_seed),
                                    policy_mean_pct=100*statistics.mean(per_seed), policy_sd_pct=100*statistics.stdev(per_seed),
                                    fixed_accuracy_pct=100*fixed, delta_pp=100*(statistics.mean(per_seed)-fixed),
                                    seeds_beating_fixed=sum(v > fixed for v in per_seed)))
    write_csv(args.out_dir / 'summary.csv', summary)
    write_csv(args.out_dir / 'policy_seed_summary.csv', seed_rows)


def validate(args):
    checks = []
    for family in FAMILIES:
        for seed in args.seeds:
            done = json.loads((args.out_dir / 'training' / f'{family}_gradient_{seed}' / 'done.json').read_text())
            assert done['updates'] == 500 and done['episodes'] == 4000
            for suite, source, start, count in [('regular', 'optimizer_families', 1900000, 10), ('extreme', 'extreme_orderings', 1920000, 8)]:
                folder = args.out_dir / suite / f'{family}_{seed}'
                rows = read_csv(folder / 'evaluation.csv')
                reference = read_csv(ROOT / 'results' / source / f'{family}_evaluation.csv')
                orders = {r['order'] for r in reference}
                expected = {(o, s) for o in orders for s in range(start, start+20)}
                assert len(rows) == count*20
                assert {(r['order'], int(r['seed'])) for r in rows} == expected
                assert all(int(r['policy_seed']) == seed and r['family'] == family for r in rows)
                assert all(math.isfinite(float(r[k])) for r in rows for k in ['accuracy', 'eval_loss', 'path_log_loss'])
                with gzip.open(folder / 'traces.csv.gz', 'rt') as handle:
                    traces = list(csv.DictReader(handle))
                assert len(traces) == len(rows)*40
                assert {(r['order'], int(r['seed']), int(r['step'])) for r in traces} == {(o, s, t) for o, s in expected for t in range(5,201,5)}
                checks.append(dict(suite=suite, family=family, seed=seed, rows=len(rows), traces=len(traces), complete=True))
    for name, digest in json.loads((args.out_dir / 'original_sha256.json').read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    write_json(args.out_dir / 'checks.json', dict(original_results_unchanged=True, evaluations=checks))
    print('VALIDATED: 12 completed trainings, 24 complete evaluation grids and traces; original tracked results unchanged', flush=True)


def pipeline(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs = args.out_dir / 'logs'
    logs.mkdir(exist_ok=True)
    manifest = args.out_dir / 'original_sha256.json'
    if not manifest.exists():
        names = subprocess.check_output(['git', 'ls-files', 'results/'], cwd=ROOT, text=True).splitlines()
        names = [n for n in names if not (ROOT / n).is_relative_to(args.out_dir)]
        write_json(manifest, {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names})
    write_json(args.out_dir / 'design.json', dict(policy_seeds=args.seeds, updates=500, episodes_per_update=8,
               workers=args.workers, threads=1, baseline_source='results/optimizer_families/*_baseline.json',
               baseline_retuning=False, resume='Completed training/evaluation jobs are reused; interrupted jobs restart from their beginning.'))
    def job(mode, family, seed, suite=None):
        command = [sys.executable, str(Path(__file__).resolve()), mode, '--out-dir', str(args.out_dir),
                   '--family', family, '--seed', str(seed)]
        if suite:
            command += ['--suite', suite]
        name = f'{mode}_{suite or "policy"}_{family}_{seed}'
        print('START', name, flush=True)
        with (logs / f'{name}.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        print('DONE', name, flush=True)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, 'train', family, seed) for family in FAMILIES for seed in args.seeds]
        for future in futures:
            future.result()
    training_seconds = time.perf_counter()-start
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, 'evaluate', family, seed, suite) for suite in ['regular','extreme'] for family in FAMILIES for seed in args.seeds]
        for future in futures:
            future.result()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, 'replay', family, seed, suite) for suite in ['regular','extreme'] for family in FAMILIES for seed in args.seeds]
        for future in futures:
            future.result()
    summarize(args)
    validate(args)
    write_json(args.out_dir / 'pipeline_done.json', dict(training_wall_seconds=training_seconds, total_wall_seconds=time.perf_counter()-start))
    print('PIPELINE COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'evaluate', 'replay', 'pipeline', 'summarize', 'validate'])
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--seeds', type=int, nargs='+', default=[340,341,342])
    parser.add_argument('--suite', choices=['regular','extreme'], default='regular')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--updates', type=int, default=500, help='Training-only smoke override; production uses 500')
    parser.add_argument('--smoke', action='store_true', help='Evaluate only four mixed episodes')
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.out_dir == ROOT / 'results' or (args.out_dir.is_relative_to(ROOT / 'results') and not args.out_dir.is_relative_to(DEFAULT_OUT)):
        parser.error('Use results/additional_seeds or a directory outside results; original studies are protected')
    if args.mode in ['train','evaluate','replay'] and (args.family is None or args.seed is None):
        parser.error('--family and --seed are required')
    if args.seed in [338,339] or len(set(args.seeds)) != len(args.seeds) or set(args.seeds) & {338,339}:
        parser.error('Additional seeds must be unique and exclude 338/339')
    if args.workers < 1 or args.updates < 1:
        parser.error('Workers and updates must be positive')
    globals()[args.mode](args)
