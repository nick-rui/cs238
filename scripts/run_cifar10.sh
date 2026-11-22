#!/usr/bin/env bash
# Full CIFAR-10 regular-suite run on at most eight one-thread processes.
# Resumable: completed training (done.json), family tuning (*_baseline.json)
# and evaluations (*_evaluation.csv) are skipped; SGD tune/evaluate are guarded
# here because study.py always regenerates them. Logs go to results/cifar10/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
export CS238_DATASET=cifar10 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MNIST_DATA_DIR=${MNIST_DATA_DIR:-data}
R=results/cifar10
LOG=$R/logs
mkdir -p "$LOG"
# Download once before launching jobs in parallel.
$PY -c 'import sys; sys.path.insert(0,"experiments/recovery_study"); import engine as e; e.MNIST(str(e.DATA_DIR),train=False,download=True)'

sgd_tune() {
  [ -f $R/recovery_study/baseline_configs.json ] || $PY experiments/recovery_study/study.py tune --threads 1
}
fam() { $PY experiments/optimizer_families/run.py "$@" --threads 1; }

# Slots 1-6: family policy training.
for family in Adam AdamW RMSprop; do
  for seed in 338 339; do
    fam train --job ${family}_gradient_${seed} > $LOG/train_${family}_${seed}.log 2>&1 &
  done
done
# Slots 7-8: SGD policies (one thread each), then the fixed-baseline tuning.
( $PY experiments/recovery_study/study.py train --job gradient_338 --threads 1 > $LOG/train_SGD_338.log 2>&1
  sgd_tune > $LOG/tune_SGD.log 2>&1
  fam tune --family Adam > $LOG/tune_Adam.log 2>&1 ) &
( $PY experiments/recovery_study/study.py train --job gradient_339 --threads 1 > $LOG/train_SGD_339.log 2>&1
  fam tune --family AdamW > $LOG/tune_AdamW.log 2>&1
  fam tune --family RMSprop > $LOG/tune_RMSprop.log 2>&1 ) &
wait

# Evaluations: each needs its tuned baseline and both policies (four processes).
[ -f $R/recovery_study/evaluation.csv ] || $PY experiments/recovery_study/study.py evaluate --threads 1 > $LOG/evaluate_SGD.log 2>&1 &
for family in Adam AdamW RMSprop; do
  fam evaluate --family $family > $LOG/evaluate_${family}.log 2>&1 &
done
wait
$PY experiments/optimizer_families/run.py import > $LOG/import.log 2>&1
$PY scripts/analyze_regular.py > $LOG/analyze.log 2>&1
echo CIFAR10 RUN COMPLETE
