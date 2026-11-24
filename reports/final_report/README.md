# Final report

`final_report.pdf` is the final report in the NeurIPS 2025 style (`neurips_2025.sty`,
`preprint` option). The body is under 6 pages, ending on page 6, followed by blank Contributions and AI Usage Statement sections, references, and appendices with method, setup, and reproduction details. The paper is authored by Nick Rui, Sandra Yang, and Aidan Whitedeer,
as listed in `../project_update/main.tex`. **The Contributions and AI Usage Statement sections are intentionally
blank; the authors must supply that text.**

## Build

From this directory:

```sh
latexmk -C -jobname=final_report main.tex
latexmk -pdf -jobname=final_report -interaction=nonstopmode -halt-on-error main.tex
pdfinfo final_report.pdf | grep Pages
```

Dependencies: `neurips_2025.sty` (included), `texlive-latex-extra`,
`texlive-fonts-recommended`, `texlive-bibtex-extra`, `texlive-science`, `biber`,
and `poppler-utils` for PDF checks. The build uses biblatex's `ieee` style and
biber, not BibTeX. Tested with TeX Live 2022/dev/Debian, latexmk 4.76, biber 2.17.
The PDF is intentionally tracked; auxiliary build files are not deliverables.

`figures/gains.tex` is the editable pgfplots/TikZ grouped bar chart.
`figures/momentum_block.pdf` is copied without modification from
`../project_update/momentum_block.pdf`; its original source is
`scripts/plot_momentum.py`. Its bands are interquartile ranges, not CIs.

## Numerical provenance

The paper synthesizes saved experiment records from PRs #1–#11, all of which are
now merged into `main`; each result directory below is in the working tree. No
classifier or policy was retrained while preparing the report. PR links are
`https://github.com/nick-rui/cs238/pull/<number>`. The pinned commits record the
exact pre-merge sources; the merged result files under the listed paths are
identical to them.

`table_number_sources.md` lists every numerical table cell and its exact line in
the preparatory digest `/home/ubuntu/final_report_digest.md`, with the source
branch, pinned commit, and repository path. It also audits all chart bars. The
digest itself is a session artifact, not required for building the paper; quoted
source lines and immutable commits are retained in the audit for portability.

| PR | Branch under `origin/devin/` | Pinned commit | Principal source directory |
|---|---|---|---|
| 1 | `1794944000-open-loop-controls` | `ccbe5378ccf135aed4149c2c02323e8d70c55d45` | `results/open_loop/regular/` |
| 2 | `1794943599-compute-matched-baseline` | `d67d0c13d2983add534288ee6e2abdbdaa55e57e` | `results/compute_matched_baseline/` |
| 3 | `1794943493-policy-ablations` | `05e17f4e40c4e3af8e3d62ce163342f80c9ac48c` | `results/ablations/` |
| 4 | `1794943419-additional-policy-seeds` | `4e24b070613967eaf1b8e47d93ac25c20c560116` | `results/additional_seeds/` |
| 5 | `1794955081-extreme-retrain` | `a5200d5167842c87a2d99ccd77b7ecb96970ca9e` | `results/extreme_retrained/` |
| 6 | `1794961752-paired-bootstrap` | `80fb3ad7d2e38bce16918720a3843a83eb16cb05` | `results/paired_bootstrap/` |
| 7 | `1794961726-ablation-start-values` | `601913154463fb61ba11760df706164b1002fc78` | `results/ablations_start_values/` |
| 8 | `1794961736-fashion-mnist` | `320475b3afb9a51eb1d91a86433964c017a90c7a` | `results/fashion_mnist/`, `results/cifar10_timing.json` |
| 9 | `1794961724-extreme-retrain-five-seeds` | `711363e662de5bb88f8c5a4b2e1e87045b92c249` | `results/extreme_retrained_five_seeds/` |
| 10 | `1794961778-warm-start-policies` | `836228f12ffdc6fd2dbcf87dda30841ab2de65e5` | `results/warm_start/` |
| 11 | `1794961735-held-out-orderings` | `eece480e9b31df210bd157f3377c948e2c1529fb` | `results/extreme_held_out/` |

Records are on `main` at the listed paths. To view the exact pinned source, for example:

```sh
git fetch origin devin/1794943419-additional-policy-seeds
git show 4e24b070613967eaf1b8e47d93ac25c20c560116:results/additional_seeds/RESULTS.md
```

## Experiment reproduction entry points

All runners are on `main`; the top-level README indexes them. Follow it and each
runner's `--help` for full options, and prefer an **isolated worktree** for reruns. Use Python 3.12 and `.venv/bin/python` with
the repository requirements. Download MNIST/Fashion-MNIST into the ignored data
directory or set `MNIST_DATA_DIR`; no dataset is embedded in the report.

- Original training/tuning/evaluation:
  `experiments/recovery_study/study.py` and
  `experiments/optimizer_families/run.py` (`train`, `tune`, `evaluate`);
  `experiments/extreme_orderings/run.py evaluate --family <family>` for frozen stress evaluation.
- Additional seeds (#4):
  `.venv/bin/python scripts/additional_seeds.py pipeline --workers 4 --seeds 340 341 342`.
- Compute matching (#2): `--grid wide --output-dir <fresh-dir>` on tuners;
  evaluators support `--fixed-only --config-path --output-dir`. The result README
  supplies the full paired old/new commands and expanded grid.
- Open loop (#1): `.venv/bin/python experiments/open_loop/run.py evaluate`, then
  `summarize`; `experiments/open_loop/check.py` verifies replay and means.
- Start-held ablations (#7, stacked on #3):
  `.venv/bin/python experiments/ablations/run.py train --held-source start`, then
  `evaluate --held-source start`; see `experiments/ablations/analyze.py` for summaries.
- Warm start (#10): `experiments/warm_start/run.py` (`train`, `evaluate`, `analyze`).
- Extreme retraining (#5/#9):
  `.venv/bin/python experiments/extreme_orderings/launch_retrain.py --workers 4`
  for original seeds; add `--five-seeds` for the extension. The extension uses
  the committed frozen cache, or `--frozen-results results/additional_seeds/extreme`
  when rebuilding those seed-matched records. #9 is stacked on #5.
- Held-out types (#11):
  `.venv/bin/python experiments/extreme_orderings/launch_held_out.py --workers 4`.
- Fashion-MNIST (#8): prefix the original runners with
  `CS238_DATASET=fashion_mnist`; outputs are isolated under `results/fashion_mnist/`.
  `scripts/time_cifar10_episode.py` is a timing harness, not an accuracy study.
- Bootstrap (#6): `.venv/bin/python scripts/paired_bootstrap.py`; reads saved
  records, resolving missing working-tree inputs from experiment refs. It covers
  the earlier saved comparisons, **not** #7/#9/#10/#11.

Some evaluation commands overwrite original CSVs or skip completed outputs;
read the relevant README before running them. Re-evaluated float32
individual rows can drift across machines/threads even when reported group
means move by no more than about 0.05 pp. The unchanged-main extreme fixed
RMSprop representative replay has a pre-existing ~0.0061 accuracy-fraction
mismatch. Do not loosen tolerances or silently replace saved evidence. Warm-start
and held-out comparisons use fresh, same-environment paired baselines.

## Verification record

The final build, PDF page count, log-warning audit, and page images were checked
in the authoring session. Page renders are session artifacts under
`/home/ubuntu/final_report_pages/page-{1..6}.png`; regenerate with:

```sh
pdftoppm -r 60 -png final_report.pdf /path/to/existing/output-dir/page
```

The report build does not run any experiments. No implementation tests were
rerun for this documentation-only change.
