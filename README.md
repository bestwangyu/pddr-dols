# PDDR-DOLS

Reproduction code for PDDR-triggered Delete-Only Local Search (PDDR-DOLS)
in heterogeneous ensemble pruning. The historical JSON identifier for the
method is `pddr_local_search_delete_only`; this is the same algorithm, not an
additional method. The repository includes source, tests, and the archived
paper results in `data/` (see `data/README.md`). Downloaded input datasets,
figures, manuscripts, and local caches are not distributed.

## Setup

Python 3.10 and Bash are required. On a machine with Conda:

```bash
conda create -n pddr-dols python=3.10 -y
conda activate pddr-dols
python -m pip install -r requirements.txt
```

The 14-dataset experiments use scikit-learn built-ins (iris, wine,
breast_cancer), OpenML version 1 (banknote, segment, spambase, magic,
optdigits, vehicle, page_blocks, letter, semeion, yeast), and UCI dataset 602
(dry_bean). First access to OpenML/UCI needs an internet connection. Cached
downloads are placed in `data/cache/` and excluded from Git. Availability and
upstream dataset licenses are governed by the respective providers.

## Reproduce

Run from the repository root with the environment activated:

```bash
python -m pytest -q code/test_*.py
bash scripts/reproduce_general.sh
bash scripts/reproduce_domain.sh
```

The first script runs the matched 14-dataset PDDR-DOLS, NSGA-II, MOEA/D, and
always delete-only comparison: five outer folds per dataset, one split seed
(20260820), five optimizer seeds (20260820-20260824), 12 classifiers, 40
individuals and 50 generations (2,000 evaluations per run). It enumerates
the 3-objective exact reference front, validates each dataset's budget and
shared objective context, computes IGD+ and HV, and reports dataset-level
statistics with 95% bootstrap confidence intervals. Results appear under
`results/general_14/`, including `moead_baseline_analysis.json` and
`moead_dataset_means.csv`.

The second script uses the same split/seed/pool design for the PEP and corrected
MDEP paper reimplementations, NSGA-II and PDDR-DOLS, plus separate native
exact reference fronts for PEP (2 objectives) and MDEP (3 objectives). It
uses 25 PEP iterations; PEP's evaluation accounting differs from the three
2,000-evaluation algorithms. It reports native-space metrics separately and
14-dataset statistics for common outcomes under `results/domain_14/`. Do not
interpret cross-objective-space HV or IGD+ as directly comparable.

Both scripts are long-running and write one raw JSON per dataset; an existing
file is skipped. The general script validates skipped files, and the domain
merge rejects missing, duplicate, or wrong-size files. Delete only the affected
generated JSON if an interrupted run left a corrupt file. Run general and
domain experiments in separate invocations: wall-clock costs and measured
inference time may vary across hardware/OS/BLAS versions. Aggregate performance
claims should be checked from rerun outputs rather than assumed bit-identical
to one host's historical measurements.

For a network-free functional smoke check before a long run:

```bash
python code/run_batch.py --datasets iris \
  --algorithms nsga2,moead,pddr_local_search_delete_only,local_search_always_delete_only \
  --seeds 20260820 --optimization-seeds 20260820 --n-splits 2 \
  --pool-size 4 --pop-size 8 --n-gen 2 --timing-repeats 1 \
  --include-exact --exact-max-classifiers 4 \
  --output results/smoke/joint_iris_seed_20260820.json
```

This writes 10 records (eight algorithm runs and two exact references). It is
only a pipeline check; it cannot reproduce the paper's reported statistics.

## Repository map

- `data/`: archived paper results, protocols, checksums, and table/figure map.
- `code/run_batch.py`, `code/batch_runner.py`: shared experiment protocol.
- `code/pddr_local_search.py`, `code/pddrff.py`: PDDR-DOLS and trigger logic.
- `code/fair_baselines.py`, `code/pep_paper.py`, `code/mdep_paper.py`: comparison methods.
- `code/data_loader.py`, `code/classifier_pool.py`, `code/ensemble_pruning.py`: data, common pool, cost and objective evaluation.
- `code/compute_unified_metrics.py`, `code/analyze_moead_baseline.py`, `code/analyze_native_exact.py`, `code/analyze_domain_baseline_statistics.py`: paper analyses.

The source code is available under the MIT License (see `LICENSE`). GitHub's
"Cite this repository" function reads `CITATION.cff`, which identifies Yu Wang
as the sole software citation author. The repository is
https://github.com/bestwangyu/pddr-dols. No release DOI or article DOI is set
yet. Create a versioned GitHub release for Zenodo archiving; do not treat a
DOI as already assigned. No `rg` executable is required by the scripts.
