# Paper result data

This directory archives the **generated experimental results** supporting the
PDDR-DOLS manuscript, not copies of the 14 upstream input datasets. Dataset
downloads are performed by the reproduction scripts; OpenML and UCI control
availability and licensing of their respective source datasets. The software
MIT license does not purport to relicense those third-party datasets.

## Contents and manuscript map

| Directory | Contents | Manuscript use |
| --- | --- | --- |
| `general_14/raw_data/` | 14 JSON files, one per dataset, 1,470 raw records in total (1,400 algorithm runs and 70 exact reference fronts) | Main PDDR-DOLS, NSGA-II, MOEA/D and always delete-only comparison; Tables 2-6 and Figure 2 |
| `general_14/unified_metrics.json` | 1,400 per-run metric records | IGD+, HV, error and representative-solution cost analyses |
| `general_14/moead_baseline_analysis.json`, `general_14/moead_dataset_means.csv` | Dataset-level estimates, paired tests, bootstrap intervals and table/plot values | Tables 2-6 and Figure 2 |
| `general_14/protocol_*.json`, `general_14/environment.txt`, `general_14/SHA256SUMS.txt` | Per-dataset protocol validations, original run environment and integrity checks | Reproducibility of the main comparison |
| `domain_14/raw_data/` | Corrected, merged 14-dataset PEP/MDEP experiment: 1,540 records, including 1,400 algorithm runs and 140 native exact-reference records | Domain baseline comparison, Table 7 and Figures 3-4 |
| `domain_14/native_metrics.json`, `domain_14/domain_baseline_statistics.json`, `domain_14/cross_dataset_statistics.json` | 700 native-space records and separate common-downstream/native analyses | Table 7 and Figures 3-4 |
| `domain_14/mdep_rerun/`, `domain_14/mdep_fix_manifest.json` | The 350 corrected MDEP runs and a historical merge manifest with SHA-256 digests | Provenance of corrected MDEP initialization |
| `resource_retest/` | 500 per-step measurement records and the summary report for Banknote, Wine, Dry Bean and Segment | Separate, independent resource replication discussed with Table 6 |

Text versions of analysis reports are retained beside their JSON equivalents.
The domain manifest retains **historical source paths** of the earlier working
directory; `domain_14/raw_data/` is the archived corrected result and
`domain_14/mdep_rerun/` contains its corrected component. The preliminary,
uncorrected MDEP merged file is intentionally not presented as paper evidence.
The independent resource replication is an archived historical measurement;
the two public reproduction scripts cover the main 14-dataset experiments,
not this separate resource timing protocol.

## Experiment design and verification

The general comparison uses five outer folds and five optimizer seeds per
dataset; each of its four algorithms receives 2,000 evaluations per run.
The domain comparison uses the same splits and optimizer seeds. PEP has a
different evaluation-accounting protocol and a two-objective native space;
native-space HV and IGD+ must not be pooled with three-objective values.
Storage and inference measurements, especially elapsed time, depend on the
host's hardware and software environment. The recorded split seed is 20260820,
and optimizer seeds are 20260820 through 20260824 (S1-S5 in the manuscript).

From the repository root, check the archived general files against their
original integrity manifest:

```bash
cd data/general_14
shasum -a 256 -c SHA256SUMS.txt
```

The historical MDEP correction manifest contains SHA-256 digests for the
corrected merged JSON and the 14 corrected rerun files. To generate new
results, follow `scripts/reproduce_general.sh` and
`scripts/reproduce_domain.sh`; they write to `results/`, which is not part of
this immutable paper-data snapshot. Comparing reruns should account for
runtime variability across machines. No Zenodo DOI is assigned in these files.
