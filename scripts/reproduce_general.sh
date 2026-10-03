#!/usr/bin/env bash
# Joint 14-dataset experiment: PDDR-DOLS, NSGA-II, MOEA/D, always delete-only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 SCIKIT_LEARN_DATA="$ROOT/data/cache"

DATASETS=(iris wine breast_cancer banknote segment dry_bean magic spambase optdigits vehicle page_blocks letter semeion yeast)
CSV="$(IFS=,; echo "${DATASETS[*]}")"
SEEDS="20260820,20260821,20260822,20260823,20260824"
ALGORITHMS="nsga2,moead,pddr_local_search_delete_only,local_search_always_delete_only"
OUT="results/general_14"
mkdir -p "$OUT/raw_data"

for dataset in "${DATASETS[@]}"; do
  raw="$OUT/raw_data/joint_moead_${dataset}_seed_20260820.json"
  if [[ ! -s "$raw" ]]; then
    echo "RUN $dataset"
    python -u code/run_batch.py \
      --datasets "$dataset" --algorithms "$ALGORITHMS" \
      --seeds 20260820 --optimization-seeds "$SEEDS" \
      --n-splits 5 --val-size 0.2 --pool-size 12 \
      --pop-size 40 --n-gen 50 --timing-repeats 5 \
      --include-exact --exact-max-classifiers 12 \
      --data-home "$SCIKIT_LEARN_DATA" --output "$raw"
  fi
  # Validate even skipped files before computing summary statistics.
  python code/validate_p0_protocol.py \
    --input "$raw" --datasets "$dataset" --data-seed 20260820 \
    --optimization-seeds "$SEEDS" --algorithms "$ALGORITHMS" \
    --n-splits 5 --pop-size 40 --n-gen 50 \
    --stage GENERAL_BASELINE_DATASET \
    --purpose "shared classifier pool and evaluation budget" \
    --lock-output "$OUT/protocol_${dataset}.json"
done

python code/compute_unified_metrics.py --raw-dir "$OUT/raw_data" \
  --output "$OUT/unified_metrics.json"
python code/analyze_moead_baseline.py \
  --metrics "$OUT/unified_metrics.json" --raw-dir "$OUT/raw_data" \
  --raw-prefix joint_moead --datasets "$CSV" --data-seed 20260820 \
  --optimization-seeds "$SEEDS" --n-splits 5 --expected-evaluations 2000 \
  --bootstrap-samples 20000 --bootstrap-seed 20260822 \
  --output "$OUT/moead_baseline_analysis.json" \
  --text-output "$OUT/moead_baseline_analysis.txt" \
  --dataset-csv "$OUT/moead_dataset_means.csv"
