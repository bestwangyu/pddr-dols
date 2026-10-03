#!/usr/bin/env bash
# Paper reimplementations: PEP and corrected MDEP with native exact fronts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 SCIKIT_LEARN_DATA="$ROOT/data/cache"

DATASETS=(iris wine breast_cancer banknote segment dry_bean magic spambase optdigits vehicle page_blocks letter semeion yeast)
CSV="$(IFS=,; echo "${DATASETS[*]}")"
OUT="results/domain_14"
mkdir -p "$OUT/raw_data"
for dataset in "${DATASETS[@]}"; do
  raw="$OUT/raw_data/joint_${dataset}_seed_20260820.json"
  if [[ ! -s "$raw" ]]; then
    echo "RUN $dataset"
    python -u code/run_native_exact_joint.py \
      --datasets "$dataset" --seeds 20260820 \
      --optimization-seeds 20260820,20260821,20260822,20260823,20260824 \
      --n-splits 5 --val-size 0.2 --pool-size 12 --pop-size 40 --n-gen 50 \
      --pep-iterations 25 --timing-repeats 5 --exact-max-classifiers 12 \
      --data-home "$SCIKIT_LEARN_DATA" --output "$raw"
  fi
done

python code/merge_joint_json.py --raw-dir "$OUT/raw_data" --datasets "$CSV" \
  --output "$OUT/joint_domain_14.json"
python code/analyze_native_exact.py --input "$OUT/joint_domain_14.json" \
  --output "$OUT/native_metrics.json" --text-output "$OUT/native_metrics.txt"
python code/analyze_domain_baseline_statistics.py \
  --joint-input "$OUT/joint_domain_14.json" --native-input "$OUT/native_metrics.json" \
  --output "$OUT/domain_baseline_statistics.json" \
  --text-output "$OUT/domain_baseline_statistics.txt" \
  --cross-dataset-output "$OUT/cross_dataset_statistics.json" \
  --cross-dataset-text-output "$OUT/cross_dataset_statistics.txt"
