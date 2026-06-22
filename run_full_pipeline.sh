#!/bin/bash
# Full collapse analysis pipeline: Motifs PredCLS export + metrics + LCompo
# Usage: bash run_full_pipeline.sh [test_dataset_size] [output_tag]
# Example: bash run_full_pipeline.sh 5000 full_test

set -euo pipefail

TEST_SIZE="${1:-5000}"
TAG="${2:-full_test}"
CKPT="outputs/pretrained/motifs/coldmanck/extracted/model_0022000.pth"
EXPORT_DIR="outputs/analysis/${TAG}/Motifs/PredCLS"
METRICS_DIR="outputs/analysis/${TAG}/Motifs/PredCLS_metrics"
LCOMPO_DIR="outputs/analysis/${TAG}/lcompo_f2c/Motifs/PredCLS"
CONDA_ENV="hsg"

echo "============================================"
echo "Full Collapse Pipeline"
echo "============================================"
echo "  Test size:  ${TEST_SIZE}"
echo "  Tag:        ${TAG}"
echo "  Checkpoint: ${CKPT}"
echo "  Conda env:  ${CONDA_ENV}"
echo ""

# Step 1: Export relation predictions
echo "[1/3] Exporting relation predictions..."
conda run -n ${CONDA_ENV} python3 tools/analysis/export_relation_predictions.py \
  --ckpt_path "${CKPT}" \
  --task PredCLS --model Motifs \
  --test_dataset_size "${TEST_SIZE}" \
  --val_batch_size 8 --num_workers 4 --device cuda \
  --output_dir "${EXPORT_DIR}"

echo "  Export done: $(wc -l < ${EXPORT_DIR}/relation_predictions.jsonl) rows"
echo ""

# Step 2: Collapse metrics
echo "[2/3] Computing collapse metrics..."
conda run -n ${CONDA_ENV} python3 tools/analysis/compute_collapse_metrics.py \
  --predictions "${EXPORT_DIR}/relation_predictions.jsonl" \
  --output_dir "${METRICS_DIR}"

echo "  Metrics done"
echo ""

# Step 3: LCompo analysis
echo "[3/3] Running LCompo analysis..."
conda run -n ${CONDA_ENV} python3 tools/analysis/compute_lcompo_f2c_analysis.py \
  --predictions "${EXPORT_DIR}/relation_predictions.jsonl" \
  --split_id 0 \
  --output_dir "${LCOMPO_DIR}"

echo "  LCompo done"
echo ""

echo "============================================"
echo "Pipeline complete"
echo "============================================"
echo "  Export:   ${EXPORT_DIR}/relation_predictions.jsonl"
echo "  Metrics:  ${METRICS_DIR}/collapse_metrics.json"
echo "  LCompo:   ${LCOMPO_DIR}/lcompo_f2c_report.md"
echo ""
echo "Check results:"
echo "  cat ${METRICS_DIR}/fine_to_coarse_collapse_report.md"
echo "  cat ${LCOMPO_DIR}/lcompo_f2c_report.md"
