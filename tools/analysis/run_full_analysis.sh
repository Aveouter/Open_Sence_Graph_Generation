#!/bin/bash
# =============================================================================
# LCompo-SGG Full Analysis Pipeline
# =============================================================================
# Runs all analysis steps for the compositional SGG paper.
#
# Usage:
#   bash tools/analysis/run_full_analysis.sh
# =============================================================================

set -e

# Ensure we run from project root
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "============================================"
echo "LCompo-SGG: Full Analysis Pipeline"
echo "============================================"
echo ""

# Step 1.1: Composition Coverage Analysis
echo ">>> Step 1.1: Composition Coverage Analysis..."
python tools/analysis/composition_coverage.py \
    --split train \
    --output_dir reports/composition
echo ""

# Step 1.2: Build Compositional Splits
echo ">>> Step 1.2: Building Compositional Splits (3 random seeds)..."
python tools/analysis/make_compositional_split.py \
    --n_splits 3 \
    --split_ratio 0.75 \
    --output_dir data/VisualGenome/composition_splits
echo ""

# Step 1.3: Generate test set composition labels
echo ">>> Step 1.3: Generating test set composition labels..."
cd tools/analysis && python composition_filter.py \
    --data_root ../../data/VisualGenome \
    --split_path ../../data/VisualGenome/composition_splits/split_0/composition_split.json \
    --eval_split test \
    --output ../../reports/composition/test_composition_labels.json
cd ../..
echo ""

echo "============================================"
echo "Analysis pipeline complete!"
echo ""
echo "Output files:"
echo "  - reports/composition/predicate_composition_stats.csv"
echo "  - reports/composition/freq_vs_composition_coverage.png"
echo "  - reports/composition/coverage_ratio_distribution.png"
echo "  - reports/composition/unique_compositions_distribution.png"
echo "  - reports/composition/top_predicates_comparison.png"
echo "  - data/VisualGenome/composition_splits/split_*/composition_split.json"
echo "  - reports/composition/test_composition_labels.json"
echo ""
echo "Next steps:"
echo "  1. Run baseline evaluation: python train.py --config_file configs/VisualGenome/<method>.py --test --ckpt_path <path>"
echo "  2. Run correlation analysis: python tools/analysis/compute_correlations.py --recall_json <per_pred_recall.json>"
echo "  3. Run composition gap: python tools/analysis/eval_composition.py --per_predicate_json <per_pred_recall.json>"
echo "============================================"
