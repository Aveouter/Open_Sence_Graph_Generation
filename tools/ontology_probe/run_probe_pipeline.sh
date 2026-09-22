#!/usr/bin/env bash
# VG150 predicate-ontology cross-pair generalization diagnostic.
#
# Stages are independently runnable, so a failed or slow step does not force a
# full restart:
#
#   bash tools/ontology_probe/run_probe_pipeline.sh s3
#   bash tools/ontology_probe/run_probe_pipeline.sh s6 --measure --limit-images 200
#
# GPU-free stages are s1..s4; s5..s8 need the visual cache.
set -euo pipefail

PY="${PY:-./.venv/hsg/python.exe}"
[ -x "$PY" ] || PY=python
DRY="${DRY:-0}"
DATA_ROOT="${DATA_ROOT:-data/VisualGenome}"
OUT="${OUT:-outputs/analysis/ontology_probe}"
CACHE="${CACHE:-$OUT/cache}"
ENCODER="${ENCODER:-clip_vit_b16}"

STAGE="${1:-all}"
shift || true

dry_flag() { [ "$DRY" = "1" ] && echo "--dry-run" || true; }

run_s1() {  # predicate / ontology statistics
  echo "=== s1: predicate statistics ==="
  "$PY" tools/ontology_probe/ontology_stats.py \
    --data-root "$DATA_ROOT" --split train \
    --output-dir "$OUT/predicate_stats" --write-predicate-frequencies $(dry_flag)
}

run_s2() {  # canonical label spaces
  echo "=== s2: canonical map ==="
  "$PY" tools/ontology_probe/build_canonical_map.py --data-root "$DATA_ROOT" $(dry_flag)
  "$PY" tools/ontology_probe/validate_canonical_map.py \
    --data-root "$DATA_ROOT" --output-dir "$OUT/predicate_stats"
}

run_s3() {  # splits + leakage audit (audit raises; it does not warn)
  echo "=== s3: splits ==="
  "$PY" tools/ontology_probe/build_splits.py \
    --data-root "$DATA_ROOT" --output-dir "$OUT/splits" $(dry_flag)
}

run_s4() {  # GPU-free prior half of the matrix
  echo "=== s4: prior matrix (B0, B1_lookup) ==="
  "$PY" tools/ontology_probe/eval_matrix.py --stage prior \
    --data-root "$DATA_ROOT" --output-dir "$OUT/matrix" $(dry_flag)
}

run_s5() {  # M5 gate: prove the features carry signal before trusting them
  echo "=== s5: visual sanity gate ==="
  "$PY" tools/ontology_probe/visual_sanity.py \
    --data-root "$DATA_ROOT" --cache-root "$CACHE" --encoder "$ENCODER" \
    --limit-images "${SANITY_IMAGES:-200}" --output-dir "$OUT/sanity"
}

run_s6() {  # full feature cache
  echo "=== s6: extract visual features ==="
  "$PY" tools/ontology_probe/extract_features.py \
    --data-root "$DATA_ROOT" --splits "${SPLITS:-train val}" \
    --encoder "$ENCODER" --output-root "$CACHE" --accept-projection "$@"
}

run_s7() {  # label/geometry probes, then the visual probes
  echo "=== s7: probes ==="
  "$PY" tools/ontology_probe/train_probes.py --data-root "$DATA_ROOT" \
    --probes B1_add B2 --levels vg50 L1_noise L2_entail \
    --splits iid pair_ood pair_known --output-dir "$OUT/probes" "$@"
  "$PY" tools/ontology_probe/train_probes.py --data-root "$DATA_ROOT" \
    --probes B3 B4 --levels vg50 L1_noise L2_entail \
    --splits iid pair_ood pair_known --cache-dir "$CACHE" --encoder "$ENCODER" \
    --output-dir "$OUT/probes" "$@"
}

run_s8() {  # interventions on the trained visual probe
  echo "=== s8: shuffle intervention (M7) ==="
  for cell in "${SHUFFLE_CELLS:-vg50__pair_ood__B4__s0 L2_entail__pair_ood__B4__s0}"; do
    "$PY" tools/ontology_probe/shuffle_test.py \
      --probes-dir "$OUT/probes" --cache-root "$CACHE" --encoder "$ENCODER" \
      --output-dir "$OUT/shuffle" --cells $cell "$@"
  done
  echo "=== s8: rescue rate (M8) ==="
  "$PY" tools/ontology_probe/rescue_rate.py --probes-dir "$OUT/probes" \
    --output-dir "$OUT/rescue" --pairs \
      vg50__pair_ood__B2__s0:vg50__pair_ood__B4__s0 \
      L2_entail__pair_ood__B2__s0:L2_entail__pair_ood__B4__s0 \
      vg50__iid__B2__s0:vg50__iid__B4__s0 \
      L2_entail__iid__B2__s0:L2_entail__iid__B4__s0 "$@"
}

run_s9() {  # pooled Delta_ontology between label spaces
  echo "=== s9: pooled ontology comparison ==="
  "$PY" tools/ontology_probe/pool_compare.py --probe B2 \
    --probes-dir "$OUT/probes" --output-dir "$OUT/matrix" "$@"
  "$PY" tools/ontology_probe/pool_compare.py --probe B4 \
    --probes-dir "$OUT/probes" --output-dir "$OUT/matrix" "$@"
}

run_s10() {  # per-predicate diagnostic table
  echo "=== s10: diagnostics table (M10) ==="
  "$PY" tools/ontology_probe/diagnostics_table.py \
    --output-root "$OUT" --output-dir "$OUT/diagnostics" "$@"
}

case "$STAGE" in
  all) for s in s1 s2 s3 s4 s5 s6 s7 s8 s9 s10; do "run_$s"; done ;;
  s1|s2|s3|s4|s5|s6|s7|s8|s9|s10) "run_$STAGE" "$@" ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
echo "=== done: $STAGE ==="
