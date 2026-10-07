#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: $0 <reference_dir> <candidate_dir> <prefix> [report_dir] [memory_bank_for_fp64_check]" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REFERENCE_DIR="$1"
CANDIDATE_DIR="$2"
PREFIX="$3"
REPORT_DIR="${4:-${CANDIDATE_DIR}/reports}"

mkdir -p "${REPORT_DIR}"

compare_float() {
    local name="$1"
    local max_mae="${2:-1e-4}"
    local max_error="${3:-1e-3}"
    python3 "${PROJECT_ROOT}/src/compare_runtime.py" \
        --reference "${REFERENCE_DIR}/${PREFIX}_${name}.npy" \
        --candidate "${CANDIDATE_DIR}/${PREFIX}_${name}.npy" \
        --max-mae "${max_mae}" \
        --max-error "${max_error}" \
        --min-cosine 0.9999 \
        --report "${REPORT_DIR}/${name}.json"
}

compare_float input
compare_float embedding
compare_float patches
python3 "${PROJECT_ROOT}/src/compare_runtime.py" \
    --reference "${REFERENCE_DIR}/${PREFIX}_nn_indices.npy" \
    --candidate "${CANDIDATE_DIR}/${PREFIX}_nn_indices.npy" \
    --exact \
    --report "${REPORT_DIR}/nn_indices.json"

if [[ $# -eq 5 ]]; then
    # Opt-in numerical profile for FP32 matrix-distance vs direct-distance
    # arithmetic. Preserve strict feature gates, exact indices, and an
    # independent strict FP64 check before applying downstream bounds.
    python3 "${PROJECT_ROOT}/jetson/scripts/validate_nn_distances.py" \
        --bank "$5" \
        --patches "${CANDIDATE_DIR}/${PREFIX}_patches.npy" \
        --indices "${CANDIDATE_DIR}/${PREFIX}_nn_indices.npy" \
        --scores "${CANDIDATE_DIR}/${PREFIX}_patch_scores.npy" \
        --report "${REPORT_DIR}/direct_fp64_distances.json"
    echo "Using opt-in downstream numerical bounds: MAE <= 5e-4, max error <= 5e-3."
    compare_float patch_scores 5e-4 5e-3
    compare_float raw_score 5e-4 5e-3
    compare_float anomaly_map 5e-4 5e-3
else
    compare_float patch_scores
    compare_float raw_score
    compare_float anomaly_map
fi

echo "All runtime correctness comparisons passed. Reports: ${REPORT_DIR}"
