#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <reference_dir> <candidate_dir> <prefix> [report_dir]" >&2
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
    python3 "${PROJECT_ROOT}/src/compare_runtime.py" \
        --reference "${REFERENCE_DIR}/${PREFIX}_${name}.npy" \
        --candidate "${CANDIDATE_DIR}/${PREFIX}_${name}.npy" \
        --max-mae 1e-4 \
        --max-error 1e-3 \
        --min-cosine 0.9999 \
        --report "${REPORT_DIR}/${name}.json"
}

compare_float input
compare_float embedding
compare_float patches
compare_float patch_scores
compare_float raw_score
compare_float anomaly_map

python3 "${PROJECT_ROOT}/src/compare_runtime.py" \
    --reference "${REFERENCE_DIR}/${PREFIX}_nn_indices.npy" \
    --candidate "${CANDIDATE_DIR}/${PREFIX}_nn_indices.npy" \
    --exact \
    --report "${REPORT_DIR}/nn_indices.json"

echo "All runtime correctness comparisons passed. Reports: ${REPORT_DIR}"
