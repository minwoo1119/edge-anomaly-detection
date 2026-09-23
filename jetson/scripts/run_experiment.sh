#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <config.yaml> <image> <experiment_id> [independent_runs]" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$1"
IMAGE="$2"
EXPERIMENT_ID="$3"
INDEPENDENT_RUNS="${4:-3}"

if [[ ! "${EXPERIMENT_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: experiment_id may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 1
fi
if [[ ! "${INDEPENDENT_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: independent_runs must be a positive integer." >&2
    exit 1
fi

BENCHMARK_CSV="${PROJECT_ROOT}/results/csv/${EXPERIMENT_ID}_benchmark.csv"
POWER_CSV="${PROJECT_ROOT}/results/csv/${EXPERIMENT_ID}_power.csv"
for ((run_index = 1; run_index <= INDEPENDENT_RUNS; ++run_index)); do
    printf -v suffix "%02d" "${run_index}"
    run_id="${EXPERIMENT_ID}-${suffix}"
    manifest="${PROJECT_ROOT}/results/metadata/${run_id}.json"
    bash "${PROJECT_ROOT}/jetson/scripts/run_benchmark.sh" \
        "${CONFIG}" \
        "${IMAGE}" \
        "${BENCHMARK_CSV}" \
        "${POWER_CSV}" \
        "${run_id}" \
        "${manifest}"
done

python3 "${PROJECT_ROOT}/src/process_benchmarks.py" \
    --benchmark-csv "${BENCHMARK_CSV}" \
    --power-csv "${POWER_CSV}" \
    --metadata-dir "${PROJECT_ROOT}/results/metadata" \
    --output "${PROJECT_ROOT}/results/processed/${EXPERIMENT_ID}_run_summary.csv"
