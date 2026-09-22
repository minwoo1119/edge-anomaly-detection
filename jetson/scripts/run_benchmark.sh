#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 <config.yaml> <image> [benchmark.csv] [power_summary.csv]" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$1"
IMAGE="$2"
BENCHMARK_CSV="${3:-${PROJECT_ROOT}/results/csv/benchmark.csv}"
POWER_CSV="${4:-${PROJECT_ROOT}/results/csv/power.csv}"
TEGRALOG="$(mktemp --suffix=.tegrastats.log)"
TEGRAPID=""

cleanup() {
    if [[ -n "${TEGRAPID}" ]] && kill -0 "${TEGRAPID}" 2>/dev/null; then
        kill "${TEGRAPID}" 2>/dev/null || true
        wait "${TEGRAPID}" 2>/dev/null || true
    fi
    rm -f -- "${TEGRALOG}"
}
trap cleanup EXIT INT TERM

if [[ ! -x "${PROJECT_ROOT}/jetson/build/edge_anomaly" ]]; then
    echo "ERROR: build jetson/build/edge_anomaly first." >&2
    exit 1
fi
if ! command -v tegrastats >/dev/null 2>&1; then
    echo "ERROR: tegrastats not found." >&2
    exit 1
fi

tegrastats --interval 100 >"${TEGRALOG}" &
TEGRAPID=$!
cd "${PROJECT_ROOT}"
"${PROJECT_ROOT}/jetson/build/edge_anomaly" \
    --config "${CONFIG}" \
    --image "${IMAGE}" \
    --benchmark-csv "${BENCHMARK_CSV}"
kill "${TEGRAPID}" 2>/dev/null || true
wait "${TEGRAPID}" 2>/dev/null || true
TEGRAPID=""

python3 "${PROJECT_ROOT}/jetson/scripts/parse_tegrastats.py" \
    --tegrastats "${TEGRALOG}" \
    --benchmark-csv "${BENCHMARK_CSV}" \
    --output "${POWER_CSV}"
