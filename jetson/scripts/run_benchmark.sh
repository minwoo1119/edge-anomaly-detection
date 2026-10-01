#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
    echo "Usage: $0 <config.yaml> <image> [benchmark.csv] [power_summary.csv] [run_id] [manifest.json]" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$1"
IMAGE="$2"
BENCHMARK_CSV="${3:-${PROJECT_ROOT}/results/csv/benchmark.csv}"
POWER_CSV="${4:-${PROJECT_ROOT}/results/csv/power.csv}"
RUN_ID="${5:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
MANIFEST="${6:-${PROJECT_ROOT}/results/metadata/${RUN_ID}.json}"
TEGRALOG="${PROJECT_ROOT}/results/raw/tegrastats/${RUN_ID}.log"
CONTROL_DIR="${PROJECT_ROOT}/results/raw/control/${RUN_ID}"
READY_FILE="${CONTROL_DIR}/warmup.ready"
START_FILE="${CONTROL_DIR}/measurement.start"
TEGRAPID=""
BENCHPID=""

cleanup() {
    if [[ -n "${BENCHPID}" ]] && kill -0 "${BENCHPID}" 2>/dev/null; then
        kill "${BENCHPID}" 2>/dev/null || true
        wait "${BENCHPID}" 2>/dev/null || true
    fi
    if [[ -n "${TEGRAPID}" ]] && kill -0 "${TEGRAPID}" 2>/dev/null; then
        kill "${TEGRAPID}" 2>/dev/null || true
        wait "${TEGRAPID}" 2>/dev/null || true
    fi
    rm -f -- "${READY_FILE}" "${START_FILE}"
    rmdir -- "${CONTROL_DIR}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ ! -x "${PROJECT_ROOT}/jetson/build/edge_anomaly" ]]; then
    echo "ERROR: build jetson/build/edge_anomaly first." >&2
    exit 1
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: run_id may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 1
fi
if [[ ! -f "${CONFIG}" || ! -f "${IMAGE}" ]]; then
    echo "ERROR: config and image must be existing regular files." >&2
    exit 1
fi
if ! command -v tegrastats >/dev/null 2>&1; then
    echo "ERROR: tegrastats not found." >&2
    exit 1
fi
if ! command -v sha256sum >/dev/null 2>&1; then
    echo "ERROR: sha256sum not found." >&2
    exit 1
fi
if [[ -e "${TEGRALOG}" ]]; then
    echo "ERROR: refusing to overwrite raw power log: ${TEGRALOG}" >&2
    exit 1
fi
if [[ -e "${MANIFEST}" ]]; then
    echo "ERROR: refusing to overwrite experiment manifest: ${MANIFEST}" >&2
    exit 1
fi
if [[ -e "${CONTROL_DIR}" ]]; then
    echo "ERROR: refusing to reuse benchmark control directory: ${CONTROL_DIR}" >&2
    exit 1
fi
mkdir -p "$(dirname "${TEGRALOG}")"
mkdir -p "${CONTROL_DIR}"

CONFIG="$(realpath "${CONFIG}")"
IMAGE="$(realpath "${IMAGE}")"
ENGINE_REL="$(awk -F ': *' '$1 == "engine_path" {print $2; exit}' "${CONFIG}")"
BANK_REL="$(awk -F ': *' '$1 == "memory_bank_path" {print $2; exit}' "${CONFIG}")"
ENGINE_MANIFEST_REL="$(awk -F ': *' '$1 == "engine_manifest_path" {print $2; exit}' "${CONFIG}")"
EXPORT_MANIFEST_REL="$(awk -F ': *' '$1 == "export_manifest_path" {print $2; exit}' "${CONFIG}")"
if [[ -z "${ENGINE_REL}" || -z "${BANK_REL}" || -z "${ENGINE_MANIFEST_REL}" || -z "${EXPORT_MANIFEST_REL}" ]]; then
    echo "ERROR: config must define engine, bank, engine-manifest, and export-manifest paths." >&2
    exit 1
fi
if [[ "${ENGINE_REL}" = /* ]]; then ENGINE="${ENGINE_REL}"; else ENGINE="${PROJECT_ROOT}/${ENGINE_REL}"; fi
if [[ "${BANK_REL}" = /* ]]; then BANK="${BANK_REL}"; else BANK="${PROJECT_ROOT}/${BANK_REL}"; fi
if [[ "${ENGINE_MANIFEST_REL}" = /* ]]; then ENGINE_MANIFEST="${ENGINE_MANIFEST_REL}"; else ENGINE_MANIFEST="${PROJECT_ROOT}/${ENGINE_MANIFEST_REL}"; fi
if [[ "${EXPORT_MANIFEST_REL}" = /* ]]; then EXPORT_MANIFEST="${EXPORT_MANIFEST_REL}"; else EXPORT_MANIFEST="${PROJECT_ROOT}/${EXPORT_MANIFEST_REL}"; fi
if [[ ! -f "${ENGINE}" || ! -f "${BANK}" ]]; then
    echo "ERROR: engine_path and memory_bank_path must resolve to existing project files." >&2
    exit 1
fi
PRECISION="$(awk -F ': *' '$1 == "precision" {print $2; exit}' "${CONFIG}")"
CORESET_RATIO="$(awk -F ': *' '$1 == "coreset_ratio" {print $2; exit}' "${CONFIG}")"
DECISION_ENABLED="$(awk -F ': *' '$1 == "decision_enabled" {print $2; exit}' "${CONFIG}")"
THRESHOLD="$(awk -F ': *' '$1 == "threshold" {print $2; exit}' "${CONFIG}")"
THRESHOLD_SOURCE="$(awk -F ': *' '$1 == "threshold_source" {print $2; exit}' "${CONFIG}")"
CATEGORY="$(awk -F ': *' '$1 == "category" {print $2; exit}' "${CONFIG}")"
THRESHOLD_MANIFEST_REL="$(awk -F ': *' '$1 == "threshold_manifest_path" {print $2; exit}' "${CONFIG}")"
THRESHOLD_ARGS=()
MANIFEST_THRESHOLD_ARGS=()
if [[ "${DECISION_ENABLED}" == "true" ]]; then
    if [[ -z "${THRESHOLD_MANIFEST_REL}" ]]; then
        echo "ERROR: decision-enabled config must define threshold_manifest_path." >&2
        exit 1
    fi
    if [[ "${THRESHOLD_MANIFEST_REL}" = /* ]]; then
        THRESHOLD_MANIFEST="${THRESHOLD_MANIFEST_REL}"
    else
        THRESHOLD_MANIFEST="${PROJECT_ROOT}/${THRESHOLD_MANIFEST_REL}"
    fi
    THRESHOLD_ARGS=(--threshold-manifest "${THRESHOLD_MANIFEST}")
    MANIFEST_THRESHOLD_ARGS=(--threshold-manifest "${THRESHOLD_MANIFEST}")
fi
python3 "${PROJECT_ROOT}/jetson/scripts/validate_artifact_chain.py" \
    --engine "${ENGINE}" \
    --memory-bank "${BANK}" \
    --engine-manifest "${ENGINE_MANIFEST}" \
    --export-manifest "${EXPORT_MANIFEST}" \
    --precision "${PRECISION}" \
    --coreset-ratio "${CORESET_RATIO}" \
    --category "${CATEGORY}" \
    --decision-enabled "${DECISION_ENABLED}" \
    --threshold "${THRESHOLD}" \
    --threshold-source "${THRESHOLD_SOURCE}" \
    "${THRESHOLD_ARGS[@]}"

CONFIG_SHA256="$(sha256sum "${CONFIG}" | awk '{print $1}')"
IMAGE_SHA256="$(sha256sum "${IMAGE}" | awk '{print $1}')"
ENGINE_SHA256="$(sha256sum "${ENGINE}" | awk '{print $1}')"
BANK_SHA256="$(sha256sum "${BANK}" | awk '{print $1}')"

cd "${PROJECT_ROOT}"
"${PROJECT_ROOT}/jetson/build/edge_anomaly" \
    --config "${CONFIG}" \
    --image "${IMAGE}" \
    --benchmark-csv "${BENCHMARK_CSV}" \
    --run-id "${RUN_ID}" \
    --config-sha256 "${CONFIG_SHA256}" \
    --engine-sha256 "${ENGINE_SHA256}" \
    --memory-bank-sha256 "${BANK_SHA256}" \
    --image-sha256 "${IMAGE_SHA256}" \
    --benchmark-ready "${READY_FILE}" \
    --benchmark-start "${START_FILE}" &
BENCHPID=$!

for _ in $(seq 1 1200); do
    if [[ -f "${READY_FILE}" ]]; then break; fi
    if ! kill -0 "${BENCHPID}" 2>/dev/null; then
        wait "${BENCHPID}"
        echo "ERROR: benchmark exited before completing warm-up." >&2
        exit 1
    fi
    sleep 0.05
done
if [[ ! -f "${READY_FILE}" ]]; then
    echo "ERROR: timed out waiting for benchmark warm-up." >&2
    exit 1
fi

tegrastats --interval 100 >"${TEGRALOG}" &
TEGRAPID=$!
touch "${START_FILE}"
set +e
wait "${BENCHPID}"
BENCHMARK_STATUS=$?
set -e
BENCHPID=""
kill "${TEGRAPID}" 2>/dev/null || true
wait "${TEGRAPID}" 2>/dev/null || true
TEGRAPID=""
if [[ ${BENCHMARK_STATUS} -ne 0 ]]; then
    echo "ERROR: benchmark executable failed with status ${BENCHMARK_STATUS}." >&2
    exit "${BENCHMARK_STATUS}"
fi

python3 "${PROJECT_ROOT}/jetson/scripts/parse_tegrastats.py" \
    --tegrastats "${TEGRALOG}" \
    --benchmark-csv "${BENCHMARK_CSV}" \
    --run-id "${RUN_ID}" \
    --output "${POWER_CSV}"

python3 "${PROJECT_ROOT}/jetson/scripts/write_experiment_manifest.py" \
    --config "${CONFIG}" \
    --image "${IMAGE}" \
    --engine "${ENGINE}" \
    --engine-manifest "${ENGINE_MANIFEST}" \
    --export-manifest "${EXPORT_MANIFEST}" \
    --memory-bank "${BANK}" \
    --executable "${PROJECT_ROOT}/jetson/build/edge_anomaly" \
    --benchmark-csv "${BENCHMARK_CSV}" \
    --power-csv "${POWER_CSV}" \
    --tegrastats "${TEGRALOG}" \
    --run-id "${RUN_ID}" \
    "${MANIFEST_THRESHOLD_ARGS[@]}" \
    --output "${MANIFEST}"
