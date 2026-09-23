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
TEGRAPID=""

cleanup() {
    if [[ -n "${TEGRAPID}" ]] && kill -0 "${TEGRAPID}" 2>/dev/null; then
        kill "${TEGRAPID}" 2>/dev/null || true
        wait "${TEGRAPID}" 2>/dev/null || true
    fi
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
mkdir -p "$(dirname "${TEGRALOG}")"

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
python3 "${PROJECT_ROOT}/jetson/scripts/validate_artifact_chain.py" \
    --engine "${ENGINE}" \
    --memory-bank "${BANK}" \
    --engine-manifest "${ENGINE_MANIFEST}" \
    --export-manifest "${EXPORT_MANIFEST}" \
    --precision "${PRECISION}"

CONFIG_SHA256="$(sha256sum "${CONFIG}" | awk '{print $1}')"
IMAGE_SHA256="$(sha256sum "${IMAGE}" | awk '{print $1}')"
ENGINE_SHA256="$(sha256sum "${ENGINE}" | awk '{print $1}')"
BANK_SHA256="$(sha256sum "${BANK}" | awk '{print $1}')"

tegrastats --interval 100 >"${TEGRALOG}" &
TEGRAPID=$!
cd "${PROJECT_ROOT}"
"${PROJECT_ROOT}/jetson/build/edge_anomaly" \
    --config "${CONFIG}" \
    --image "${IMAGE}" \
    --benchmark-csv "${BENCHMARK_CSV}" \
    --run-id "${RUN_ID}" \
    --config-sha256 "${CONFIG_SHA256}" \
    --engine-sha256 "${ENGINE_SHA256}" \
    --memory-bank-sha256 "${BANK_SHA256}" \
    --image-sha256 "${IMAGE_SHA256}"
kill "${TEGRAPID}" 2>/dev/null || true
wait "${TEGRAPID}" 2>/dev/null || true
TEGRAPID=""

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
    --output "${MANIFEST}"
