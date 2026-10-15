#!/bin/bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
    echo "Usage: $0 <config.yaml> <image> <run_id> <period_ms> <frames> <deadline_ms> <queue_capacity> [tegrastats_interval_ms]" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$(realpath "$1")"
IMAGE="$(realpath "$2")"
RUN_ID="$3"
PERIOD_MS="$4"
FRAMES="$5"
DEADLINE_MS="$6"
QUEUE_CAPACITY="$7"
TEGRA_INTERVAL_MS="${8:-100}"
TRACE="${PROJECT_ROOT}/results/csv/${RUN_ID}_fixed_rate.csv"
TEGRA="${PROJECT_ROOT}/results/raw/tegrastats/${RUN_ID}_fixed_rate.log"
SUMMARY="${PROJECT_ROOT}/results/metadata/${RUN_ID}_fixed_rate.json"
CONTROL="${PROJECT_ROOT}/results/raw/control/${RUN_ID}-fixed-rate"
READY="${CONTROL}/warmup.ready"
START="${CONTROL}/measurement.start"

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then echo "ERROR: invalid run_id" >&2; exit 1; fi
if [[ -e "${TRACE}" || -e "${TEGRA}" || -e "${SUMMARY}" || -e "${CONTROL}" ]]; then
    echo "ERROR: refusing to overwrite fixed-rate outputs" >&2; exit 1
fi
for value in "${FRAMES}" "${QUEUE_CAPACITY}" "${TEGRA_INTERVAL_MS}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then echo "ERROR: integer arguments must be positive" >&2; exit 1; fi
done

ENGINE_REL="$(awk -F ': *' '$1 == "engine_path" {print $2; exit}' "${CONFIG}")"
BANK_REL="$(awk -F ': *' '$1 == "memory_bank_path" {print $2; exit}' "${CONFIG}")"
ENGINE_MANIFEST_REL="$(awk -F ': *' '$1 == "engine_manifest_path" {print $2; exit}' "${CONFIG}")"
EXPORT_MANIFEST_REL="$(awk -F ': *' '$1 == "export_manifest_path" {print $2; exit}' "${CONFIG}")"
resolve_project_path() { if [[ "$1" = /* ]]; then printf '%s' "$1"; else printf '%s/%s' "${PROJECT_ROOT}" "$1"; fi; }
ENGINE="$(resolve_project_path "${ENGINE_REL}")"
BANK="$(resolve_project_path "${BANK_REL}")"
ENGINE_MANIFEST="$(resolve_project_path "${ENGINE_MANIFEST_REL}")"
EXPORT_MANIFEST="$(resolve_project_path "${EXPORT_MANIFEST_REL}")"
PRECISION="$(awk -F ': *' '$1 == "precision" {print $2; exit}' "${CONFIG}")"
RATIO="$(awk -F ': *' '$1 == "coreset_ratio" {print $2; exit}' "${CONFIG}")"
CATEGORY="$(awk -F ': *' '$1 == "category" {print $2; exit}' "${CONFIG}")"
DECISION="$(awk -F ': *' '$1 == "decision_enabled" {print $2; exit}' "${CONFIG}")"
THRESHOLD="$(awk -F ': *' '$1 == "threshold" {print $2; exit}' "${CONFIG}")"
THRESHOLD_SOURCE="$(awk -F ': *' '$1 == "threshold_source" {print $2; exit}' "${CONFIG}")"
python3 "${PROJECT_ROOT}/jetson/scripts/validate_artifact_chain.py" \
  --engine "${ENGINE}" --memory-bank "${BANK}" --engine-manifest "${ENGINE_MANIFEST}" \
  --export-manifest "${EXPORT_MANIFEST}" --precision "${PRECISION}" --coreset-ratio "${RATIO}" \
  --category "${CATEGORY}" --decision-enabled "${DECISION}" --threshold "${THRESHOLD}" \
  --threshold-source "${THRESHOLD_SOURCE}"

mkdir -p "$(dirname "${TRACE}")" "$(dirname "${TEGRA}")" "${CONTROL}"
TEGRAPID=""; RUNTIMEPID=""
cleanup() {
    if [[ -n "${RUNTIMEPID}" ]] && kill -0 "${RUNTIMEPID}" 2>/dev/null; then kill "${RUNTIMEPID}" 2>/dev/null || true; fi
    if [[ -n "${TEGRAPID}" ]] && kill -0 "${TEGRAPID}" 2>/dev/null; then kill "${TEGRAPID}" 2>/dev/null || true; fi
    rm -f -- "${READY}" "${START}"; rmdir -- "${CONTROL}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "${PROJECT_ROOT}"
jetson/build/edge_anomaly --config "${CONFIG}" --image "${IMAGE}" \
  --fixed-rate-csv "${TRACE}" --arrival-period-ms "${PERIOD_MS}" --frame-count "${FRAMES}" \
  --deadline-ms "${DEADLINE_MS}" --queue-capacity "${QUEUE_CAPACITY}" --run-id "${RUN_ID}" \
  --config-sha256 "$(sha256sum "${CONFIG}" | awk '{print $1}')" \
  --engine-sha256 "$(sha256sum "${ENGINE}" | awk '{print $1}')" \
  --memory-bank-sha256 "$(sha256sum "${BANK}" | awk '{print $1}')" \
  --image-sha256 "$(sha256sum "${IMAGE}" | awk '{print $1}')" \
  --executable-sha256 "$(sha256sum jetson/build/edge_anomaly | awk '{print $1}')" \
  --benchmark-ready "${READY}" --benchmark-start "${START}" &
RUNTIMEPID=$!
for _ in $(seq 1 18000); do
    [[ -f "${READY}" ]] && break
    kill -0 "${RUNTIMEPID}" 2>/dev/null || { wait "${RUNTIMEPID}"; exit 1; }
    sleep 0.1
done
[[ -f "${READY}" ]] || { echo "ERROR: warm-up timeout" >&2; exit 1; }
tegrastats --interval "${TEGRA_INTERVAL_MS}" >"${TEGRA}" &
TEGRAPID=$!
touch "${START}"
wait "${RUNTIMEPID}"; RUNTIMEPID=""
kill "${TEGRAPID}" 2>/dev/null || true; wait "${TEGRAPID}" 2>/dev/null || true; TEGRAPID=""
python3 jetson/scripts/analyze_fixed_rate.py --trace "${TRACE}" --tegrastats "${TEGRA}" --output "${SUMMARY}"
