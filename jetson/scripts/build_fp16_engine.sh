#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ONNX_MODEL="${PROJECT_ROOT}/models/patchcore_feature_extractor.onnx"
ENGINE_MODEL="${PROJECT_ROOT}/models/patchcore_feature_extractor_fp16.engine"
ENGINE_MANIFEST="${ENGINE_MODEL}.json"
EXPORT_MANIFEST="${PROJECT_ROOT}/models/patchcore_export_manifest.json"

if [[ ! -f "${ONNX_MODEL}" ]]; then
    echo "ERROR: ONNX model not found: ${ONNX_MODEL}" >&2
    exit 1
fi
if [[ -e "${ENGINE_MODEL}" || -e "${ENGINE_MANIFEST}" ]]; then
    echo "ERROR: refusing to overwrite engine or manifest." >&2
    exit 1
fi
if [[ ! -f "${EXPORT_MANIFEST}" ]]; then
    echo "ERROR: export manifest not found: ${EXPORT_MANIFEST}" >&2
    exit 1
fi

if command -v trtexec >/dev/null 2>&1; then
    TRTEXEC="$(command -v trtexec)"
elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
else
    echo "ERROR: trtexec not found." >&2
    exit 1
fi

"${TRTEXEC}" \
    --onnx="${ONNX_MODEL}" \
    --saveEngine="${ENGINE_MODEL}" \
    --fp16 \
    --skipInference

python3 "${PROJECT_ROOT}/jetson/scripts/write_engine_manifest.py" \
    --onnx "${ONNX_MODEL}" \
    --engine "${ENGINE_MODEL}" \
    --precision fp16 \
    --trtexec "${TRTEXEC}" \
    --output "${ENGINE_MANIFEST}" \
    --export-manifest "${EXPORT_MANIFEST}"

echo "FP16 engine created: ${ENGINE_MODEL}"
