#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

ONNX_MODEL="${PROJECT_ROOT}/models/patchcore_feature_extractor.onnx"

ENGINE_MODEL="${PROJECT_ROOT}/models/patchcore_feature_extractor_fp32.engine"
ENGINE_MANIFEST="${ENGINE_MODEL}.json"
EXPORT_MANIFEST="${PROJECT_ROOT}/models/patchcore_export_manifest.json"

echo "======================================"
echo " PatchCore TensorRT FP32 Build"
echo "======================================"

echo "ONNX   : ${ONNX_MODEL}"
echo "ENGINE : ${ENGINE_MODEL}"

if [ ! -f "${ONNX_MODEL}" ]; then
    echo
    echo "ERROR: ONNX model not found."
    echo "${ONNX_MODEL}"
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

if command -v trtexec &> /dev/null; then
    TRTEXEC="$(command -v trtexec)"
elif [ -f /usr/src/tensorrt/bin/trtexec ]; then
    TRTEXEC="/usr/src/tensorrt/bin/trtexec"
else
    echo
    echo "ERROR: trtexec not found."
    exit 1
fi

echo
echo "trtexec: ${TRTEXEC}"
echo

"${TRTEXEC}" \
    --onnx="${ONNX_MODEL}" \
    --saveEngine="${ENGINE_MODEL}" \
    --skipInference

python3 "${PROJECT_ROOT}/jetson/scripts/write_engine_manifest.py" \
    --onnx "${ONNX_MODEL}" \
    --engine "${ENGINE_MODEL}" \
    --precision fp32 \
    --trtexec "${TRTEXEC}" \
    --output "${ENGINE_MANIFEST}" \
    --export-manifest "${EXPORT_MANIFEST}"

echo
echo "======================================"
echo " TensorRT engine build completed"
echo "======================================"

ls -lh "${ENGINE_MODEL}"
