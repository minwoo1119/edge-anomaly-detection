#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ONNX_MODEL="${PROJECT_ROOT}/models/patchcore_feature_extractor.onnx"
ENGINE_MODEL="${PROJECT_ROOT}/models/patchcore_feature_extractor_fp16.engine"

if [[ ! -f "${ONNX_MODEL}" ]]; then
    echo "ERROR: ONNX model not found: ${ONNX_MODEL}" >&2
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

echo "FP16 engine created: ${ENGINE_MODEL}"
