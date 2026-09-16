#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

ONNX_MODEL="${PROJECT_ROOT}/models/patchcore_feature_extractor.onnx"

ENGINE_MODEL="${PROJECT_ROOT}/models/patchcore_feature_extractor_fp32.engine"

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

if [ ! -f "${ONNX_MODEL}.data" ]; then
    echo
    echo "ERROR: ONNX external data not found."
    echo "${ONNX_MODEL}.data"
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
    --saveEngine="${ENGINE_MODEL}"

echo
echo "======================================"
echo " TensorRT engine build completed"
echo "======================================"

ls -lh "${ENGINE_MODEL}"