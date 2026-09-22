#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CALIBRATION_DIR="${1:-${PROJECT_ROOT}/datasets/bottle/train/good}"

python3 "${PROJECT_ROOT}/jetson/scripts/build_int8_engine.py" \
    --onnx "${PROJECT_ROOT}/models/patchcore_feature_extractor.onnx" \
    --engine "${PROJECT_ROOT}/models/patchcore_feature_extractor_int8.engine" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --cache "${PROJECT_ROOT}/models/patchcore_int8_calibration.cache" \
    --images 100 \
    --width 256 \
    --height 256
