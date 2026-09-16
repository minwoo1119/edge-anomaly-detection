#!/bin/bash

set -e

echo "======================================"
echo " Jetson Environment Check"
echo "======================================"

echo
echo "[L4T / JetPack]"
cat /etc/nv_tegra_release || true

echo
echo "[JetPack Package]"
dpkg-query --show nvidia-jetpack 2>/dev/null || true

echo
echo "[CUDA]"
nvcc --version || true

echo
echo "[TensorRT]"
dpkg -l | grep -i tensorrt || true

echo
echo "[trtexec]"

if command -v trtexec &> /dev/null; then
    which trtexec
    trtexec --version
elif [ -f /usr/src/tensorrt/bin/trtexec ]; then
    /usr/src/tensorrt/bin/trtexec --version
else
    echo "trtexec not found."
fi

echo
echo "[OpenCV]"
pkg-config --modversion opencv4 || true

echo
echo "[CMake]"
cmake --version | head -n 1 || true

echo
echo "[G++]"
g++ --version | head -n 1 || true

echo
echo "======================================"
echo " Environment check completed"
echo "======================================"