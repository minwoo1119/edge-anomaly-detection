# FAISS GPU on Jetson Orin Nano

FAISS GPU is source-buildable on this Jetson Orin Nano. The verified build used
FAISS v1.15.1, aarch64, CUDA 12.6, compute capability 8.7, GCC 11.4, OpenBLAS,
and CMake 3.31.6. JetPack 6.2.1 supplies CMake 3.22.1, while FAISS v1.15.1
requires CMake 3.24 or newer. The build script installs a newer CMake inside the
ignored `artifacts/` tree when needed; it does not replace the system CMake.

Build and install:

```bash
FAISS_BUILD_JOBS=2 bash jetson/scripts/build_faiss_gpu.sh
```

Configure the runtime with the optional exact-L2 backend:

```bash
cmake -S jetson -B jetson/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DEDGE_ENABLE_FAISS_GPU=ON \
  -DFAISS_ROOT="$PWD/artifacts/faiss-v1.15.1"
cmake --build jetson/build -j2
ctest --test-dir jetson/build --output-on-failure
```

Run the environment probe and one inference:

```bash
python3 jetson/scripts/probe_gpu_baselines.py \
  --faiss-root artifacts/faiss-v1.15.1 \
  --output results/metadata/gpu_baseline_probe_faiss.json
./jetson/build/edge_anomaly \
  --config configs/jetson_fp16_faiss_gpu_s0.yaml \
  --image datasets/MVTecAD/bottle/test/good/000.png
```

The backend uses `faiss::gpu::GpuIndexFlatL2` with an FP32 GPU-resident bank.
FAISS returns squared L2 distance; the adapter applies a clamped square root so
its output matches the runtime's Euclidean-distance contract. S0 timing covers
the synchronous host-query input, GPU search, and host result output. Nsight
reports are required to separate its GPU-resident operator time and transfers.
