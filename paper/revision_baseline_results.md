# Revision R1 Exact-GPU Baseline Results

## Controlled exact-retrieval matrix

The CPU, five custom CUDA variants, and FP32 cuBLAS exact-L2 backend were
evaluated on the Bottle 10% FP32 memory bank with the FP16 feature engine and
S0 data path at commit `d042e41`. Every row uses the same clean Release binary,
engine, memory bank, input image, evaluation dataset, 50 warm-up iterations,
and 200 measured iterations. Each backend has three performance repetitions
and one complete 83-image accuracy evaluation.

| Backend | NN mean (ms) | E2E mean (ms) | E2E p95 (ms) | Power (W) | Energy (J/image) |
|---|---:|---:|---:|---:|---:|
| CPU | 14395.536 | 14463.545 | 14511.370 | 6.265 | 90.610 |
| Direct CUDA | 4315.330 | 4422.354 | 4426.831 | 15.640 | 69.166 |
| Warp CUDA | 736.410 | 843.809 | 846.982 | 10.951 | 9.241 |
| Tiled CUDA | 523.054 | 625.913 | 670.998 | 16.165 | 10.116 |
| Async tiled CUDA | 441.627 | 542.195 | 604.645 | 12.741 | 6.907 |
| Async tiled + transpose | 422.215 | 524.181 | 528.375 | 15.335 | 8.062 |
| cuBLAS | **69.221** | **140.361** | **153.092** | 11.153 | **1.565** |

Values are means across the three repetitions; each run-level p95 is first
computed over its 200 measurements and then averaged for the table. All seven
backends retain image AUROC 1.000000 and pixel AUROC approximately 0.985455.
The exact cuBLAS value is 0.985454527; the small last-digit differences among
backends do not change the reported accuracy.

Under this controlled configuration, cuBLAS is 103.05x faster than CPU and
3.73x faster than the best custom CUDA variant in end-to-end mean latency. Its
NN stage is 207.97x faster than CPU and 6.10x faster than the best custom CUDA
variant. This changes the paper's interpretation: the custom kernels establish
the optimization path and its failure modes, while cuBLAS is the strongest
implemented exact-retrieval baseline and the current deployment candidate.

The reported CUDA memory value is a before/after device-memory delta, not peak
GPU allocation. Its three-run mean for cuBLAS is 456.772 MB and it must not be
presented as a peak-memory result.

Canonical local evidence:

- `results/processed/revision_r1_d042_nn_backend_results.csv`;
- `results/csv/runtime_accuracy_r1_d042.csv`;
- `results/metadata/revision_nn_backend_r1_d042_matrix.json`;
- `results/processed/revision_r1_d042_<backend>_run_summary.csv`.

The matrix uses executable SHA-256
`a2d67ed12dc01c3d6e9958f5a72c8c476a1b59bd41b47b22b1cfa572ce8d5884`.
The engine, memory-bank, image, and evaluation-dataset hashes each have exactly
one unique value across all 21 performance rows. The seven configuration hashes
differ only because each configuration selects a different retrieval backend.

## Matched FAISS GPU and cuBLAS result

FAISS v1.15.1 was built from source on aarch64 with CUDA 12.6 and compute
architecture 87. JetPack's CMake 3.22.1 was too old, so CMake 3.31.6 was
installed inside the ignored artifact tree without replacing the system tool.
The resulting `GpuIndexFlatL2` backend and cuBLAS backend were then rebuilt into
the same clean executable at commit `63bb4ec` and evaluated with common engine,
bank, image, and dataset hashes.

| Backend | Image AUROC | Pixel AUROC | NN mean (ms) | E2E mean (ms) | E2E p95 (ms) | Energy (J/image) | Device-memory delta (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| cuBLAS | 1.000000 | 0.985454527 | 68.584 | **138.594** | **151.351** | **1.573** | **553.428** |
| FAISS GPU | 1.000000 | 0.985454528 | 68.728 | 139.385 | 152.588 | 1.599 | 1084.787 |

Both rows are means over three 50/200 runs, apart from AUROC, which uses the
complete 83-image test set. FAISS is 0.57% slower in E2E mean latency, a small
difference relative to run variation, and its device-memory delta is 1.96x the
cuBLAS value. This delta is not a peak-memory metric. The evidence supports
performance parity for the tested exact search while favoring the simpler
cuBLAS backend on measured memory use.

Canonical local evidence:

- `results/processed/r2_exact_gpu_libraries_results.csv`;
- `results/metadata/faiss_gpu_build_1.15.1.json`;
- `results/metadata/fp16_faiss_gpu_evaluation_r1.json`;
- `results/metadata/fp16_cublas_evaluation_r2.json`.

Nsight Systems profiles identify SGEMM as the dominant retrieval kernel for
both implementations. Nsight Compute hardware-counter collection still
requires a privileged `sudo ncu` run on the Jetson.
