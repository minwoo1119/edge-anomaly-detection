# Revision R2 Profiling Results

## Scope and limitation

Nsight Systems 2024.5.4 captured one cold S0 inference for the cuBLAS and FAISS
GPU exact-L2 backends at executable commit `63bb4ec`. These traces include
runtime initialization and profiling overhead, so their absolute durations are
not substituted for the warmed 50/200 benchmark results. They are used to
identify operator composition, launch counts, transfers, and synchronization.

Nsight Compute did not create a report because Jetson performance counters
require elevated privileges. Each profile directory contains the exact
`sudo ncu` command needed to complete hardware-counter collection. Occupancy
and achieved-memory-bandwidth claims remain pending until those reports exist.

## Kernel composition

| Backend | Dominant kernels in the cold trace | Total traced duration |
|---|---|---:|
| cuBLAS | `ampere_sgemm_128x64_tn` (1 launch) | 106.177 ms |
| cuBLAS | bank/query squared norms (2 launches) | 7.578 ms |
| cuBLAS | L2 reduction (1 launch) | 2.247 ms |
| FAISS GPU | `ampere_sgemm_128x64_tn` (2 launches) | 126.207 ms |
| FAISS GPU | row-major L2 norms (2 launches) | 9.159 ms |
| FAISS GPU | top-1 L2 selection (2 launches) | 1.960 ms |

SGEMM accounts for 84.9% of total GPU-kernel time in both cold traces. FAISS
splits the flat search into two SGEMM and two selection launches for this
21,401 by 1,536 bank and 1,024-query workload; the custom cuBLAS adapter uses
one SGEMM followed by one reduction kernel. The warmed three-run measurements
remain nearly equal: 68.584 ms NN for cuBLAS and 68.728 ms for FAISS.

## Transfers and allocation behavior

| Backend | H2D time / count | D2H time / count | CUDA allocation observations |
|---|---:|---:|---|
| cuBLAS | 31.140 ms / 4 | 1.804 ms / 3 | `cudaMalloc` API total 83.221 ms |
| FAISS GPU | 32.943 ms / 6 | 1.866 ms / 3 | `cudaMalloc` 257.938 ms plus `cudaHostAlloc` 80.060 ms |

These cold-trace allocation totals explain FAISS's larger initialization cost
and are consistent with its larger benchmark device-memory delta. They do not
measure peak memory. Warmed latency shows that allocation occurs outside the
steady search loop and does not materially separate the two exact libraries.

Local reports:

- `results/profiling/r2_cublas_s0_63bb4ec/timeline.nsys-rep`;
- `results/profiling/r2_cublas_s0_63bb4ec/nsys_stats.csv`;
- `results/profiling/r2_faiss_gpu_s0_63bb4ec/timeline.nsys-rep`;
- `results/profiling/r2_faiss_gpu_s0_63bb4ec/nsys_stats.csv`.
