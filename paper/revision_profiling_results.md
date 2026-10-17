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
and achieved-memory-bandwidth claims for cuBLAS and FAISS remain pending until
those reports exist.

An earlier full Nsight Compute capture is available for the custom tiled
kernel at commit `1a9ca6e` in
`results/profiling/tiled_1a9ca6e/kernel.ncu-rep`. Under profiler replay, the
kernel reports 594.97 ms duration, 88.47% achieved occupancy, 41.86% memory
throughput, and 56.48% SM throughput. Long-scoreboard stalls account for about
55.5% of the average interval between issued warp instructions, while no warp
is eligible in 43.41% of scheduler cycles. These counters support a
memory-access-latency diagnosis despite high occupancy. The replayed duration
is not substituted for the warmed benchmark result. Library-kernel occupancy
and bandwidth counters remain pending because the cuBLAS/FAISS reports still
require elevated profiling permission.

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

## Bank-size scaling

The same profiling protocol was repeated with independently generated Bottle
1%, 10%, and 20% banks. The embedding dimension and query count remain 1,536
and 1,024. Within every bank size, the two backends use the same clean
executable, engine, bank, and image hashes.

| Bank ratio | Rows | Backend | Cold retrieval kernels (ms) | Warm NN mean (ms) | Warm E2E mean (ms) | Energy (J/image) |
|---:|---:|---|---:|---:|---:|---:|
| 1% | 2,140 | cuBLAS | 21.176 | 20.554 | 67.679 | 0.437 |
| 1% | 2,140 | FAISS GPU | 31.295 | **19.509** | **65.262** | **0.430** |
| 10% | 21,401 | cuBLAS | **116.001** | **68.584** | **138.594** | **1.573** |
| 10% | 21,401 | FAISS GPU | 137.356 | 68.728 | 139.385 | 1.599 |
| 20% | 42,803 | cuBLAS | **185.973** | 131.635 | 247.137 | 3.002 |
| 20% | 42,803 | FAISS GPU | 197.617 | **118.878** | **236.037** | **2.725** |

Cold profiling exaggerates the library gap, especially for the small bank:
FAISS's extra launches and initialization cost are included once, while the
warmed loop amortizes them. Warmed 1% and 10% latency is effectively close;
at 20%, FAISS reduces NN mean by 9.69% and E2E mean by 4.49% relative to the
cuBLAS adapter. The result supports a bank-size-dependent backend choice rather
than a universal winner.

The measured device-memory delta is not monotonic for cuBLAS across these
runs, confirming that it cannot serve as a bank-scaling or peak-memory metric.
R4 must record deterministic allocation bytes in the implementations and add
an external peak-memory corroboration before making memory-scaling claims.

Local reports:

- `results/profiling/r2_cublas_s0_63bb4ec/timeline.nsys-rep`;
- `results/profiling/r2_cublas_s0_63bb4ec/nsys_stats.csv`;
- `results/profiling/r2_faiss_gpu_s0_63bb4ec/timeline.nsys-rep`;
- `results/profiling/r2_faiss_gpu_s0_63bb4ec/nsys_stats.csv`;
- `results/processed/r2_bank_size_nsys.csv`;
- `results/processed/r2_bank_size_benchmark.csv`.
