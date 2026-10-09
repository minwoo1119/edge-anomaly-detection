# C++ / CUDA Optimization

## System ablation stage contract

System optimization은 `optimization_stage`와 실제 실행 경로가 일치해야 합니다.

```text
S0: frame마다 device input/output buffer 할당 + pageable host memory + synchronous copy
S1: persistent device input/output buffer
S2: S1 + persistent pinned host staging buffer
S3: S2 + asynchronous H2D/D2H on the default stream
S4: S3 + dedicated non-blocking CUDA stream
S5: GPU-resident NN (TensorRT NCHW output을 CUDA NN이 직접 접근)
S6: S5 + 다음 frame CPU preprocessing을 현재 frame GPU/runtime과 overlap
```

S5는 전체 embedding D2H와 query H2D를 제거하고 patch distance/index 및 image
score reweighting에 필요한 최대-distance patch만 host로 복사합니다. S6는 별도
CPU worker에서 다음 입력을 전처리하는 동안 현재 입력의 TensorRT, GPU NN 및
postprocess를 수행합니다. S6의 overlap과 성능 향상은 Jetson Nsight Systems와
실측 CSV로 검증하기 전에는 논문 결과로 사용하지 않습니다.

환경 정보와 artifact path를 채운 baseline config에서 S0-S6 config를 생성합니다.

```bash
python3 src/generate_system_configs.py \
  --baseline configs/orin_bottle_fp32_cpu.yaml \
  --output-dir results/configs/bottle_system
```

생성 manifest에는 baseline과 각 config의 SHA-256가 기록됩니다.

## 1. 목표

C++을 단순 TensorRT 호출 glue code가 아니라 시스템 최적화의 핵심으로 사용합니다.

포함:
- C++17
- RAII
- smart pointers
- memory ownership
- buffer reuse
- multithreading
- CUDA streams
- pinned memory
- custom kernel
- profiling

---

## 2. RAII

대상:
- TensorRT Runtime
- Engine
- ExecutionContext
- CUDA stream
- device buffer
- pinned host buffer

금지:
- raw ownership
- per-frame malloc/free
- per-frame engine load

---

## 3. Persistent Buffers

startup:
```text
allocate once
```

frame:
```text
reuse
```

비교:
- allocation overhead
- total latency

---

## 4. Pinned Memory

후보:
- `cudaHostAlloc`
- `cudaMallocHost`

측정:
- H2D
- D2H
- total

---

## 5. Async Pipeline

baseline:
```text
H2D → TRT → D2H → NN
```

optimized:
```text
cudaMemcpyAsync
enqueueV3
CUDA stream
```

advanced:
```text
Frame N TRT
overlap
Frame N+1 preprocess
```

---

## 6. Avoid D2H Embedding

현재 output:
```text
[1,1536,32,32]
≈ 1.57M float
≈ 6 MB/frame FP32
```

목표:
```text
TensorRT GPU output
→ GPU NN
→ small result only D2H
```

---

## 7. CPU NN Baseline

Query:
```text
1024 x 1536
```

Bank:
```text
N x 1536
```

Distance:
- squared L2

Complexity:
```text
O(Q*N*D)
```

---

## 8. OpenMP

비교:
- 1 thread
- 2
- 4
- all cores

power도 함께 측정.

---

## 9. CUDA NN

Full FP16 bottle evaluation for `cuda_tiled_async` measured Image AUROC 1.0
and Pixel AUROC 0.985454531. Three independent runs of
`e6-bottle-fp16-cuda-tiled-async` averaged 416.590 ms NN time, 516.773 ms total
latency, 13.460 W, and 6.958 J/image. These fixed-image runs recorded uncommitted
source on top of `1a9ca6e`; retain their original manifests and power logs.
Relative to the tiled experiment, measured mean latency decreased about 19%
and energy/image about 30%. Comparisons across runs do not imply clock control.

For subsequent S4/S5 comparisons, config generation preserves an existing CUDA
backend (including `cuda_tiled_async`). CPU/OpenMP baselines switch to `cuda`
only at S5/S6. This keeps kernel selection fixed during system ablations.

Experimental `nn_backend: cuda_tiled_async` alternates two four-row bank
buffers. On SM80+ and dimensions divisible by four, aligned 16-byte
`cp.async` copies prefetch the next bank tile before the current tile's distance
calculation. Per-thread async waits and a block barrier complete each stage
before swapping buffers. Odd dimensions and older compiled architectures use
bounded synchronous copies; tail queries always participate in barriers.
At 1536 dimensions the two buffers require 48 KiB dynamic shared memory, with
device capacity checks and explicit opt-in. Existing backends remain unchanged.
Unit tests cover both odd dimensions and the async aligned path, bank/query
tails, cross-warp ties, and device NCHW. FP32 normal/anomalous outputs matched
the tiled baseline exactly. Performance and full FP16 AUROC remain separate
validation steps; more shared memory can reduce occupancy.

Experimental `nn_backend: cuda_tiled_cached` also caches eight query vectors in
shared memory, alongside four bank rows. At 1536 dimensions it requires 72 KiB
dynamic shared memory and explicitly opts into the device's larger shared-memory
limit. Both layouts and partial query tiles are supported; existing backends
are unchanged. FP32 normal/anomalous image dumps matched `cuda_tiled` exactly,
and GPU unit tests passed.

A preliminary comparison (3 warm-up, 10 measured frames, one run per backend)
measured NN means of 589.890 ms for tiled and 1331.05 ms for cached. This does
not establish controlled performance or energy rankings: it lacks independent
runs, sustained warm-up, and clock control. Increased shared-memory residency
may reduce occupancy; this mechanism has not been measured for the cached
kernel. Keep `cuda_tiled` as the measured speed baseline; the cached configs
are experimental and should not be promoted based on correctness alone.

`nn_backend: cuda_tiled` preserves the warp baseline as a separate backend and
stages four bank rows in shared memory for reuse by eight query warps per block.
One warp computes one query's distances, using direct squared differences and
FP32 accumulation. Both patch-major and device NCHW input are supported. Tail
queries participate in block barriers without reading or writing invalid queries;
tail bank rows are bounded and ties select the lowest bank index. At 1536
dimensions the bank tile uses 24 KiB shared memory. Unsupported shared-memory
requirements are rejected during initialization.

FP16 tiled bottle evaluation measured Image AUROC 1.0 and Pixel AUROC
0.985454531. Three runs of `e5-bottle-fp16-cuda-tiled` averaged 535.578 ms
NN time, 635.681 ms total latency, 15.728 W, and 9.992 J/image. Relative to
the warp experiment, latency decreased about 25% but energy/image increased
about 6.6%. These fixed-image measurements preserve the original logs and
dirty-build provenance; they do not establish a universal backend winner.

Use `configs/jetson_fp16_cuda_tiled.yaml` or
`configs/jetson_fp32_cuda_tiled.yaml`. Validate against `cuda_warp` before
collecting full AUROC and three-run timing/power measurements. Performance is
not assumed to improve: barriers and shared-memory occupancy can offset reuse.

`nn_backend: cuda` preserves the original thread-per-bank-row baseline.
`nn_backend: cuda_warp` uses one warp per bank row, divides dimensions across
32 lanes, reduces partial squared distances with warp shuffles, and caches the
query in shared memory. Both patch-major and device NCHW input paths are supported;
equal distances select the lowest bank index. The bank remains FP32.

Use `configs/jetson_fp32_cuda_warp.yaml` or
`configs/jetson_fp16_cuda_warp.yaml` for isolated comparisons at S0. Existing CUDA
configs keep the original kernel. Unit tests cover tail dimensions, small banks,
cross-warp ties, and NCHW inputs. Normal and anomalous bottle images passed the
existing numerical comparison bounds. Full bottle evaluation measured Image
AUROC 1.0 and Pixel AUROC 0.985454531 for FP16 with `cuda_warp`.

The three-run `e4-bottle-fp16-cuda-warp` experiment at MAXN_SUPER (ID 2), with
50 warm-up and 200 measured iterations per run, averaged 740.944 ms NN time,
846.976 ms total latency, 11.071 W, and 9.377 J/image. These measurements use
one fixed normal image, not the full test split. Raw CSV, power logs, and manifests
remain in the local `results/` directory; the binary recorded commit `34f0530`
and a dirty worktree because the warp changes had not yet been committed.

후보:
- tiling
- shared memory
- coalesced access
- parallel reduction
- FP16 storage
- FP32 accumulation

Nsight Compute로 분석.

---

## 10. FAISS

환경에서 안정 사용 가능 시:
- development complexity
- latency
- memory
- correctness
- deployment overhead

비교.

---

## 11. Multithreading

```text
Capture
→ bounded queue
→ Inference
→ result queue
→ Render/Logging
```

기본:
- mutex
- condition_variable

선택:
- lock-free SPSC queue

---

## 12. Correctness Tests

- preprocessing
- TensorRT shape
- memory bank load
- toy NN example
- CPU vs GPU NN
- anomaly score


## S5 device input transpose

`cuda_tiled_async_transpose` converts device NCHW features to patch-major layout
on the GPU before the existing asynchronous tiled nearest-neighbor search.
Use `configs/jetson_fp16_cuda_tiled_async_transpose_s5.yaml` (FP32 equivalent
also provided). The existing direct NCHW backend remains available for comparison.
The padded 32x32 transpose reuses the persistent query buffer; its cost is
included in NN latency. It requires 4224 bytes of shared memory per block and
no additional global buffer. Tests cover dimension/patch tails, repeated calls
and maximum-distance query extraction.

The 2026-10-09 smoke comparison (3 warm-up, 10 measured) reduced direct S5
NN mean from 1671.31 to 503.048 ms and total from 1750.48 to 582.857 ms.
Normal and broken_large outputs match direct S5 exactly. These smoke results
do not establish superiority over S0/S4; full accuracy evaluation and formal
three-run benchmarking are still required. Raw measurements and conditions:
`results/profiling/s5_transpose_smoke/README.md`.
