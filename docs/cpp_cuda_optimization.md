# C++ / CUDA Optimization

## System ablation stage contract

System optimization은 `optimization_stage`와 실제 실행 경로가 일치해야 합니다.

```text
S0: frame마다 device input/output buffer 할당 + pageable host memory + synchronous copy
S1: persistent device input/output buffer
S2: S1 + persistent pinned host staging buffer
S3: S2 + asynchronous H2D/D2H on the default stream
S4: S3 + dedicated non-blocking CUDA stream
S5: GPU-resident NN (아직 선택 불가)
S6: pipeline overlap (아직 선택 불가)
```

S5/S6가 실제로 구현되기 전에는 runtime이 해당 stage를 거부합니다. 따라서
metadata만 S5/S6로 바꿔 성능 결과를 생성할 수 없습니다.

환경 정보와 artifact path를 채운 baseline config에서 S0-S4 config를 생성합니다.

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
