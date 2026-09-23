# Benchmarking Protocol

## 1. Stage Timing

```text
preprocess
H2D
TensorRT
D2H
reshape
NN
postprocess
total
```

CPU:
```cpp
std::chrono::steady_clock
```

GPU:
```text
cudaEventRecord
cudaEventElapsedTime
```

---

## 2. Warm-up

권장:
```text
50 warm-up
200 measured
```

---

## 3. Statistics

- mean
- median
- std
- min
- max
- p95

---

## 4. CSV Schema

```text
timestamp
git_commit
device
jetpack
cuda
tensorrt
opencv
power_mode
category
model
precision
coreset_ratio
bank_precision
nn_backend
run_id

image_auroc
pixel_auroc

preprocess_ms
h2d_ms
trt_ms
d2h_ms
nn_ms
post_ms
total_ms
fps

host_memory_mb
bank_memory_mb
engine_size_mb

avg_power_w
peak_power_w
energy_per_image_mj
```

---

## 5. Power

Jetson `tegrastats` 기반.

기록:
- power mode
- temperature
- CPU/GPU utilization
- RAM
- average / peak power

가능하면:
- FPS/W
- mJ/image

---

## 6. Fairness

Precision 비교:
- 동일 images
- 동일 power mode
- 동일 coreset
- 동일 backend

NN backend 비교:
- 동일 query
- 동일 bank
- 동일 precision
- correctness 먼저 확인

---

## 7. Throughput vs Latency

batch=1 latency를 primary metric으로 사용.

Multithread pipeline에서는:
- per-frame latency
- sustained FPS

둘 다 보고합니다.

---

## 8. Profiling

Nsight Systems:
- CPU/GPU overlap
- memcpy
- TensorRT
- idle gap

Nsight Compute:
- custom CUDA NN
- bandwidth
- occupancy
- warp efficiency

---

## 9. Regression

최적화 후:
- known embedding
- known NN
- known anomaly score

---

## 10. Reproducible benchmark runner

성능 측정은 직접 실행 파일을 호출하지 않고 다음 runner를 사용합니다.

```bash
bash jetson/scripts/run_benchmark.sh \
  configs/jetson_fp32.yaml \
  datasets/mvtec/bottle/test/good/000.png
```

Runner는 매 실행마다 고유한 `run_id`를 만들고 다음 원시 자료를 보존합니다.

- 반복별 stage timing CSV
- 실행별 power summary CSV
- 원본 `tegrastats` 로그
- config/image/engine/memory bank/executable SHA-256가 포함된 JSON manifest
- full git commit, dirty 상태, Release build 유형

동일 CSV에 다른 schema를 append하는 동작은 오류로 처리합니다. benchmark mode는 Release build와 네 개 입력 artifact의 SHA-256가 없으면 실행되지 않습니다.

CPU backend 비교는 `configs/jetson_fp32.yaml`의 단일 스레드 baseline과 `configs/jetson_fp32_openmp.yaml`의 OpenMP variant를 구분해서 수행합니다.

원시 timing/power CSV를 실행별 통계 CSV로 변환합니다.

```bash
python3 src/process_benchmarks.py \
  --benchmark-csv results/csv/benchmark.csv \
  --power-csv results/csv/power.csv \
  --output results/processed/run_summary.csv
```

이 파일은 Figure 3(Stage-Level Latency), Figure 6(CPU/GPU NN), Figure 9(Power/Energy)와 Table 2/4/6의 입력으로 사용합니다. AUROC는 별도의 evaluation 결과와 artifact hash를 기준으로 결합해야 하며 benchmark runner가 임의로 채우지 않습니다.

논문 결과는 기본 3회의 독립 실행으로 수집합니다. 아래 명령은 각 실행에 분리된 `run_id`, power log, manifest를 만들고 마지막에 processed CSV를 생성합니다.

```bash
bash jetson/scripts/run_experiment.sh \
  configs/jetson_fp32.yaml \
  datasets/mvtec/bottle/test/good/000.png \
  e1-bottle-fp32-cpu \
  3
```

를 reference와 비교합니다.
