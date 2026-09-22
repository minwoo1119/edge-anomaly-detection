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

를 reference와 비교합니다.
