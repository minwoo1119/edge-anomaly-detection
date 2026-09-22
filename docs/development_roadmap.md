# Development Roadmap

## Phase 0 — Completed

- [x] MVTec AD bottle exploration
- [x] PatchCore baseline
- [x] Memory Bank generation
- [x] Image / Pixel AUROC
- [x] heatmap / score analysis
- [x] checkpoint
- [x] Feature Extractor / Memory Bank partition
- [x] ONNX export
- [x] ONNX checker
- [x] PyTorch ↔ ONNX validation
- [x] Jetson environment scripts
- [x] FP32 engine build script
- [x] Jetson FP32 engine build
- [x] `trtexec` inference validation

Jetson:
```text
TensorRT 10.3.0
CUDA 12.6
OpenCV 4.8.0
```

---

## Phase 1 — C++ FP32 End-to-End Baseline

- [ ] `Preprocessor`
- [ ] `TensorRTInferencer`
- [ ] persistent GPU buffers
- [ ] embedding output validation
- [ ] `MemoryBank` loader
- [ ] CPU brute-force NN
- [ ] anomaly map
- [ ] image-level score
- [ ] Python ↔ C++ correctness comparison

Acceptance:
- same output shape
- TensorRT/PyTorch embedding close
- C++ anomaly score close to Python
- no leak
- repeatable build

---

## Phase 2 — Benchmark Infrastructure

- [ ] stage timer
- [ ] CUDA event timing
- [ ] warm-up
- [ ] repeated measurement
- [ ] CSV logger
- [ ] environment metadata
- [ ] tegrastats parser
- [ ] git hash logging

Stages:
```text
preprocess
H2D
TensorRT
D2H
NN
postprocess
total
```

---

## Phase 3 — FP16

- [ ] FP16 engine
- [ ] FP32 vs FP16 embedding error
- [ ] accuracy
- [ ] latency
- [ ] memory
- [ ] power

---

## Phase 4 — INT8

- [ ] calibration dataset
- [ ] 100 normal image baseline
- [ ] INT8 engine
- [ ] embedding distortion
- [ ] NN consistency
- [ ] AUROC
- [ ] latency / power

---

## Phase 5 — Memory Bank Optimization

Coreset:
```text
1%
2.5%
5%
10%
20%
```

Bank precision:
```text
FP32
FP16
```

---

## Phase 6 — NN Search Optimization

- [ ] CPU baseline
- [ ] OpenMP
- [ ] CUDA or FAISS GPU

최소 논문 버전:
- CPU
- GPU

---

## Phase 7 — System-Level Optimization

- [ ] persistent buffers
- [ ] pinned memory
- [ ] async memcpy
- [ ] CUDA stream
- [ ] avoid D2H embedding copy
- [ ] multithread pipeline
- [ ] optional double buffering
- [ ] Nsight profile

---

## Phase 8 — Multi-Category

우선 8개:
```text
bottle
capsule
hazelnut
screw
transistor
carpet
grid
leather
```

가능하면 15개 전체.

---

## Phase 9 — Additional Baseline

권장:
```text
EfficientAD
```

---

## Phase 10 — Paper-Ready Output

필수 Figure:
1. Architecture
2. Stage latency
3. FP32/FP16/INT8 trade-off
4. Coreset vs latency/memory
5. Coreset vs AUROC
6. NN backend
7. Accuracy-latency Pareto
8. Power / energy

필수 Table:
1. Environment
2. Precision comparison
3. Category-wise AUROC
4. Coreset ablation
5. NN backend
6. System optimization ablation
