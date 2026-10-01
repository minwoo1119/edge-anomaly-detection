# Development Roadmap

`[x]`는 코드 또는 자동화 경로가 repository에 구현됐다는 뜻입니다. Jetson/Colab에서
artifact를 생성하거나 correctness·성능을 실측해야 하는 항목은 별도로 "실행 대기"로
표시하며, 실행하지 않은 결과를 완료로 간주하지 않습니다.

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

- [x] `Preprocessor` 구현 (Jetson reference 비교 실행 대기)
- [x] `TensorRTInferencer` 구현 (Jetson build/실행 대기)
- [x] persistent GPU buffers 구현
- [x] embedding output validation 도구
- [x] `MemoryBank` loader 구현
- [x] CPU brute-force NN 구현
- [x] anomaly map 구현
- [x] image-level score/reweighting 구현
- [x] Python ↔ C++ correctness comparison 자동화 (실제 artifact 검증 대기)

Acceptance:
- same output shape
- TensorRT/PyTorch embedding close
- C++ anomaly score close to Python
- no leak
- repeatable build

---

## Phase 2 — Benchmark Infrastructure

- [x] stage timer
- [x] CUDA event timing
- [x] warm-up
- [x] repeated measurement
- [x] CSV logger
- [x] environment metadata
- [x] tegrastats parser 및 measured-iteration 동기화
- [x] git hash logging

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

- [x] FP16 engine build 및 lineage 자동화 (Jetson engine 생성 대기)
- [x] FP32 vs FP16 embedding/NN/score/map 비교 도구 (실행 대기)
- [ ] runtime accuracy 실측
- [ ] latency 실측
- [ ] memory 실측
- [ ] power 실측

---

## Phase 4 — INT8

- [x] train-normal calibration dataset export
- [x] 100 normal image 기본 protocol
- [x] INT8 engine build 및 calibration lineage 구현 (Jetson 실행 대기)
- [x] embedding distortion 분석 도구 (실행 대기)
- [x] NN consistency 분석 도구 (실행 대기)
- [ ] runtime AUROC 실측
- [ ] latency / power 실측

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

- [x] ratio별 재학습·export·bundle 자동화 (Colab 실행 대기)
- [x] FP16 bank 변환 및 provenance 검증
- [ ] coreset/bank precision 정확도·성능 실측

---

## Phase 6 — NN Search Optimization

- [x] CPU baseline 구현
- [x] OpenMP backend 구현
- [x] custom CUDA backend 및 GPU-resident S5 구현 (Jetson correctness gate 대기)

최소 논문 버전:
- CPU
- GPU

---

## Phase 7 — System-Level Optimization

- [x] persistent buffers
- [x] pinned memory
- [x] async memcpy
- [x] CUDA stream
- [x] avoid D2H embedding copy
- [x] multithread preprocessing pipeline (Jetson overlap/성능 검증 대기)
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

- [x] category별 traced experiment matrix 자동화
- [ ] 대표 5개 full ablation 실행
- [ ] 15개 category 최종 후보 실행

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

- [x] raw CSV → processed CSV → table/SVG 생성 자동화
- [ ] Jetson/Colab 실험 결과 채우기 및 publication coverage gate 통과
