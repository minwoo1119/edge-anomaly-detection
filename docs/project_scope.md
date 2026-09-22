# Project Scope

## 1. 프로젝트 정의

본 프로젝트의 최종 목표는 단순한 PatchCore Jetson 배포가 아니라,

> **Memory-bank 기반 산업용 anomaly detection을 resource-constrained edge device에서 효율적으로 실행하기 위해 모델·알고리즘·시스템 수준의 최적화를 통합하고, 정확도–지연시간–메모리–전력 간 trade-off를 정량적으로 분석하는 것**

입니다.

대상 하드웨어는 NVIDIA Jetson Orin Nano입니다.

---

## 2. Research Questions

### RQ1. Feature precision이 정확도와 latency에 미치는 영향은 무엇인가?

비교:
- FP32
- FP16
- INT8

측정:
- Image AUROC
- Pixel AUROC
- Feature extraction latency
- End-to-end latency
- FPS
- memory
- power
- embedding distortion

### RQ2. Memory Bank를 얼마나 줄여도 정확도를 유지할 수 있는가?

Coreset ratio:
- 1%
- 2.5%
- 5%
- 10%
- 20%

측정:
- Memory Bank entry 수
- memory footprint
- NN latency
- Image AUROC
- Pixel AUROC

### RQ3. Memory Bank precision을 낮추면 어떤 trade-off가 발생하는가?

비교:
- FP32 bank
- FP16 bank
- 선택적으로 INT8 / compressed representation

### RQ4. NN search가 실제 edge bottleneck인가?

비교 후보:
- CPU brute-force
- OpenMP CPU
- CUDA brute-force
- FAISS GPU

최소 논문 버전에서는 CPU baseline과 GPU backend를 비교합니다.

### RQ5. 모델 최적화와 시스템 최적화 중 어느 쪽이 end-to-end 성능에 더 크게 기여하는가?

비교:
- FP32 → FP16 → INT8
- Coreset 감소
- NN backend 변경
- pinned memory
- async memcpy
- CUDA stream
- GPU-resident search

### RQ6. 결과가 다양한 anomaly category에서도 유지되는가?

MVTec AD multi-category 실험으로 검증합니다.

---

## 3. 논문 Contribution 후보

실제 실험 결과로 확인된 항목만 최종 contribution으로 주장합니다.

1. Deployment-aware PatchCore partitioning
2. Precision / coreset / memory-bank / NN backend의 joint trade-off 분석
3. C++/CUDA system-level optimization의 단계별 ablation
4. Jetson Orin Nano에서의 reproducible end-to-end benchmark

---

## 4. Dataset Scope

현재 baseline:
- bottle

논문 권장:
- 가능하면 MVTec AD 15개 전체

시간 제약 시 최소 8개:
- bottle
- capsule
- hazelnut
- screw
- transistor
- carpet
- grid
- leather

Object와 texture를 모두 포함합니다.

---

## 5. Model Scope

Main:
- PatchCore

권장 baseline:
- EfficientAD

선택:
- PaDiM
- FastFlow

논문의 중심은 PatchCore 최적화이며, 다른 모델은 deployment baseline 역할로 사용합니다.

---

## 6. Engineering Success Criteria

- Jetson C++ runtime에서 실제 이미지 입력
- TensorRT feature extraction
- Memory Bank NN search
- anomaly score / anomaly map 생성
- Python baseline과 C++ runtime 정합성 확인
- FP32 / FP16 / INT8 benchmark
- CSV 자동 기록
- 재현 가능한 build/run procedure

---

## 7. Paper Readiness Criteria

- multi-category 결과
- FP32 / FP16 / INT8
- coreset ablation
- NN backend 비교
- stage-level latency breakdown
- memory / power 측정
- system optimization ablation
- 반복 측정 및 통계 요약
- 자동 생성되는 CSV / figure / table
