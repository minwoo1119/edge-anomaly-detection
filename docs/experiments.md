# Experiment Design

## 1. 원칙

- 주요 변수는 한 번에 하나씩 변경
- 동일 power mode
- 동일 input size
- warm-up 후 측정
- 반복 측정
- mean / median / p95 보고
- git commit hash 기록
- test set threshold tuning 금지

---

## 2. Primary Variables

Precision:
- FP32
- FP16
- INT8

Coreset:
- 0.01
- 0.025
- 0.05
- 0.10
- 0.20

Memory Bank precision:
- FP32
- FP16

NN backend:
- CPU
- GPU

---

## 3. Accuracy Metrics

필수:
- Image AUROC
- Pixel AUROC

추가:
- F1
- Precision
- Recall
- Per-category recall

---

## 4. Embedding Distortion

FP32 reference 기준:
- MAE
- RMSE
- L2 relative error
- Cosine similarity

NN consistency:
- Top-1 agreement
- Top-k overlap
- distance rank correlation

Score consistency:
- Pearson
- Spearman
- absolute score error

---

## 5. Performance Metrics

- preprocessing latency
- H2D
- TensorRT
- D2H
- NN search
- postprocess
- total
- FPS
- host memory
- bank memory
- engine size
- average power
- peak power
- energy per image

---

## 6. Repetition Protocol

권장:
```text
warm-up: 50
measurement: 200
repeat runs: 3
```

보고:
- mean
- median
- std
- p95

---

## 7. Core Experiments

### E1 Precision
고정:
```text
coreset=0.10
bank=FP32
backend 동일
```
변경:
```text
FP32 / FP16 / INT8
```

### E2 Coreset
고정:
```text
precision 동일
bank 동일
backend 동일
```
변경:
```text
1 / 2.5 / 5 / 10 / 20 %
```

### E3 Bank Precision
변경:
```text
FP32 / FP16
```

### E4 NN Backend
변경:
```text
CPU / GPU
```

### E5 System Optimization
순차:
```text
sync baseline
+ persistent buffer
+ pinned memory
+ async memcpy
+ CUDA stream
+ GPU-resident search
+ pipelining
```

---

## 8. Category Protocol

Stage A:
대표 5개에서 full ablation
```text
bottle
capsule
screw
carpet
grid
```

Stage B:
선택 configuration을 전체 category에 평가.

---

## 9. Experiment ID

예:
```text
mvtec-bottle_patchcore_fp16_c005_bankfp16_cuda_run01
```
