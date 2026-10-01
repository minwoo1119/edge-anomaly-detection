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

## Reproducible Python baseline

학습은 `train/good`만 사용하며 checkpoint reload 시 memory bank가 exact match하는 경우에만 완료됩니다.

```bash
python src/train.py \
  --dataset-root datasets/MVTecAD \
  --category bottle \
  --checkpoint models/patchcore_bottle.ckpt \
  --manifest results/metadata/train_bottle.json
```

평가는 고정 checkpoint에 대해 별도로 수행합니다. Test set에서는 AUROC만 primary metric으로 사용하며 test-derived F1 threshold를 deployment config로 내보내지 않습니다.

```bash
python src/evaluate.py \
  --checkpoint models/patchcore_bottle.ckpt \
  --dataset-root datasets/MVTecAD \
  --category bottle \
  --output-csv results/csv/accuracy.csv \
  --manifest results/metadata/evaluate_bottle.json
```

Training/evaluation manifest에는 checkpoint와 dataset fingerprint, 실제 package version을 기록합니다.

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

---

## 10. Deployment Runtime Accuracy

Precision, coreset, bank precision, NN backend를 비교할 때 PyTorch checkpoint의
AUROC를 재사용하지 않습니다. 각 C++/TensorRT configuration을 MVTec test split
전체에 실행해 raw score와 anomaly map으로 AUROC를 다시 계산합니다.

```bash
python3 jetson/scripts/evaluate_runtime.py \
  --executable jetson/build/edge_anomaly \
  --config configs/jetson_fp16.yaml \
  --dataset-root datasets/mvtec \
  --category bottle \
  --work-dir results/raw/runtime_accuracy/bottle-fp16 \
  --output-csv results/csv/runtime_accuracy.csv \
  --predictions-csv results/csv/bottle-fp16-predictions.csv \
  --manifest results/metadata/bottle-fp16-accuracy.json
```

이 평가는 test set으로 threshold를 선택하지 않으며, `config_sha256`,
`engine_sha256`, `memory_bank_sha256`, dataset fingerprint를 함께 기록합니다.
생성된 `runtime_accuracy.csv`가 Table 2–5와 Figure 4/5/8의 정확도 입력입니다.

Jetson benchmark summary와 runtime accuracy는 config/artifact hash를 기준으로
결합합니다. 일치하지 않는 결과나 PyTorch checkpoint accuracy는 거부됩니다.

```bash
python3 src/merge_experiment_results.py \
  --benchmark-summary results/processed/precision_run_summary.csv \
  --runtime-accuracy results/csv/runtime_accuracy.csv \
  --output results/processed/precision_results.csv
```
