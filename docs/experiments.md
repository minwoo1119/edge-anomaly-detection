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

Python reference와 C++ runtime의 correctness report는
`jetson/scripts/validate_runtime_outputs.sh`로 생성합니다. 각 report에는 shape,
MAE/RMSE/max error/relative L2/cosine similarity와 양쪽 tensor의 min/max/mean/std,
선택 위치 값이 기록되며 NN index report에는 element agreement rate가 포함됩니다.

FP16/INT8 precision consistency는 여러 이미지 dump를 묶어 분석합니다.

```bash
python3 src/analyze_precision_consistency.py \
  --reference-dir outputs/correctness/fp32 \
  --candidate-dir outputs/correctness/int8 \
  --candidate-label int8 \
  --output-csv results/csv/precision_consistency.csv \
  --manifest results/metadata/int8_consistency.json
```

embedding MAE/RMSE/relative L2/cosine, Top-1 NN agreement, distance와 score의
Pearson/Spearman correlation, anomaly-map error를 기록합니다. Top-k dump가 실제로
존재하는 경우에만 overlap을 계산하며 없는 값을 임의로 채우지 않습니다.

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

모든 ablation 결과가 준비되면 다음 명령이 공정성 조건, 200회 measurement,
configuration별 3회 independent run, runtime accuracy 출처를 검증하고 논문용
CSV/Markdown table과 data-driven SVG figure를 재생성합니다.

```bash
python3 scripts/generate_paper_results.py \
  --baseline results/processed/baseline_results.csv \
  --precision results/processed/precision_results.csv \
  --coreset results/processed/coreset_results.csv \
  --nn-backend results/processed/nn_backend_results.csv \
  --system results/processed/system_results.csv \
  --multi-category results/processed/multi_category_results.csv \
  --output-root results
```

개발 중 일부 조합만 점검할 때만 `--allow-incomplete`를 사용합니다. 최종 논문
산출물에서는 이 옵션을 사용하지 않습니다. 입력 CSV hash와 누락 조합은
`results/result_coverage.json`에 기록됩니다.

Jetson의 전체 accuracy/benchmark 실행은 JSON matrix로 자동화할 수 있습니다.

```bash
python3 jetson/scripts/run_experiment_matrix.py \
  --plan configs/experiment_matrix.json \
  --dataset-root datasets/mvtec \
  --manifest results/metadata/experiment_matrix.json
```

실행 전에는 `--dry-run`으로 command와 group 구성을 확인할 수 있습니다. 실제
실행은 clean git worktree와 configuration별 최소 3회 independent run을 강제하며,
각 group의 `<group>_results.csv`까지 자동으로 생성합니다.


## Bottle S5 GPU transpose results (2026-10-09)

FP16 extractor, FP32 bank, MAXN_SUPER_ID2. Three runs each use 50 warm-up
and 200 measured images. Mean total latency: 490.549 ms; NN: 403.916 ms;
power: 13.512 W; energy: 6628.344 mJ/image. Full bottle test accuracy
(83 images): image AUROC 1.0, pixel AUROC 0.9854545306151844.

Normal and broken_large outputs match direct S5 exactly. The GPU transpose
reuses the existing query buffer without additional global allocation and
uses 4224 bytes shared memory per block. Its latency is included in NN.
Zero embedding D2H/reshape stages do not exclude NN result transfers.

Joined data: `results/processed/e9-bottle-fp16-async-transpose-s5_results.csv`.
Configuration, engine and bank hashes are checked when joining accuracy.
Coreset ratios are compared numerically so 0.10 and 0.1 are equivalent.

Original measurements retain f01eed4 and git_dirty=true provenance. They
are development measurements; committing later does not change their
metadata. Reconfigure and rebuild after committing and use a new run ID
for final clean-build measurements. Historical E6/S0 mean total latency
was 516.773 ms; the 5.1% reduction is a comparison across separate sessions.
