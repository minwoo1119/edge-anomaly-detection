# Colab Artifact Handoff

Dataset과 checkpoint가 Colab에만 있는 경우 학습을 반복하지 않고, 동일 checkpoint에서 평가·export·calibration/reference 생성을 수행한 뒤 검증 가능한 bundle로 전달합니다.

## 가장 간단한 실행 방법

아래 명령 하나가 평가, ONNX/Memory Bank export, INT8 calibration tensor, 정상/이상 correctness reference와 ZIP bundle 생성을 순서대로 수행합니다.

```bash
python src/run_colab_pipeline.py \
  --checkpoint models/patchcore_bottle.ckpt \
  --training-manifest results/metadata/train_bottle.json \
  --dataset-root datasets/MVTecAD \
  --category bottle \
  --output-root outputs/colab_bottle_run01 \
  --device cuda \
  --export-device cpu
```

이 명령은 test set을 사용하지 않고 `train/good`의 raw PatchCore score 99.5% 분위수로
배포 threshold manifest도 생성합니다. AUROC 평가와 threshold calibration은 서로 분리됩니다.
평가·threshold·reference는 CUDA에서 실행하고, ONNX 수치 검증은 ONNX Runtime과 동일한
CPU 기준으로 수행해 장치별 커널 오차가 export gate에 섞이지 않게 합니다.
PyTorch 2.11의 `dynamo` exporter가 정적 PatchCore feature graph에서 실패하는 환경을
피하기 위해 pipeline은 `torchscript` exporter를 명시하며, 생성 결과에는 동일하게
ONNX checker와 ONNX Runtime 수치 검증을 적용합니다.
FP32 convolution 구현 차이는 최대 절대 오차 `3e-4`로 제한하고, 일부 지점의 큰 오차가
전체 분포 오차를 가리지 않도록 평균 절대 오차도 `1e-5` 이하로 별도 검증합니다.

출력 디렉터리는 매 실행마다 비어 있는 새 경로를 사용합니다. 아래 개별 단계는 실패 지점을 따로 재현하거나 설정을 변경해야 할 때 사용합니다.

## 1. Colab repository 동기화

```bash
git pull
pip install -r requirements.txt
```

PyTorch와 torchvision은 Colab CUDA 환경에 맞는 기존 설치를 사용합니다.

## 2. 고정 checkpoint 평가

```bash
python src/evaluate.py \
  --checkpoint models/patchcore_bottle.ckpt \
  --dataset-root datasets/MVTecAD \
  --category bottle \
  --output-csv results/csv/accuracy.csv \
  --manifest results/metadata/evaluate_bottle.json
```

Image/Pixel AUROC는 논문 정확도 결과에 사용합니다. Test-derived F1 threshold는 deployment threshold로 내보내지 않습니다.

## 3. ONNX와 Memory Bank 재생성

```bash
python src/export_onnx.py \
  --checkpoint models/patchcore_bottle.ckpt \
  --training-manifest results/metadata/train_bottle.json \
  --onnx models/patchcore_feature_extractor.onnx \
  --memory-bank models/patchcore_memory_bank.npy \
  --manifest models/patchcore_export_manifest.json \
  --validation-image datasets/MVTecAD/bottle/train/good/000.png
```

기존 ONNX/Memory Bank가 있으면 별도 디렉터리에 새로 생성한 후 검증이 끝났을 때 교체합니다. 스크립트는 기존 결과를 자동 덮어쓰지 않습니다.

## 4. INT8 calibration tensor 생성

```bash
python src/export_int8_calibration.py \
  --input-dir datasets/MVTecAD/bottle/train/good \
  --output-dir models/calibration/bottle \
  --images 100
```

Calibration에는 test image를 사용하지 않습니다.

## 5. Python correctness reference 생성

정상과 이상 이미지를 각각 최소 한 장 이상 선택합니다.

```bash
python src/export_patchcore_reference.py \
  --checkpoint models/patchcore_bottle.ckpt \
  --image datasets/MVTecAD/bottle/test/good/000.png \
  --output-dir results/reference \
  --prefix bottle_good_000

python src/export_patchcore_reference.py \
  --checkpoint models/patchcore_bottle.ckpt \
  --image datasets/MVTecAD/bottle/test/broken_large/000.png \
  --output-dir results/reference \
  --prefix bottle_broken_large_000
```

이 test image 사용은 threshold/hyperparameter tuning이 아니라 고정 구현의 correctness 비교에만 사용합니다.

## 6. Bundle 생성

먼저 배포 threshold를 train-normal에서 고정합니다.

```bash
python src/calibrate_threshold.py \
  --checkpoint models/patchcore_bottle.ckpt \
  --training-manifest results/metadata/train_bottle.json \
  --dataset-root datasets/MVTecAD \
  --category bottle \
  --output results/metadata/threshold_bottle.json
```

```bash
python src/package_colab_artifacts.py \
  --export-manifest models/patchcore_export_manifest.json \
  --training-manifest results/metadata/train_bottle.json \
  --threshold-manifest results/metadata/threshold_bottle.json \
  --calibration-dir models/calibration/bottle \
  --reference-dir results/reference \
  --evaluation-manifest results/metadata/evaluate_bottle.json \
  --accuracy-csv results/csv/accuracy.csv \
  --output outputs/bottle_artifacts.zip
```

생성 파일:

```text
outputs/bottle_artifacts.zip
outputs/bottle_artifacts.zip.sha256
```

## 7. 로컬 검증

Bundle을 이 repository로 전달한 뒤 압축 해제 전에 검증합니다.

```bash
python src/verify_artifact_bundle.py \
  --bundle outputs/bottle_artifacts.zip
```

검증된 bundle에 포함된 ONNX/Memory Bank/calibration/reference는 Jetson 전달 대상이며, checkpoint와 accuracy 결과는 로컬 분석 및 재현성 보관 대상입니다.

## Coreset ablation artifacts

전용 노트북 `notebooks/run_coreset_ablation.ipynb`을 Colab GPU에서 위에서 아래로
실행하면 아래 명령과 bundle 검증·Drive 복사를 한 흐름으로 수행합니다.

1%, 2.5%, 5%, 10%, 20% coreset은 동일 memory bank 파일의 metadata만 바꾸지
않고 각각 train-normal에서 다시 생성합니다.

```bash
python src/run_coreset_ablation.py \
  --dataset-root /content/MVTecAD \
  --category bottle \
  --output-root /content/artifacts/coreset_bottle \
  --accelerator gpu
```

각 ratio는 독립 checkpoint, training manifest, ONNX, memory bank, export manifest,
calibration/reference 및 bundle을 갖습니다. 이후 Jetson config의 `coreset_ratio`와
`memory_bank_path`는 반드시 같은 variant를 가리켜야 합니다.

생성된 `c001`, `c0025`, `c005`, `c01`, `c02` 디렉터리를 구조를 유지한 채
Jetson의 `outputs/transfers/coreset_bottle`로 복사합니다. Jetson에서는 다음 한
명령으로 다섯 bundle을 검증·압축 해제하고, 각 ONNX에서 FP16 engine을 빌드하고,
최종 CUDA backend 설정과 experiment matrix를 생성합니다.

```bash
python3 jetson/scripts/prepare_coreset_ablation.py \
  --bundle-root outputs/transfers/coreset_bottle \
  --category bottle \
  --artifact-root artifacts/coreset \
  --config-dir results/configs/paper_coreset_bottle \
  --plan-output results/plans/paper_coreset_bottle.json \
  --benchmark-image datasets/MVTecAD/bottle/test/good/000.png
```

준비가 끝나면 생성된 계획을 먼저 검토합니다.

```bash
python3 jetson/scripts/run_experiment_matrix.py \
  --plan results/plans/paper_coreset_bottle.json \
  --dataset-root datasets/MVTecAD \
  --manifest results/metadata/paper_coreset_bottle_dry_run.json \
  --dry-run
```

실제 실행 시에는 `--dry-run`을 빼고 새로운 manifest 이름을 사용합니다. 각
configuration은 전체 test 정확도 평가 1회와 50회 warm-up/200회 측정 3회를
수행하며, 마지막에 `results/processed/coreset_results.csv`를 생성합니다.
긴 실행이 완전한 configuration 사이에서 중단됐다면 새 manifest 경로와
`--resume`을 사용합니다. 일부 출력만 존재하는 configuration은 자동으로
건너뛰지 않으며, 오염 방지를 위해 먼저 점검하도록 중단됩니다.

## 15-category publication artifacts

`notebooks/run_multicategory_publication.ipynb`은 MVTec AD 15개 category를 seed
42, 10% coreset으로 각각 학습하고, category별 검증된 bundle을 Drive의
`edge-anomaly-output/mvtec15_seed42_c01/<category>/`에 저장합니다. Colab 세션
제한을 고려해 `CATEGORIES_TO_RUN`을 3–5개씩 나눌 수 있으며,
`COMPLETE.json`이 존재하는 category는 재실행 시 건너뜁니다.
중단된 category 폴더가 Drive에 남아 있어도 폴더 전체를 지울 필요가 없습니다.
노트북은 새 로컬 임시 폴더에서 해당 category를 다시 생성하고, 검증된 ZIP과
체크섬을 교체한 뒤 `COMPLETE.json`을 마지막에 기록합니다.
카테고리의 train-normal 이미지가 `CALIBRATION_IMAGES`보다 적으면 가용한
이미지를 모두 사용합니다. 예를 들어 toothbrush는 100장 요청 시 60장을
사용하며, 실제 장수는 calibration manifest에 기록됩니다.

각 category bundle은 독립 checkpoint, training/export/evaluation manifest,
ONNX, memory bank, train-normal INT8 calibration 입력과 correctness reference를
포함합니다. 15개 `COMPLETE.json`이 모두 생기기 전에는 multi-category 결과가
완료된 것으로 간주하지 않습니다.

노트북은 자동 다운로드한 category 데이터도
`edge-anomaly-input/datasets/MVTecAD/<category>/`에 보존합니다. 완료 후 다음 두
Drive 디렉터리를 구조 그대로 Jetson으로 복사합니다.

```text
edge-anomaly-output/mvtec15_seed42_c01/
edge-anomaly-input/datasets/MVTecAD/
```

Jetson 저장 위치는 각각 아래와 같습니다.

```text
outputs/transfers/mvtec15_seed42_c01/
datasets/MVTecAD/
```

15개 bundle과 데이터셋을 받은 뒤 한 명령으로 bundle을 검증하고, category별
FP16 TensorRT engine/config와 experiment matrix를 생성합니다.

```bash
python3 jetson/scripts/prepare_multicategory_publication.py \
  --bundle-root outputs/transfers/mvtec15_seed42_c01 \
  --dataset-root datasets/MVTecAD \
  --artifact-root artifacts/mvtec15 \
  --config-dir results/configs/paper_mvtec15 \
  --plan-output results/plans/paper_mvtec15.json
```

그 다음 `run_experiment_matrix.py`를 `--dry-run`으로 검토한 후 실제 실행합니다.
완료 결과는 `results/processed/multi_category_results.csv`에 병합됩니다.
