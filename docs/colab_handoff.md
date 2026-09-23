# Colab Artifact Handoff

Dataset과 checkpoint가 Colab에만 있는 경우 학습을 반복하지 않고, 동일 checkpoint에서 평가·export·calibration/reference 생성을 수행한 뒤 검증 가능한 bundle로 전달합니다.

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

```bash
python src/package_colab_artifacts.py \
  --export-manifest models/patchcore_export_manifest.json \
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
