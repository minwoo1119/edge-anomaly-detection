# Jetson Orin Nano Deployment

## Confirmed Environment

```text
TensorRT : 10.3.0.30
CUDA nvcc: 12.6
OpenCV   : 4.8.0
trtexec  : /usr/src/tensorrt/bin/trtexec
```

---

## Repository

최초:
```bash
git clone <REPOSITORY_URL>
cd edge-anomaly-detection
```

이후:
```bash
git pull
```

---

## Model Artifacts

Git 제외:

```text
models/
├── patchcore_feature_extractor.onnx
├── patchcore_feature_extractor.onnx.data
└── patchcore_memory_bank.npy
```

Jetson 생성:
```text
patchcore_feature_extractor_fp32.engine
patchcore_feature_extractor_fp16.engine
patchcore_feature_extractor_int8.engine
```

---

## Reproducible ONNX and Memory Bank Export

Notebook의 수동 export 대신 아래 CLI를 사용해 ONNX와 memory bank를 동일 checkpoint에서 함께 생성합니다.

```bash
python src/export_onnx.py \
  --checkpoint models/patchcore_bottle.ckpt \
  --onnx models/patchcore_feature_extractor.onnx \
  --memory-bank models/patchcore_memory_bank.npy \
  --manifest models/patchcore_export_manifest.json \
  --validation-image datasets/mvtec/bottle/train/good/000.png
```

완료 조건은 다음과 같습니다.

- export wrapper와 Anomalib `generate_embedding`의 exact equality
- ONNX checker 통과
- PyTorch/ONNX Runtime 출력 tolerance 통과
- embedding shape `[1, 1536, 32, 32]`
- memory bank shape `[N, 1536]`, FP32, finite
- checkpoint/ONNX/external data/memory bank SHA-256 manifest 기록

`--opset`은 TensorRT parser 호환성을 실제 Jetson 환경에서 확인한 경우에만 지정합니다. 스크립트는 기존 artifact를 자동 덮어쓰지 않습니다.

FP16 memory-bank ablation artifact는 source export manifest를 연결해 생성합니다.

```bash
python src/convert_memory_bank.py \
  --input models/patchcore_memory_bank.npy \
  --output models/patchcore_memory_bank_fp16.npy \
  --dtype fp16 \
  --source-manifest models/patchcore_export_manifest.json
```

생성된 `.npy.json`에는 source/output hash와 FP32 복원 오차가 기록됩니다. Benchmark runner는 이 provenance가 없거나 hash가 다르면 FP16 bank 실험을 거부합니다.

---

## Environment Check

```bash
chmod +x jetson/scripts/check_env.sh
./jetson/scripts/check_env.sh
```

---

## FP32 Engine

```bash
chmod +x jetson/scripts/build_fp32_engine.sh
./jetson/scripts/build_fp32_engine.sh
```

Validation:

```bash
/usr/src/tensorrt/bin/trtexec   --loadEngine=models/patchcore_feature_extractor_fp32.engine   --shapes=input:1x3x256x256
```

---

## FP16

```bash
/usr/src/tensorrt/bin/trtexec   --onnx=models/patchcore_feature_extractor.onnx   --saveEngine=models/patchcore_feature_extractor_fp16.engine   --fp16
```

FP32/FP16 build script는 기존 engine을 덮어쓰지 않으며, ONNX·external data·engine SHA-256와 `trtexec` version을 `<engine>.json`에 기록합니다.

---

## INT8 Calibration Inputs

INT8 calibration에서는 OpenCV로 원본 이미지를 다시 전처리하지 않습니다. 학습 환경에서 Anomalib transform을 사용해 `train/good` tensor를 먼저 고정합니다.

```bash
python src/export_int8_calibration.py \
  --input-dir datasets/mvtec/bottle/train/good \
  --output-dir models/calibration/bottle \
  --images 100
```

그 후 Jetson에서 tensor와 manifest의 SHA-256를 검증하고 engine을 생성합니다.

```bash
bash jetson/scripts/build_int8_engine.sh models/calibration/bottle
```

Calibration cache는 기본적으로 재사용하지 않습니다. 재사용 시에는 동일 ONNX와 동일 calibration tensor에서 생성된 cache인지 별도로 확인한 뒤 `--reuse-cache`를 명시해야 합니다. Test image는 calibration에 사용할 수 없습니다.

실제 옵션은 `trtexec --help`로 확인.

---

## INT8

representative normal calibration image 사용.

현재 baseline:
```text
100 normal images
```

Random calibration 금지.

---

## C++ Build

```bash
mkdir -p jetson/build
cd jetson/build

cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

Benchmark 실행 시 binary가 Release가 아니면 runtime이 측정을 거부합니다. Git commit과 dirty 상태는 configure 시 binary에 기록되므로 소스 변경 뒤에는 CMake configure/build를 다시 수행합니다.

---

## Workflow

Windows:
```text
edit
commit
push
```

Jetson:
```text
git pull
build
run
benchmark
```

Jetson은 실행/검증 장비로 사용합니다.

---

## Python ↔ C++ Preprocessing Correctness

성능 측정 전에 동일 이미지의 Python/Anomalib 입력 tensor와 C++ 입력 tensor를 비교합니다.

Python reference 생성:

```bash
python3 src/export_preprocessing_reference.py \
    --image datasets/MVTecAD/bottle/test/good/000.png \
    --output outputs/correctness/python_input.npy
```

C++ tensor 생성:

```bash
./jetson/build/edge_anomaly \
    --config configs/jetson_fp32.yaml \
    --image datasets/MVTecAD/bottle/test/good/000.png \
    --dump-input outputs/correctness/cpp_input.npy \
    --preprocess-only
```

수치 비교:

```bash
python3 src/compare_runtime.py \
    --reference outputs/correctness/python_input.npy \
    --candidate outputs/correctness/cpp_input.npy \
    --max-mae 1e-4 \
    --max-error 1e-3 \
    --min-cosine 0.9999 \
    --report outputs/correctness/preprocessing_comparison.json
```

Reference manifest에는 source image hash, output hash, Anomalib/PyTorch/Torchvision version과 전처리 크기를 기록합니다. 실제 이미지 비교 결과가 확보되기 전에는 resize interpolation을 변경하지 않습니다.

---

## Python ↔ C++ FP32 End-to-End Correctness

Python에서 raw PatchCore reference를 생성합니다. 이 출력은 normalization 전 raw score와 anomaly map을 포함합니다.

```bash
python3 src/export_patchcore_reference.py \
    --checkpoint models/patchcore_bottle.ckpt \
    --image datasets/MVTecAD/bottle/test/good/000.png \
    --output-dir outputs/correctness/python \
    --prefix bottle_good_000
```

Jetson C++에서 동일 artifact를 생성합니다.

```bash
./jetson/build/edge_anomaly \
    --config configs/jetson_fp32.yaml \
    --image datasets/MVTecAD/bottle/test/good/000.png \
    --dump-input outputs/correctness/cpp/bottle_good_000_input.npy \
    --dump-embedding outputs/correctness/cpp/bottle_good_000_embedding.npy \
    --dump-patches outputs/correctness/cpp/bottle_good_000_patches.npy \
    --dump-patch-scores outputs/correctness/cpp/bottle_good_000_patch_scores.npy \
    --dump-nn-indices outputs/correctness/cpp/bottle_good_000_nn_indices.npy \
    --dump-score outputs/correctness/cpp/bottle_good_000_raw_score.npy \
    --dump-map outputs/correctness/cpp/bottle_good_000_anomaly_map.npy
```

모든 단계를 비교하고 단계별 JSON report를 생성합니다.

```bash
./jetson/scripts/validate_runtime_outputs.sh \
    outputs/correctness/python \
    outputs/correctness/cpp \
    bottle_good_000
```

검증 순서는 `input → embedding → patches → patch scores/indices → raw score → anomaly map`입니다. 앞 단계가 실패하면 이후 성능 비교를 진행하지 않습니다.
