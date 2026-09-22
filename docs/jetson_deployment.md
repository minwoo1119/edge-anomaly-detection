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
