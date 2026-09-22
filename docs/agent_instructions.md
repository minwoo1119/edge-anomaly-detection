# Agent Development Instructions

## 1. Priority

1. Correctness
2. Reproducibility
3. Measurability
4. Maintainability
5. Performance

성능을 위해 correctness를 희생하지 않습니다.

---

## 2. Baseline

```text
PatchCore
WideResNet50-2
layer2 + layer3
256x256
coreset 0.10
Memory Bank [21401,1536]
TensorRT 10.3
CUDA 12.6
OpenCV 4.8
Jetson Orin Nano
```

Feature Extractor:
```text
input [1,3,256,256]
output [1,1536,32,32]
```

---

## 3. Architecture Boundary

기본:
```text
TensorRT
→ Feature Extractor

C++ / CUDA
→ Memory Bank Search
→ Score
→ Map
→ Threshold
```

임의 변경 금지.

---

## 4. File Responsibility

`main.cpp`에 모든 로직을 넣지 않습니다.

분리:
- Preprocessor
- TensorRTInferencer
- MemoryBank
- NearestNeighborSearch
- Benchmark

---

## 5. Optimization Rule

새 optimization에는 반드시:
- baseline
- correctness
- latency
- memory

비교가 있어야 합니다.

---

## 6. Config Rule

hard-code 금지:
- input size
- coreset
- precision
- warmup
- repeat count
- threshold
- category

---

## 7. Git

작업 단위 commit.

예:
```text
feat: TensorRT C++ FP32 추론 파이프라인 구현
feat: CPU 기반 Memory Bank 최근접 이웃 검색 추가
perf: CUDA 비동기 메모리 전송 적용
refactor: NN 검색 인터페이스 분리
fix: FP16 임베딩 reshape 오류 수정
docs: Jetson 벤치마크 절차 문서화
```

---

## 8. 파일 이동

반드시 기록:

```text
[파일 이동]
src/foo.cpp
→ jetson/src/foo.cpp
```

---

## 9. Workflow

Windows:
```text
edit → commit → push
```

Jetson:
```text
git pull → build → run → benchmark
```

---

## 10. TensorRT

Engine은 Jetson에서 생성.

Git 제외:
```text
*.engine
*.onnx
*.onnx.data
*.npy
*.ckpt
```

---

## 11. Correctness Reference

반드시 비교:
- Python preprocessing vs C++
- PyTorch vs TensorRT
- Python NN vs C++ CPU NN
- CPU NN vs GPU NN
- Python score vs C++ score

---

## 12. Benchmark Output

권장:
```text
benchmark → CSV → analysis → figure/table
```

수동 숫자 복붙 최소화.

---

## 13. 연구 질문 연결

새 기능 전에 답할 것:
1. 어떤 RQ를 검증하는가?
2. 무엇과 비교하는가?
3. 어떤 metric으로 판단하는가?

세 질문에 답할 수 없으면 우선순위가 낮습니다.

---

## 14. 하지 말 것

- test set tuning
- warm-up 생략
- correctness 없는 speed comparison
- per-frame engine load
- per-frame cudaMalloc
- artifact Git commit
- environment metadata 누락
- git hash 없는 benchmark
