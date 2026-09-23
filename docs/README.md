# Edge-Optimized Industrial Anomaly Detection Docs

이 문서 세트는 PatchCore 기반 산업용 이상 탐지를 NVIDIA Jetson Orin Nano에 배포하고,
모델·알고리즘·시스템 레벨 최적화를 통해 논문 수준의 실험을 수행하기 위한 개발 명세입니다.

## 문서 구성

- `project_scope.md` — 프로젝트 목표, 연구 질문, 범위, 성공 기준
- `architecture.md` — 전체 시스템 구조와 Python / ONNX / TensorRT / C++ / CUDA 책임 분리
- `repository_structure.md` — 저장소 구조와 파일 책임
- `development_roadmap.md` — 현재 상태부터 논문 완성까지 단계별 개발 로드맵
- `experiments.md` — 논문용 실험 설계와 ablation
- `benchmarking.md` — latency / FPS / memory / power / AUROC 측정 규칙
- `colab_handoff.md` — Colab checkpoint/dataset 기반 artifact 생성 및 전달 절차
- `cpp_cuda_optimization.md` — C++17 / CUDA / memory / async / NN 최적화 계획
- `jetson_deployment.md` — Jetson Orin Nano 배포 및 빌드 절차
- `paper_plan.md` — 논문 가설, 기여점, figure/table 및 섹션 구성
- `agent_instructions.md` — 개발 에이전트가 따라야 할 규칙

## 현재 상태

완료:
- MVTec AD `bottle` 데이터 탐색
- PatchCore + WideResNet50-2 baseline
- Memory Bank 생성
- Image / Pixel AUROC 평가
- anomaly heatmap 및 score 분석
- checkpoint 저장
- full-model ONNX export 한계 확인
- Feature Extractor / Memory Bank 분리
- Feature Extractor ONNX export
- PyTorch ↔ ONNX numerical equivalence 확인
- Jetson Orin Nano에서 TensorRT FP32 engine 생성
- `trtexec` 기반 engine 실행 검증

현재 주요 artifact:

```text
patchcore_feature_extractor.onnx
patchcore_feature_extractor.onnx.data
patchcore_memory_bank.npy
patchcore_feature_extractor_fp32.engine
```

현재 baseline:

```text
Backbone     : WideResNet50-2
Layers       : layer2 + layer3
Input        : 256 x 256
Coreset      : 0.10
Memory Bank  : [21401, 1536]
TensorRT     : 10.3.0
CUDA nvcc    : 12.6
OpenCV       : 4.8.0
Jetson       : Orin Nano
```

ONNX 정합성 확인:

```text
Input                : [1, 3, 256, 256]
Output               : [1, 1536, 32, 32]
Max absolute error   : 1.5258789e-05
Mean absolute error  : 2.7548654e-07
PyTorch ≈ ONNX       : True
```
