# JRTIP Publication Target Requirements

## 1. Target Journal

본 프로젝트의 1차 논문 투고 목표 저널은 다음과 같습니다.

**Journal of Real-Time Image Processing (JRTIP)**

본 프로젝트는 단순 anomaly detection 모델 구현이 아니라,

> **PatchCore 기반 memory-bank anomaly detection을 NVIDIA Jetson Orin Nano에서 실시간으로 동작시키기 위해 TensorRT, C++/CUDA, memory-bank retrieval, memory transfer, precision optimization을 통합하고, 정확도–지연시간–메모리–전력 trade-off를 정량적으로 분석하는 연구**

로 포지셔닝합니다.

개발 에이전트는 향후 모든 구현 및 최적화 작업에서
**“이 변경이 JRTIP 논문의 실험 결과 또는 시스템 기여로 연결되는가?”**
를 우선 판단 기준으로 사용해야 합니다.

---

# 2. Paper Positioning

논문의 핵심 메시지는 다음과 같습니다.

```text
PatchCore와 같은 memory-bank 기반 anomaly detector를
edge device에서 실시간으로 배포할 때,
CNN feature extractor만 TensorRT로 최적화하는 것으로는 충분하지 않다.

실제 end-to-end latency는
memory-bank retrieval,
host-device memory transfer,
post-processing,
runtime scheduling 등에 의해 제한될 수 있다.

따라서 model-level, algorithm-level,
system-level optimization을 함께 적용해야 한다.
```

논문의 주인공은 단순히 PatchCore 자체가 아니라 다음입니다.

```text
Real-time Edge Anomaly Detection System
+
Deployment-aware PatchCore architecture
+
C++/CUDA optimization
+
End-to-end bottleneck analysis
```

---

# 3. Working Paper Title

현재 가제:

**End-to-End Optimization of Memory-Bank Anomaly Detection for Real-Time Industrial Inspection on NVIDIA Jetson Orin Nano**

대안:

- Deployment-Aware Optimization of PatchCore for Real-Time Edge Anomaly Detection
- Real-Time Industrial Anomaly Detection on Edge Devices through TensorRT and GPU-Accelerated Memory Retrieval
- System-Level Optimization of Memory-Bank Anomaly Detection for NVIDIA Jetson

---

# 4. Required Research Questions

최종 논문은 최소한 다음 질문에 데이터로 답해야 합니다.

## RQ1. TensorRT precision은 accuracy와 latency에 어떤 영향을 주는가?

비교:

```text
FP32
FP16
INT8
```

측정:

```text
Image AUROC
Pixel AUROC
Feature extraction latency
End-to-end latency
FPS
Memory
Power
Embedding distortion
```

## RQ2. Memory Bank size를 줄이면 accuracy–latency–memory trade-off가 어떻게 변하는가?

Coreset ratio:

```text
1%
2.5%
5%
10%
20%
```

측정:

```text
Memory Bank entries
Memory footprint
NN latency
Total latency
Image AUROC
Pixel AUROC
```

## RQ3. Memory-bank retrieval이 실제 runtime bottleneck인가?

최소 비교:

```text
CPU brute-force
GPU-accelerated search
```

가능한 확장:

```text
OpenMP
CUDA
FAISS GPU
```

## RQ4. System-level optimization이 추가적으로 얼마나 성능을 개선하는가?

순차적으로 비교합니다.

```text
Baseline synchronous runtime
+ persistent buffer
+ pinned memory
+ asynchronous memcpy
+ CUDA stream
+ GPU-resident NN search
+ pipelining
```

각 optimization은 독립적인 ablation 결과를 남겨야 합니다.

## RQ5. 최적화가 다양한 anomaly category에서도 유지되는가?

가능하면 MVTec AD 15개 전체를 평가합니다.

최소 범위:

```text
Objects
- bottle
- capsule
- hazelnut
- screw
- transistor

Textures
- carpet
- grid
- leather
```

---

# 5. JRTIP-Oriented Development Priorities

## Priority 1 — Correct FP32 End-to-End Baseline

반드시 다음 pipeline이 C++에서 완성되어야 합니다.

```text
Image
↓
OpenCV preprocessing
↓
TensorRT feature extraction
↓
Embedding
↓
Memory Bank
↓
Nearest-neighbor search
↓
Anomaly Map
↓
Image-level anomaly score
↓
Decision
```

필수 검증:

```text
Python preprocessing ≈ C++
PyTorch embedding ≈ TensorRT
Python NN ≈ C++ NN
Python anomaly score ≈ C++ anomaly score
```

성능 최적화보다 correctness가 우선입니다.

## Priority 2 — Stage-Level Profiling

다음 항목을 각각 독립적으로 측정해야 합니다.

```text
Preprocessing
H2D copy
TensorRT inference
D2H copy
Embedding reshape
Nearest-neighbor search
Post-processing
Total
```

## Priority 3 — TensorRT Precision Optimization

다음 engine을 모두 생성합니다.

```text
FP32
FP16
INT8
```

각 precision에 대해:

```text
Image AUROC
Pixel AUROC
Latency
FPS
Memory
Power
Embedding error
```

를 함께 기록합니다.

## Priority 4 — Memory Bank Optimization

다음 변수를 실험합니다.

```text
Coreset ratio
Memory Bank precision
```

Memory Bank precision:

```text
FP32
FP16
```

가능하면 다음도 검토합니다.

```text
FP16 storage + FP32 accumulation
```

## Priority 5 — GPU Nearest-Neighbor Search

CPU baseline을 correctness reference로 유지합니다.

이후 최소 하나의 GPU implementation을 추가합니다.

```text
Custom CUDA
or
FAISS GPU
```

## Priority 6 — Runtime Optimization

다음 optimization은 가능한 한 각각 별도 commit과 benchmark 결과를 남깁니다.

```text
Persistent GPU buffers
Pinned host memory
cudaMemcpyAsync
CUDA streams
Avoid D2H embedding transfer
GPU-resident NN
Double buffering
Multithread pipeline
```

여러 optimization을 한 번에 적용한 뒤 결과만 비교하지 않습니다.

---

# 6. Required Experimental Metrics

## Accuracy

필수:

```text
Image AUROC
Pixel AUROC
```

선택 추가:

```text
F1-score
Precision
Recall
Per-category Recall
```

## Runtime Performance

필수:

```text
Mean latency
Median latency
p95 latency
FPS
```

Stage-level:

```text
Preprocess
H2D
TensorRT
D2H
NN
Postprocess
Total
```

## Resource Usage

```text
Host memory
Memory Bank size
TensorRT engine size
GPU / unified memory usage where available
```

## Power

Jetson에서는 가능한 경우 `tegrastats`를 이용해 다음을 기록합니다.

```text
Average power
Peak power
Energy per image
FPS/W
```

Power mode와 clock configuration도 기록해야 합니다.

---

# 7. Benchmark Protocol

권장 baseline:

```text
Warm-up: 50 inference
Measurement: 200 inference
Independent runs: 3
```

보고:

```text
mean
median
standard deviation
p95
```

모든 benchmark 결과에 반드시 다음 metadata를 저장합니다.

```text
timestamp
git commit hash
JetPack
CUDA
TensorRT
OpenCV
power mode
category
precision
coreset ratio
memory-bank precision
NN backend
```

---

# 8. Required Paper Figures

개발 과정에서 다음 figure를 생성할 수 있도록 데이터를 저장해야 합니다.

## Figure 1 — Overall System Architecture

```text
PyTorch training
→ ONNX
→ TensorRT
→ C++/CUDA runtime
→ GPU NN search
→ anomaly output
```

## Figure 2 — Baseline vs Optimized Runtime

```text
Baseline:
TRT → D2H → CPU NN

Optimized:
TRT → GPU-resident NN → Result
```

## Figure 3 — Stage-Level Latency Breakdown

Stacked bar chart:

```text
Preprocess
H2D
TensorRT
D2H
NN
Postprocess
```

## Figure 4 — FP32 / FP16 / INT8 Trade-off

예:

```text
x-axis: latency
y-axis: AUROC
```

## Figure 5 — Coreset Trade-off

```text
Coreset ratio
vs
AUROC
Latency
Memory
```

## Figure 6 — CPU vs GPU NN Search

```text
NN latency
Total latency
Memory
Power
```

## Figure 7 — System Optimization Ablation

```text
Baseline
+ Persistent Buffer
+ Pinned Memory
+ Async
+ GPU NN
+ Pipeline
```

## Figure 8 — Accuracy–Latency Pareto Frontier

최종 configuration 선택 근거로 사용합니다.

## Figure 9 — Power / Energy Comparison

가능한 경우:

```text
FP32
FP16
INT8
```

의 power / energy per image를 비교합니다.

---

# 9. Required Paper Tables

## Table 1 — Hardware / Software Environment

```text
Jetson Orin Nano
JetPack
CUDA
TensorRT
OpenCV
Compiler
Power Mode
```

## Table 2 — Precision Comparison

```text
FP32
FP16
INT8
```

Metrics:

```text
Image AUROC
Pixel AUROC
Latency
FPS
Memory
Power
```

## Table 3 — Coreset Ablation

```text
1%
2.5%
5%
10%
20%
```

## Table 4 — NN Backend Comparison

```text
CPU
OpenMP
CUDA / FAISS
```

## Table 5 — Category-Wise Accuracy

각 MVTec category의:

```text
Image AUROC
Pixel AUROC
```

## Table 6 — System Optimization Ablation

각 system-level optimization 단계의 성능을 비교합니다.

---

# 10. Experimental Validity Requirements

## Test leakage 금지

MVTec test set을 사용해 threshold, quantization parameter 또는 hyperparameter를 직접 tuning하지 않습니다.

## 동일 조건 비교

Precision 비교 시 다음을 고정합니다.

```text
dataset
input size
coreset
NN backend
power mode
warm-up
measurement count
```

## Correctness before Performance

CPU와 CUDA 결과가 다르면 latency 비교를 중단하고 correctness를 먼저 해결합니다.

## Release Build

성능 측정은 반드시 Release build에서 수행합니다.

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

---

# 11. Paper Story to Preserve

개발 에이전트는 논문 스토리가 흐려지는 기능 추가를 지양합니다.

논문에서 보여주고 싶은 흐름:

```text
1. PatchCore는 높은 anomaly detection 성능을 제공한다.

2. 그러나 memory-bank architecture 때문에
   edge deployment에는 계산/메모리 문제가 존재한다.

3. TensorRT로 CNN만 최적화해도
   전체 pipeline에서는 다른 bottleneck이 남는다.

4. Profiling을 통해 NN search와 memory transfer 병목을 확인한다.

5. Coreset reduction, mixed precision,
   GPU NN search, C++/CUDA runtime optimization을 적용한다.

6. Accuracy degradation을 제한하면서
   end-to-end latency, memory, power를 개선한다.

7. Multi-category benchmark를 통해
   결과의 일반성을 검증한다.
```

---

# 12. Current Target Paper Structure

```text
1. Introduction

2. Related Work
   2.1 Industrial Anomaly Detection
   2.2 PatchCore and Memory-Bank Methods
   2.3 Edge AI and TensorRT
   2.4 Real-Time GPU Image Processing

3. Proposed Deployment Architecture
   3.1 PatchCore Baseline
   3.2 Deployment-Aware Partitioning
   3.3 TensorRT Feature Extraction
   3.4 Memory-Bank Retrieval

4. Implementation and Optimization
   4.1 C++ Runtime
   4.2 Precision Optimization
   4.3 Memory-Bank Optimization
   4.4 GPU NN Search
   4.5 Memory Transfer and Pipeline Optimization

5. Experimental Setup
   5.1 Dataset
   5.2 Jetson Environment
   5.3 Metrics
   5.4 Benchmark Protocol

6. Experimental Results
   6.1 Accuracy
   6.2 Runtime Performance
   6.3 Memory
   6.4 Power

7. Ablation Study
   7.1 Precision
   7.2 Coreset Ratio
   7.3 NN Backend
   7.4 System-Level Optimization

8. Discussion

9. Conclusion
```

---

# 13. Agent Decision Rule

새 기능을 구현하기 전에 반드시 다음 질문을 확인합니다.

```text
1. 이 기능이 어느 Research Question을 검증하는가?
2. 어떤 baseline과 비교할 것인가?
3. 어떤 metric으로 효과를 판단할 것인가?
4. 결과가 논문의 어느 Figure/Table에 들어갈 수 있는가?
```

네 질문에 답하지 못하는 기능은 논문 완성 전에는 우선순위를 낮춥니다.

---

# 14. Definition of Done for JRTIP Submission

- [ ] C++ FP32 end-to-end inference correctness 확인
- [ ] Python ↔ C++ 결과 정합성 확인
- [ ] FP32 / FP16 / INT8 완료
- [ ] Coreset ablation 완료
- [ ] CPU / GPU NN 비교 완료
- [ ] System optimization ablation 완료
- [ ] Multi-category evaluation 완료
- [ ] Stage-level latency 확보
- [ ] Memory usage 확보
- [ ] Power / energy 결과 확보
- [ ] Benchmark 반복 측정 완료
- [ ] CSV 자동 저장
- [ ] Figure 생성 script 준비
- [ ] Table 생성 가능
- [ ] 모든 주요 결과에 git commit / environment metadata 존재

이 체크리스트가 완료되기 전에는 단순 최고 FPS 수치만을 근거로 논문 결론을 확정하지 않습니다.
