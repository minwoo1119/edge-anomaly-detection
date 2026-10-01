# JRTIP Result Flow Implementation Guide

## 1. 목적

이 문서는 현재 구현된 Edge AI anomaly detection 프로젝트가 **Journal of Real-Time Image Processing (JRTIP)** 논문에 적합한 실험 흐름을 만들 수 있도록 코드와 benchmark 구조를 보정하기 위한 개발 지침이다.

핵심 연구 흐름은 다음과 같다.

```text
FP32 Baseline
→ Stage-level Profiling
→ Bottleneck Identification
→ Precision Optimization
→ Memory-Bank Optimization
→ NN Search Optimization
→ System-Level Optimization
→ Accuracy Preservation
→ Multi-Category Validation
→ Pareto Analysis
```

에이전트는 모든 코드 변경이 위 흐름의 어느 단계에 해당하는지 명시해야 한다.

---

## 2. 논문에서 보여줄 핵심 스토리

```text
1. PatchCore는 높은 anomaly detection 성능을 제공한다.
2. 하지만 Memory Bank 기반 구조는 edge device에서 latency와 memory overhead를 유발한다.
3. TensorRT로 CNN만 최적화해도 end-to-end bottleneck이 남을 수 있다.
4. Profiling을 통해 NN search, memory transfer 등의 병목을 찾는다.
5. Precision, coreset, Memory Bank, NN backend를 각각 최적화한다.
6. C++/CUDA runtime 수준에서 memory transfer와 pipeline overhead를 줄인다.
7. 각 최적화의 accuracy degradation을 검증한다.
8. 여러 MVTec category에서 동일한 경향을 확인한다.
9. Accuracy–Latency–Memory–Power Pareto frontier로 최종 configuration을 선정한다.
```

결과를 특정 방향으로 강제하지 않는다. 실제 실험 결과에 따라 FP16, INT8, 특정 coreset ratio 등이 최종 선택될 수 있어야 한다.

---

# 3. Phase 1 — FP32 End-to-End Baseline

논문의 모든 비교 기준이 되는 baseline을 먼저 고정한다.

```text
Feature Precision     : FP32
Coreset Ratio         : 10%
Memory Bank Precision : FP32
NN Backend            : CPU brute-force
Runtime               : synchronous
Host Memory           : pageable
```

Pipeline:

```text
Image
→ Preprocessing
→ H2D
→ TensorRT FP32
→ D2H Embedding
→ Embedding Transform
→ CPU NN Search
→ Anomaly Map
→ Image Score
→ Decision
```

구성 요소를 분리한다.

```text
Preprocessor
TensorRTInferencer
MemoryBank
EmbeddingAdapter
INearestNeighborSearch
CpuBruteForceSearch
AnomalyScorer
AnomalyMapGenerator
BenchmarkRunner
```

`main.cpp`에는 orchestration만 남기고 세부 구현을 몰아넣지 않는다.

---

# 4. Correctness Gate

성능 최적화 전에 아래 검증을 반드시 통과한다.

## 4.1 Preprocessing

비교:

```text
Python/Anomalib
vs
C++
```

검증:

```text
shape
min/max
mean/std
selected tensor values
MAE
max absolute error
```

## 4.2 TensorRT Embedding

비교:

```text
PyTorch
vs
ONNX Runtime
vs
TensorRT C++
```

검증:

```text
shape
MAE
RMSE
max error
cosine similarity
```

## 4.3 Embedding Layout

TensorRT output:

```text
[1,1536,32,32]
```

NN query:

```text
[1024,1536]
```

NCHW memory를 단순 reinterpret하면 안 된다.

정확한 mapping:

```text
query_idx = h * W + w
channel   = c
source    = output[c * H * W + h * W + w]
target    = query[query_idx * C + c]
```

이를 unit test로 검증한다.

## 4.4 Nearest Neighbor

비교:

```text
Python PatchCore NN
vs
C++ CPU NN
```

검증:

```text
Top-1 index agreement
Top-1 distance
Top-k distances
```

## 4.5 Anomaly Score / Map

Anomalib PatchCore의 실제 구현과 비교한다.

반드시 확인:

```text
nearest-neighbor semantics
image-level score
reweighting 여부
anomaly map reshape
interpolation
smoothing/post-processing
threshold application
```

단순화한 score를 사용했다면 별도 baseline임을 명시해야 하며 PatchCore와 동일하다고 주장하면 안 된다.

---

# 5. Stage-Level Profiling

필수 stage:

```text
Preprocess
H2D
TensorRT
D2H
Embedding Transform
NN Search
Postprocess
Total
```

CPU timing:

```cpp
std::chrono::steady_clock
```

GPU timing:

```text
cudaEventRecord
cudaEventElapsedTime
```

GPU kernel latency를 CPU wall-clock만으로 측정하지 않는다.

---

# 6. Benchmark Protocol

기본:

```text
Warm-up        : 50
Measurement    : 200
Independent Run: 3
Build Type     : Release
```

통계:

```text
mean
median
std
p95
min
max
```

모든 결과는 CSV로 저장한다.

---

# 7. Benchmark CSV Schema

필수 metadata:

```text
timestamp
git_commit
device
jetpack
cuda
tensorrt
opencv
compiler
build_type
power_mode
category
model
precision
coreset_ratio
bank_precision
nn_backend
optimization_stage
run_id
```

Accuracy:

```text
image_auroc
pixel_auroc
```

Latency:

```text
preprocess_ms
h2d_ms
trt_ms
d2h_ms
embedding_transform_ms
nn_ms
post_ms
total_ms
fps
```

Resource:

```text
bank_entries
bank_size_mb
engine_size_mb
host_memory_mb
avg_power_w
peak_power_w
energy_per_image_mj
```

---

# 8. Baseline Bottleneck Analysis

Baseline에서 다음 비율을 자동 계산한다.

```text
preprocess_share
h2d_share
trt_share
d2h_share
nn_share
post_share
```

질문에 답할 수 있어야 한다.

```text
TensorRT가 bottleneck인가?
NN search가 bottleneck인가?
Embedding D2H가 의미 있는 overhead인가?
Post-processing 비용은 어느 정도인가?
```

결과는 stacked latency figure로 생성 가능해야 한다.

---

# 9. Precision Ablation

비교:

```text
FP32
FP16
INT8
```

고정:

```text
same dataset
same input size
same coreset
same bank precision
same NN backend
same power mode
```

기록:

```text
Image AUROC
Pixel AUROC
TRT latency
Total latency
FPS
Memory
Power
Embedding MAE
Embedding cosine similarity
Top-1 NN agreement
```

---

# 10. INT8 Embedding Analysis

FP32를 reference로:

```text
MAE
RMSE
Relative L2 Error
Cosine Similarity
```

NN consistency:

```text
Top-1 agreement
Top-k overlap
distance correlation
```

Score consistency:

```text
Pearson correlation
Spearman correlation
absolute anomaly-score error
```

INT8의 accuracy 변화가 발생했을 때 원인을 분석할 수 있어야 한다.

---

# 11. Coreset Ablation

비교:

```text
1%
2.5%
5%
10%
20%
```

기록:

```text
Memory Bank entries
Memory Bank size
NN latency
Total latency
Image AUROC
Pixel AUROC
Power
```

다음 figure를 만들 수 있어야 한다.

```text
Coreset Ratio vs AUROC
Coreset Ratio vs NN Latency
Coreset Ratio vs Memory
```

---

# 12. Memory Bank Precision

비교:

```text
FP32 Bank
FP16 Bank
```

선택:

```text
FP16 storage + FP32 accumulation
```

측정:

```text
Memory
NN latency
Image AUROC
Pixel AUROC
NN agreement
```

---

# 13. NN Backend Interface

공통 interface를 둔다.

```cpp
class INearestNeighborSearch {
public:
    virtual SearchResult search(...) = 0;
    virtual ~INearestNeighborSearch() = default;
};
```

후보:

```text
CpuBruteForceSearch
OpenMPNearestNeighborSearch
CudaNearestNeighborSearch
FaissNearestNeighborSearch
```

최소 논문 비교:

```text
CPU
GPU
```

---

# 14. GPU NN Correctness Gate

GPU 결과는 CPU reference와 비교한다.

```text
Top-1 index agreement
Distance error
Image score error
Anomaly map error
```

correctness가 통과하기 전에는 GPU speedup을 논문 결과로 사용하지 않는다.

---

# 15. System Optimization Ablation

다음 단계는 순차적으로 적용하고 각각 결과를 남긴다.

```text
S0 Baseline synchronous
S1 Persistent device buffers
S2 Pinned host memory
S3 Async H2D/D2H
S4 CUDA stream
S5 GPU-resident NN
S6 Pipeline overlap
```

CSV에:

```text
optimization_stage=S0
...
optimization_stage=S6
```

를 기록한다.

여러 최적화를 한 번에 적용하고 총 성능만 비교하지 않는다.

---

# 16. GPU-Resident Pipeline

최종 후보:

```text
Input
→ Pinned Host Buffer
→ Async H2D
→ TensorRT
→ Device Embedding
→ GPU NN
→ Device Patch Score
→ Small Result D2H
```

목표는 TensorRT output 전체를 매 frame CPU로 복사하는 비용을 줄이는 것이다.

효과를 별도 ablation으로 측정한다.

---

# 17. Multithread / Pipeline Optimization

선택적 최종 구조:

```text
Capture/Input Thread
→ Bounded Queue
→ Inference Thread
→ Result Queue
→ Visualization/Logging Thread
```

반드시 별도로 측정:

```text
single-frame latency
sustained throughput/FPS
```

throughput 향상을 latency 향상으로 표현하지 않는다.

---

# 18. Power Measurement

Jetson benchmark에 `tegrastats` 결과를 연결한다.

기록:

```text
Average Power
Peak Power
Temperature
CPU/GPU Utilization
```

파생 metric:

```text
Energy per image
FPS/W
```

모든 power 결과는 experiment ID와 연결한다.

---

# 19. Multi-Category Validation

가능하면 MVTec AD 15개 전체를 평가한다.

최소:

```text
bottle
capsule
hazelnut
screw
transistor
carpet
grid
leather
```

최종 selected configuration 최소 세 개를 비교한다.

예:

```text
FP32 Baseline
FP16 Optimized
INT8 Optimized
```

각 category별:

```text
Image AUROC
Pixel AUROC
Latency
FPS
```

---

# 20. Pareto Analysis

최종 configuration은 단순 최고 FPS로 선정하지 않는다.

동시에 고려:

```text
Accuracy
Latency
Memory
Power
```

분석 script는 Pareto frontier를 생성할 수 있어야 한다.

예:

```text
x = latency
y = Image AUROC
marker size = memory
```

Power는 별도 plot 또는 marker attribute로 추가 가능하다.

---

# 21. 결과 해석 규칙

결과는 특정 방향으로 나올 필요가 없다.

예:

```text
INT8 accuracy drop 큼
→ FP16이 더 좋은 Pareto point일 수 있음

2.5% coreset에서 accuracy 급락
→ 5%가 sweet spot일 수 있음

GPU NN의 power 증가가 큼
→ latency-power trade-off 분석 가능
```

실험 결과를 그대로 해석한다.

---

# 22. 논문 Figure와 데이터 매핑

```text
Figure 1  Overall Architecture
Figure 2  Baseline vs Optimized Pipeline
Figure 3  Stage-Level Latency Breakdown
Figure 4  Precision Trade-off
Figure 5  Coreset Trade-off
Figure 6  CPU vs GPU NN
Figure 7  System Optimization Ablation
Figure 8  Accuracy-Latency Pareto
Figure 9  Power / Energy
```

권장 데이터:

```text
results/csv/baseline.csv
results/csv/precision_ablation.csv
results/csv/coreset_ablation.csv
results/csv/nn_backend_ablation.csv
results/csv/system_optimization_ablation.csv
results/csv/multi_category.csv
```

---

# 23. 권장 Analysis Scripts

```text
scripts/
├── analyze_baseline.py
├── analyze_precision.py
├── analyze_coreset.py
├── analyze_nn_backend.py
├── analyze_system_optimization.py
├── analyze_power.py
├── build_pareto_frontier.py
└── generate_paper_tables.py
```

출력:

```text
results/
├── csv/
├── figures/
└── tables/
```

Figure와 Table을 가능한 한 수동 편집하지 않고 재생성 가능하게 만든다.

---

# 24. Agent 작업 보고 형식

각 작업은 다음 형식으로 보고한다.

```text
Task:
Related RQ:
Modified Files:
Reason:
Expected Paper Output:
Correctness Validation:
Benchmark Validation:
Done Criteria:
```

예:

```text
Task:
GPU-resident NN search 구현

Related RQ:
RQ3, RQ4

Modified Files:
jetson/include/NearestNeighborSearch.hpp
jetson/src/CudaNearestNeighborSearch.cu
jetson/src/TensorRTInferencer.cpp

Reason:
D2H embedding copy와 CPU NN bottleneck 제거

Expected Paper Output:
Figure 6
Figure 7
Table 4
Table 6

Correctness Validation:
CPU NN과 Top-1 index/distance 비교

Benchmark Validation:
NN latency / total latency / power

Done Criteria:
Correctness tolerance 만족
Release build 성공
CSV benchmark 생성
```

---

# 25. Git Commit 단위

권장:

```text
feat: FP32 PatchCore C++ end-to-end baseline 완성
feat: 단계별 latency benchmark 및 CSV 로깅 추가
feat: TensorRT FP16 실험 파이프라인 추가
feat: TensorRT INT8 임베딩 정합성 분석 추가
feat: coreset ratio 실험 자동화 추가
feat: CPU 기반 Memory Bank 검색 baseline 구현
perf: CUDA Memory Bank 최근접 이웃 검색 구현
perf: pinned memory 기반 전송 최적화 적용
perf: CUDA stream 비동기 inference 적용
perf: GPU-resident NN pipeline 구현
feat: Jetson power benchmark 자동 수집 추가
feat: JRTIP 논문 결과 그래프 생성 자동화
```

파일 이동 시 반드시:

```text
[파일 이동]
source
→ destination
```

을 보고한다.

---

# 26. 금지 사항

- 예상 결과를 강제로 만들기
- 유리한 configuration만 선택적으로 보고
- test set 기반 tuning
- 여러 최적화를 동시에 적용해 원인 분리 불가
- correctness 불일치 상태에서 speed comparison
- Debug build benchmark
- warm-up 없는 benchmark
- 단 1회 latency 측정
- `trtexec` latency를 end-to-end latency로 표현
- benchmark CSV 수동 조작
- 실패 결과를 근거 없이 제외
- 환경 metadata 누락
- git commit hash 누락

---

# 27. Definition of Done

다음 흐름이 실제 코드와 실험 결과로 생성 가능해야 한다.

```text
Baseline
↓
Bottleneck Analysis
↓
Precision Ablation
↓
Coreset Ablation
↓
Memory-Bank Precision Ablation
↓
NN Backend Ablation
↓
System Optimization Ablation
↓
Power Analysis
↓
Multi-Category Validation
↓
Pareto Analysis
↓
Paper Figures / Tables
```

체크리스트:

- [ ] FP32 end-to-end correctness
- [ ] Python ↔ C++ correctness
- [ ] stage-level profiling
- [ ] FP32 / FP16 / INT8
- [ ] embedding distortion analysis
- [ ] coreset ablation
- [ ] FP32 / FP16 Memory Bank
- [ ] CPU / GPU NN comparison
- [ ] system optimization ablation
- [ ] power measurement
- [ ] multi-category evaluation
- [ ] CSV automation
- [ ] figure automation
- [ ] table automation
- [ ] Pareto analysis
- [ ] environment metadata
- [ ] git traceability

이 조건이 충족되면 본 프로젝트는 단순 Edge deployment demo가 아니라,
**JRTIP 논문의 연구 흐름을 재현 가능한 데이터로 뒷받침하는 research-grade implementation** 상태로 간주한다.
