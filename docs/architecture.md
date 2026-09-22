# System Architecture

## 1. Overall Architecture

```text
                     DEVELOPMENT / TRAINING

MVTec AD
   ↓
Anomalib / PyTorch
   ↓
PatchCore
   ├──────────────────────────────┐
   │                              │
   ↓                              ↓
Feature Extractor            Memory Bank
WideResNet50-2               [N,1536]
layer2 + layer3                  │
   ↓                              │
Feature Fusion                   │
   ↓                              │
ONNX                             │
   ↓                              │
TensorRT Engine                  │
                                  │
                     JETSON       │
                                  │
Input Image                       │
   ↓                              │
OpenCV Preprocessing              │
   ↓                              │
TensorRT Feature Extractor        │
   ↓                              │
Embedding [1,1536,32,32]          │
   ↓                              │
Reshape / transpose               │
[1024,1536]                       │
   └───────────────┬──────────────┘
                   ↓
          Nearest Neighbor Search
                   ↓
             Patch Distances
                   ↓
              Anomaly Map
                   ↓
              Image Score
                   ↓
               Threshold
                   ↓
                OK / NG
```

---

## 2. Deployment Boundary

전체 PatchCore를 하나의 ONNX/TensorRT graph로 export하지 않습니다.

이유:
- Anomalib PostProcessor의 data-dependent branch
- Memory Bank retrieval은 CNN dense operator와 성격이 다름
- TensorRT와 C++/CUDA가 각각 잘하는 영역을 분리 가능

따라서:

```text
TensorRT
→ CNN feature extraction + feature aggregation

C++ / CUDA
→ Memory-bank retrieval
→ distance computation
→ anomaly scoring
→ anomaly map
→ threshold
```

---

## 3. Python Responsibilities

### Training
- Dataset setup
- PatchCore construction
- Normal feature extraction
- Coreset sampling
- Memory Bank generation
- Checkpoint save

### Evaluation
- Image AUROC
- Pixel AUROC
- heatmap
- score distribution
- threshold analysis
- FP/FN analysis

### Export
- checkpoint load
- Memory Bank extraction
- Feature Extractor wrapper
- ONNX export
- ONNX checker
- PyTorch ↔ ONNX comparison

---

## 4. Jetson C++ Responsibilities

### Preprocessor
Input:
```text
cv::Mat BGR
```

Output:
```text
FP32 NCHW [1,3,256,256]
```

Operations:
```text
resize
BGR → RGB
uint8 → float
0~1 scaling
ImageNet normalization
HWC → CHW
```

### TensorRTInferencer
- deserialize engine
- create execution context
- persistent device buffers
- H2D
- `enqueueV3`
- output handling

TensorRT 10.x:
- `getNbIOTensors`
- `getIOTensorName`
- `getTensorShape`
- `setTensorAddress`
- `enqueueV3`

### MemoryBank
- load memory bank
- dtype conversion
- metadata validation
- optional FP16 bank

### NearestNeighborSearch
교체 가능한 interface로 설계:

```cpp
class INearestNeighborSearch {
public:
    virtual SearchResult search(...) = 0;
};
```

Implementations:
- CPU brute-force
- OpenMP
- CUDA
- FAISS GPU

---

## 5. Runtime Pipeline

Baseline:
```text
image
→ preprocess
→ H2D
→ TensorRT
→ D2H embedding
→ NN search
→ score
→ output
```

Optimized target:
```text
capture/input
→ bounded queue
→ preprocess
→ pinned buffer
→ async H2D
→ TensorRT stream
→ GPU NN search
→ minimal D2H
→ output
```

---

## 6. Memory Ownership

CUDA/TensorRT resource는 RAII 기반으로 관리합니다.

금지:
- per-frame `cudaMalloc`
- per-frame engine deserialize
- per-frame large `std::vector` 재할당
- raw ownership ambiguity

권장:
- persistent buffers
- smart pointers
- explicit ownership
- move semantics where useful

---

## 7. Precision Strategy

Feature Extractor:
- FP32
- FP16
- INT8

Memory Bank:
- FP32
- FP16
- optional INT8/compressed

Distance accumulation:
- FP32 baseline
- FP16 storage + FP32 accumulation 후보

---

## 8. Correctness Chain

반드시 다음 순서로 검증합니다.

```text
PyTorch
≈ ONNX Runtime
≈ TensorRT
≈ C++ runtime
```

추가:
```text
Python NN
≈ CPU C++ NN
≈ CUDA NN
```

평가:
- shape
- MAE
- max error
- cosine similarity
- nearest-neighbor agreement
- final anomaly score
