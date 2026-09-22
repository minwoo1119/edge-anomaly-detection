# Repository Structure

```text
edge-anomaly-detection/
│
├── configs/
│   ├── datasets/
│   ├── models/
│   └── experiments/
│
├── notebooks/
│   ├── mvtec_exploration.ipynb
│   ├── patchcore_training.ipynb
│   ├── anomaly_evaluation.ipynb
│   └── model_export.ipynb
│
├── src/
│   ├── training/
│   ├── evaluation/
│   ├── export/
│   └── common/
│
├── jetson/
│   ├── include/
│   │   ├── Preprocessor.hpp
│   │   ├── TensorRTInferencer.hpp
│   │   ├── MemoryBank.hpp
│   │   ├── NearestNeighborSearch.hpp
│   │   ├── Benchmark.hpp
│   │   └── CudaBuffer.hpp
│   │
│   ├── src/
│   │   ├── main.cpp
│   │   ├── Preprocessor.cpp
│   │   ├── TensorRTInferencer.cpp
│   │   ├── MemoryBank.cpp
│   │   ├── CpuBruteForceSearch.cpp
│   │   ├── CudaBruteForceSearch.cu
│   │   └── Benchmark.cpp
│   │
│   ├── scripts/
│   │   ├── check_env.sh
│   │   ├── build_fp32_engine.sh
│   │   ├── build_fp16_engine.sh
│   │   ├── build_int8_engine.sh
│   │   └── run_benchmark.sh
│   │
│   └── CMakeLists.txt
│
├── benchmarks/
│   ├── configs/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── csv/
│   ├── figures/
│   └── tables/
│
├── models/
│   └── .gitkeep
│
├── datasets/
│   └── .gitkeep
│
├── docs/
│   ├── README.md
│   ├── project_scope.md
│   ├── architecture.md
│   ├── repository_structure.md
│   ├── development_roadmap.md
│   ├── experiments.md
│   ├── benchmarking.md
│   ├── cpp_cuda_optimization.md
│   ├── jetson_deployment.md
│   ├── paper_plan.md
│   └── agent_instructions.md
│
├── .gitignore
├── README.md
└── requirements.txt
```

## configs
실험 변수는 hard-code하지 않습니다.

예:
```yaml
dataset:
  name: mvtec_ad
  category: bottle

model:
  name: patchcore
  backbone: wide_resnet50_2
  layers: [layer2, layer3]
  input_size: 256
  coreset_ratio: 0.10

deployment:
  precision: fp16
  memory_bank_precision: fp16
  nn_backend: cuda
```

## models
Git 제외:
```text
*.ckpt
*.onnx
*.onnx.data
*.engine
*.npy
```

## results
논문용 산출물:
- CSV
- Figure
- Table

가능하면 script에서 재생성 가능해야 합니다.
