# Manuscript Outline

Working title: **End-to-End Optimization of Memory-Bank Anomaly Detection for
Industrial Inspection on NVIDIA Jetson Orin Nano**

Target: Journal of Real-Time Image Processing, maximum 12 double-column pages
including references and biographies.

## 1. Introduction

- Industrial anomaly detection often has abundant normal samples and sparse,
  changing defect classes.
- PatchCore offers strong unsupervised accuracy but its large memory-bank search
  becomes a deployment bottleneck on an edge SoC.
- The paper studies the complete runtime rather than reporting CNN engine time.
- Contributions: traced C++/TensorRT/CUDA runtime; stage-level bottleneck study;
  layout-aware GPU retrieval; accuracy–latency–memory–energy analysis; and
  multi-category validation (pending).

## 2. Related Work

- Industrial unsupervised anomaly detection and PatchCore
- Coreset sampling and nearest-neighbor retrieval
- Edge inference and TensorRT precision optimization
- GPU memory layout, tiled search and asynchronous transfer
- Real-time industrial inspection on embedded SoCs

## 3. Method and Runtime Architecture

- Training uses train-normal images only.
- WideResNet50-2 layer2/layer3 features, 256×256 input, 1536×32×32 embedding.
- ONNX/TensorRT feature boundary and C++ postprocessing.
- Memory-bank layout and exact Euclidean nearest-neighbor scoring.
- CPU reference, CUDA variants, tiled bank reuse, `cp.async` double buffering,
  and NCHW-to-patch-major shared-memory transpose.
- Artifact lineage, hash validation and threshold leakage prevention.

## 4. Experimental Setup

- MVTec AD categories and split policy
- Jetson Orin Nano, MAXN_SUPER_ID2, JetPack 6.2.1, TensorRT 10.3
- Accuracy: image AUROC and pixel AUROC
- Runtime: 50 warm-up, 200 measurements, three independent runs; mean, median,
  standard deviation and p95
- Resource metrics: bank/engine/host memory, average/peak power, energy/image
- Fixed-condition and hash-join policy

## 5. Results

- FP32/FP16/INT8 precision comparison
- Coreset ratio comparison: 1%, 2.5%, 5%, 10%, 20%
- Retrieval backend and kernel progression
- End-to-end system ablation
- Category-wise accuracy and performance
- Accuracy–latency Pareto frontier and selected deployment point

## 6. Discussion

- Why TensorRT time is a small fraction of the original end-to-end latency
- Why direct NCHW GPU search regressed and why explicit transpose recovered it
- Unified-memory and power trade-offs on Orin Nano
- Meaning of real-time relative to an explicit inspection interval
- Limits: one edge platform, MVTec domain, exact brute-force retrieval

## 7. Conclusion

State only claims marked supported in `claims_and_evidence.md`. Quantitative
headline results must reference the final clean merged CSVs.

## Planned figures and tables

- Fig. 1 complete training/deployment architecture
- Fig. 2 baseline and optimized data paths
- Fig. 3 stage latency breakdown
- Fig. 4 precision accuracy/latency
- Fig. 5 coreset accuracy/latency/memory
- Fig. 6 retrieval backend latency
- Fig. 7 system optimization ablation
- Fig. 8 Pareto frontier
- Fig. 9 power and energy
- Tables 1–6 as defined in `docs/jrtip_publication_target.md`
