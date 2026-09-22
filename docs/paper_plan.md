# Paper Plan

## Working Titles

1. End-to-End Optimization of PatchCore for Real-Time Industrial Anomaly Detection on NVIDIA Jetson
2. System-Level Optimization of Memory-Bank Anomaly Detection for Resource-Constrained Edge Devices
3. Deployment-Aware Optimization of PatchCore: Accuracy–Latency–Memory Trade-offs on NVIDIA Jetson Orin Nano

---

## Core Message

부족한 메시지:
```text
PatchCore를 Jetson에 올렸다.
```

목표 메시지:
```text
Memory-bank 기반 anomaly detection의 edge deployment에서는
CNN inference 최적화뿐 아니라
memory-bank size, retrieval backend, precision,
memory transfer, runtime architecture를 함께 최적화해야 한다.
```

---

## Hypotheses

H1:
FP16은 FP32 대비 accuracy degradation이 작으면서 latency를 줄일 것이다.

H2:
INT8은 embedding distortion 때문에 anomaly distance ranking에 더 민감할 수 있다.

H3:
Coreset 감소는 일정 구간까지 AUROC 손실을 제한하면서 memory와 NN latency를 크게 줄일 것이다.

H4:
TensorRT 최적화 이후 NN search가 주요 bottleneck으로 이동할 것이다.

H5:
GPU-resident NN search와 async runtime은 추가적인 end-to-end latency 감소를 제공할 것이다.

---

## Paper Structure

1. Introduction
2. Related Work
3. Method
   - PatchCore baseline
   - Deployment-aware partition
   - TensorRT feature extraction
   - Memory Bank optimization
   - NN search
   - C++/CUDA runtime
4. Experimental Setup
5. Results
6. Ablation
7. Discussion
8. Conclusion

---

## Figures

1. Overall architecture
2. Stage latency breakdown
3. Precision vs AUROC/latency
4. Coreset vs latency/memory
5. Coreset vs AUROC
6. NN backend latency
7. Accuracy-latency Pareto
8. Power / energy comparison

---

## Tables

1. Hardware/software
2. FP32/FP16/INT8
3. Category-wise AUROC
4. Coreset ablation
5. NN backend
6. System optimization ablation

---

## Avoid

- bottle 단일 결과로 일반화
- test set threshold tuning
- 1회 latency 측정
- FPS만 제시
- `trtexec`와 end-to-end latency 혼용
- accuracy 확인 없는 INT8
- 여러 최적화를 동시에 적용해 원인을 모호하게 함

---

## Paper Readiness Checklist

- [ ] multi-category
- [ ] FP32/FP16/INT8
- [ ] coreset ablation
- [ ] NN backend
- [ ] latency breakdown
- [ ] power
- [ ] repeated measurement
- [ ] system ablation
- [ ] reproducible config
- [ ] auto CSV
- [ ] auto figures
- [ ] baseline model
