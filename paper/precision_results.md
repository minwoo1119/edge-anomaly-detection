# Controlled Precision Experiment

The bottle precision experiment compares FP32, FP16 and entropy-calibrated INT8
feature extractors with the same 10% FP32 memory bank and the final
`cuda_tiled_async_transpose` S5 retrieval path. All configurations used the same
Release executable from commit `6985796`, 50 warm-up iterations, 200 measured
iterations, three independent runs, and MAXN_SUPER_ID2. Accuracy was evaluated
on all 83 bottle test images without tuning a threshold on the test set.

| Precision | Image AUROC | Pixel AUROC | TRT mean (ms) | NN mean (ms) | Total mean (ms) | Run SD (ms) | Mean p95 (ms) | Power (W) | Energy (J/image) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 1.000000 | 0.985461 | 20.212 | 459.677 | 558.511 | 0.981 | 623.319 | 13.381 | 7.473 |
| FP16 | 1.000000 | 0.985455 | 3.785 | 467.057 | 550.245 | 13.934 | 616.640 | 13.194 | 7.261 |
| INT8 | 0.688889 | 0.761489 | 2.703 | 440.633 | 523.076 | 32.805 | 575.864 | 14.130 | 7.418 |

FP16 preserves bottle accuracy while reducing feature extraction time by 81.3%
relative to FP32. The end-to-end reduction is only 1.48% because exact
memory-bank retrieval accounts for roughly 82–85% of total latency. This is the
central systems result: reporting TensorRT latency alone would greatly overstate
the deployment benefit.

The tested INT8 PTQ path is rejected. It reduces TensorRT time by another
1.08 ms relative to FP16 and total mean latency by 4.94%, but image AUROC drops
by 0.311 and pixel AUROC by 0.224. The result applies to the recorded
train-normal-only, 100-image entropy calibration procedure; it does not establish
that QAT or layer-selective mixed precision would fail.

The canonical inputs are `results/processed/precision_results.csv` and
`results/metadata/paper_precision_matrix_run01.json`. Generated Table 2,
Figures 4/8/9, and their input-hash coverage manifest are under
`results/paper_precision/`. These generated files are intentionally excluded
from Git because the raw and processed experiment tree is an artifact store.
