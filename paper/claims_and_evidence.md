# Claims and Evidence Ledger

This file controls what the manuscript may claim. A claim becomes publishable
only when its evidence row is complete and artifact hashes match.

| ID | Candidate claim | Current evidence | Status / limitation |
|---|---|---|---|
| C1 | Optimizing only the feature extractor does not remove the PatchCore edge bottleneck. | FP16 S0: TensorRT 3.435 ms, NN 4332.665 ms, total 4435.535 ms. In the final 15-category S5 matrix, mean TensorRT time is 3.44 ms while category mean NN time ranges from 119.257 to 780.257 ms. `results/processed/multi_category_results.csv` | Supported for the tested 10% coreset artifacts across all 15 categories. |
| C2 | CUDA memory-bank retrieval substantially reduces end-to-end latency, with kernel organization determining most of the gain. | Controlled FP16/S0 backend matrix: CPU/direct CUDA/warp/tiled/async tiled total latency is 14505.975/4420.934/844.294/613.777/510.285 ms. Async tiled gives 28.43x end-to-end and 35.70x NN speedup over CPU. `results/processed/nn_backend_results.csv` | Supported for bottle with the fixed 10% bank. S0 transpose-assisted async is 3.46% slower than async tiled and is retained as a negative result. |
| C3 | The final FP16 runtime preserves bottle detection accuracy. | Controlled S5 precision matrix: FP32 image/pixel AUROC 1.0/0.985461354; FP16 1.0/0.985454531 on 83 test images. Accuracy and performance rows share executable, config, engine and bank hashes. | Supported for bottle only. |
| C4 | The staged runtime reduces steady-state interval and energy per image while preserving accuracy. | Controlled S0–S6 matrix: S0 interval 548.333 ms and 7.121 J/image; S6 interval 482.350 ms and 6.426 J/image. All stages retain image/pixel AUROC 1.0/0.985455. | Supported for bottle. S6 single-frame completion latency is 964.625 ms and must be reported separately from interval. |
| C5 | Coreset reduction exposes a useful accuracy-latency-memory Pareto frontier. | Five independently trained bottle ratios preserve image AUROC 1.0; 1%/20% pixel AUROC is 0.984961/0.985500 while latency is 72.093/992.520 ms and bank size is 12.539/250.799 MB. `results/processed/coreset_results.csv` | Supported for bottle with seed 42. The 1% variant dominates the tested higher ratios under image AUROC; the small pixel-AUROC differences must be reported. |
| C6 | The selected configuration retains strong accuracy across industrial object and texture categories. | All 15 MVTec AD categories: macro image AUROC 0.983314 and macro pixel AUROC 0.979761; 45 performance runs. `results/processed/multi_category_results.csv` and `results/paper_mvtec15_complete/tables/table_5_category_accuracy.csv` | Supported for MVTec AD with independently trained category-specific 10% coresets. Toothbrush image AUROC (0.919444) and wood pixel AUROC (0.929994) bound the claim. |
| C7 | The system is real-time for industrial inspection. | Bottle p95 about 514 ms, throughput about 2 FPS. | Must define an application deadline at or above measured p95 and complete sustained-run validation. |
| C8 | Naive entropy-calibrated INT8 PTQ is unsuitable for this PatchCore deployment. | Controlled S5 matrix: image AUROC fell from 1.0 (FP16) to 0.688889 and pixel AUROC from 0.985455 to 0.761489. Mean total latency changed from 550.245 to 523.076 ms because NN search remained dominant. | Supported as a bottle-specific negative result; do not generalize to QAT or mixed precision. |

## Reporting rules

- Use clean-build measurements for final headline numbers.
- Join accuracy and benchmark rows only when executable, config, engine and
  memory-bank SHA-256 values all match.
- Keep negative results: direct device-NCHW S5 regressed to about 1.52 s because
  the search accessed channel-major queries with a poor memory pattern.
- Never select a coreset ratio, INT8 calibration setting or threshold on the
  MVTec test set.
- Report three-run variation and p95, not only mean latency or FPS.
- Distinguish kernel-search latency from end-to-end latency and from `trtexec`.
- Treat `tegrastats` utilization as telemetry; a raw 101% sample is a tool
  artifact and is not evidence of utilization beyond 100%.
