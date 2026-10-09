# Claims and Evidence Ledger

This file controls what the manuscript may claim. A claim becomes publishable
only when its evidence row is complete and artifact hashes match.

| ID | Candidate claim | Current evidence | Status / limitation |
|---|---|---|---|
| C1 | Optimizing only the feature extractor does not remove the PatchCore edge bottleneck. | FP16 S0: TensorRT 3.435 ms, NN 4332.665 ms, total 4435.535 ms. `results/paper_draft/processed/precision_aggregate.csv` | Supported for bottle 10%; multi-category confirmation pending. |
| C2 | GPU-resident, layout-aware memory-bank retrieval substantially reduces end-to-end latency. | Clean final S5: total 498.486 ms, NN 413.914 ms. FP32 CPU baseline: total about 14.5 s. `results/processed/e9-bottle-fp16-async-transpose-s5-clean01_results.csv` | Combined-stack improvement is supported; isolated attribution requires the final ablation table. |
| C3 | The final FP16 runtime preserves bottle detection accuracy. | Image AUROC 1.0, pixel AUROC 0.9854545306151844 on 83 test images; correctness dumps match direct S5. | Supported for bottle only. |
| C4 | Retrieval optimization improves energy efficiency as well as latency. | Original FP16 CUDA: about 69.75 J/image; final clean S5: 6.70 J/image. | Cross-session comparison; final controlled comparison pending. |
| C5 | Coreset reduction exposes a useful accuracy–latency–memory Pareto frontier. | None yet. | Do not claim until five-ratio experiment completes. |
| C6 | The selected configuration generalizes across industrial object and texture categories. | Bottle only. | Do not claim until all 15 MVTec AD categories are evaluated. |
| C7 | The system is real-time for industrial inspection. | Bottle p95 about 514 ms, throughput about 2 FPS. | Must define an application deadline at or above measured p95 and complete sustained-run validation. |

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
