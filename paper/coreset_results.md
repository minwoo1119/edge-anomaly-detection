# Bottle Coreset Trade-off

Five bottle models were trained independently with seed 42 at coreset ratios
1%, 2.5%, 5%, 10%, and 20%. No smaller bank is a truncation of the 10% export.
Each configuration uses an FP16 TensorRT engine, FP32 bank storage, the
`cuda_tiled_async_transpose` S5 runtime, the full 83-image test set, and three
independent 50-warm-up/200-measured Jetson runs.

| Ratio | Entries | Bank (MB) | Image AUROC | Pixel AUROC | Mean latency (ms) | Mean p95 (ms) | Energy (J/image) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 2,140 | 12.539 | 1.000000 | 0.984961 | 72.093 | 80.091 | 0.865 |
| 2.5% | 5,350 | 31.348 | 1.000000 | 0.985233 | 155.172 | 171.042 | 2.181 |
| 5% | 10,700 | 62.695 | 1.000000 | 0.985332 | 283.882 | 314.485 | 3.685 |
| 10% | 21,401 | 125.396 | 1.000000 | 0.985452 | 570.156 | 634.470 | 8.917 |
| 20% | 42,803 | 250.799 | 1.000000 | 0.985500 | 992.520 | 1097.090 | 13.599 |

All ratios retain image AUROC 1.0, and the complete pixel-AUROC range is only
0.000540. The 1% variant therefore dominates the tested larger banks under
image AUROC, latency, storage, and energy. This supports a bottle-specific
accuracy-latency-memory frontier; it does not establish that 1% is optimal for
other categories or training seeds.

Canonical evidence:

- `results/processed/coreset_results.csv`: 15 hash-joined performance rows.
- `results/csv/paper_coreset_accuracy.csv`: five full-test accuracy rows.
- `results/metadata/paper_coreset_bottle_run02.json`: matrix provenance.
- `results/paper_mvtec15_complete/tables/table_3_coreset.csv`: generated table.
- `results/paper_mvtec15_complete/figures/figure_5_coreset_tradeoff.svg`:
  generated accuracy, latency, and memory figure.
