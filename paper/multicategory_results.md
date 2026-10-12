# MVTec AD 15-Category Results

The selected deployment configuration uses an FP16 TensorRT feature extractor,
an FP32 memory bank, the `cuda_tiled_async_transpose` nearest-neighbor backend,
and system stage S5. Every category uses an independently trained seed-42 model
and 10% coreset. Accuracy is measured over the complete category test split
without test-set threshold tuning. Performance uses 50 warm-up iterations and
200 measured iterations in each of three independent Jetson runs.

Across all 15 categories, macro image AUROC is 0.983314 and macro pixel AUROC is
0.979761. The 10 object categories average 0.978531/0.984692, while the five
texture categories average 0.992882/0.969898. Toothbrush has the lowest image
AUROC at 0.919444, and wood has the lowest pixel AUROC at 0.929994.

Macro mean end-to-end latency is 574.334 ms and mean throughput is 1.991
images/s. Mean power is 14.043 W and mean energy is 8.128 J/image. Category mean
latency ranges from 164.050 ms for toothbrush (6,144 bank entries) to 906.637 ms
for hazelnut (40,038 entries). The feature extractor remains a small fraction of
the pipeline: category mean TensorRT time is approximately 3.3--3.8 ms, while
exact nearest-neighbor search ranges from 119.257 to 780.257 ms.

Canonical evidence:

- `results/processed/multi_category_results.csv`: 45 hash-joined benchmark rows.
- `results/csv/paper_mvtec15_runtime_accuracy.csv`: one full-test accuracy row
  for every category.
- `results/metadata/paper_mvtec15_run01.json`: experiment-matrix provenance.
- `results/paper_mvtec15_complete/tables/table_5_category_accuracy.csv`:
  generated three-run category aggregates.
- `results/paper_mvtec15_complete/result_coverage.json`: input hashes and
  category coverage.
