# Memory-Bank Storage Precision

The controlled comparison uses the same FP16 feature engine, 10% coreset,
S5 `cuda_tiled_async_transpose` path, executable from commit `09f6523`, and
MAXN_SUPER_ID2. The FP16 bank derives from the FP32 export through a manifest
that records source, output and export-manifest SHA-256 values.

| Bank file | Image AUROC | Pixel AUROC | Serialized payload | Runtime bank memory | Total mean (ms) | Run SD (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 1.000000 | 0.985454531 | 131,487,872 B | 125.396 MB | 503.152 | 12.464 |
| FP16 | 1.000000 | 0.985454581 | 65,743,872 B | 125.396 MB | 523.113 | 20.246 |

FP16 serialization halves the memory-bank payload and preserves bottle
accuracy. It does not reduce runtime bank memory in the current implementation:
`MemoryBank::loadNpy` converts FP16 elements to FP32 before creating CPU and GPU
buffers. The observed latency difference is within the substantial run-to-run
NN variability and provides no evidence of a speedup. A true runtime-memory
claim requires native FP16 bank buffers with FP32 distance accumulation.

The canonical merged input is `results/processed/bank_storage_results.csv`.
The FP16 conversion manifest is
`models/patchcore_memory_bank_fp16.npy.json`; model artifacts remain outside Git.

## Native FP16 GPU bank revision

The `faiss_gpu_fp16` backend added at commit `7040fd1` keeps the retrieval
index in FP16 on the GPU through `GpuIndexFlatConfig::useFloat16`. A matched
clean-build comparison against the FP32 FAISS index used the same FP16 feature
engine, Bottle 10% coreset lineage, image, power mode, 50 warm-up iterations,
200 measured iterations, and three repetitions.

| GPU bank | Image AUROC | Pixel AUROC | GPU index payload (MiB) | NN mean (ms) | E2E mean (ms) | E2E p95 (ms) | Power (W) | Energy (J/image) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 1.000000 | 0.985454528 | 125.396 | **68.610** | **138.888** | **149.674** | 11.594 | 1.610 |
| FP16 | 1.000000 | 0.985454541 | **62.698** | 69.785 | 142.640 | 155.776 | **9.730** | **1.388** |

Native FP16 halves the configured GPU index payload and preserves full-test
Bottle AUROC. It does not improve latency: NN and E2E mean increase by 1.71%
and 2.70%, respectively. Average power decreases by 16.08%, and the combined
effect reduces measured energy per image by 13.81%. This is a resource and
energy trade-off rather than a speedup claim.

Across all 83 test images, FP32/FP16 anomaly scores have MAE 0.002192,
maximum absolute error 0.009306, Pearson correlation 0.999999989, and cosine
similarity 0.999999999. Mean per-image anomaly-map MAE is 0.004000. For the
representative good and broken-large images, nearest-neighbor index agreement
is 100% and 99.8047%, respectively. The executable and accuracy artifacts are
hash-matched at commit `7040fd1`.

Canonical local evidence:

- `results/processed/r3_native_fp16_bank_comparison.json`;
- `results/processed/r3-bottle-fp16-faiss-bank-fp32_run_summary.csv`;
- `results/processed/r3-bottle-fp16-faiss-bank-fp16_run_summary.csv`;
- `results/csv/r3_native_bank_accuracy.csv`.

`device_bank_storage_mb` is the exact configured retrieval-index payload.
The separate process-level `device_memory_delta_mb` remains a before/after
allocator delta and is not reported as peak GPU memory.
