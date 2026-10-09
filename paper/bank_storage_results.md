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
