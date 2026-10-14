# Revision R1 Exact-GPU Baseline Results

## cuBLAS implementation check

The FP32 cuBLAS exact-L2 backend was evaluated on the Bottle 10% FP32 memory
bank with the FP16 feature engine and S0 data path at commit `154d66a`.

- full 83-image test evaluation: image AUROC 1.000000, pixel AUROC 0.985454527;
- three 50-warm-up/200-measured runs: NN 68.720 ms, E2E 139.761 ms,
  mean power 11.222 W, and energy 1.568 J/image (means across runs);
- run-level E2E means: 139.923, 138.425, and 140.936 ms;
- run-level NN means: 68.720, 68.142, and 69.298 ms.

Canonical local evidence:

- `results/csv/runtime_accuracy_revision.csv`;
- `results/metadata/fp16_cublas_s0_evaluation_run02.json`;
- `results/processed/r1-bottle-fp16-cublas-s0_run_summary.csv`.

These performance numbers are **preliminary**. They use a clean cuBLAS-enabled
binary, but the older CPU/custom-CUDA rows were produced by earlier executable
hashes. The apparent 3.65x E2E advantage over the old async-tiled S0 mean and
103.79x advantage over the old CPU S0 mean are useful for experiment planning,
not a controlled publication comparison. R1 requires rerunning all available
backends from one executable hash. The runs also predate the
`device_memory_delta_mb` field and must be repeated for the final memory table.

The environment probe found CUDA 12.6 and usable cuBLAS headers/libraries, but
no installed FAISS headers, libraries, or Python GPU API. This establishes only
that FAISS GPU is unavailable in the current environment. An ARM64 source-build
attempt remains required before reporting it as infeasible.
