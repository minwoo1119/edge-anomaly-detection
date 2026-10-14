# Revision R1 Exact-GPU Baseline Results

## cuBLAS implementation check

The FP32 cuBLAS exact-L2 backend was evaluated on the Bottle 10% FP32 memory
bank with the FP16 feature engine and S0 data path at commit `920959d`.

- full 83-image test evaluation: image AUROC 1.000000, pixel AUROC 0.985454527;
- three 50-warm-up/200-measured runs: NN 68.932 ms, E2E 140.104 ms,
  E2E p95 151.923 ms, mean power 11.300 W, and energy 1.583 J/image
  (means across runs);
- run-level E2E means: 140.018, 139.662, and 140.631 ms;
- run-level NN means: 69.033, 68.693, and 69.069 ms;
- run-level CUDA memory deltas: 573.543, 529.504, and 553.113 MB
  (mean 552.053 MB). Their spread is retained and the metric is not described
  as peak GPU memory.

Canonical local evidence:

- `results/csv/runtime_accuracy_revision_r1_final.csv`;
- `results/metadata/fp16_cublas_s0_evaluation_final01.json`;
- `results/processed/r1-bottle-fp16-cublas-s0-final_run_summary.csv`.

These performance numbers are **preliminary**. They use a clean cuBLAS-enabled
binary, but the older CPU/custom-CUDA rows were produced by earlier executable
hashes. The apparent 3.65x E2E advantage over the old async-tiled S0 mean and
103.79x advantage over the old CPU S0 mean are useful for experiment planning,
not a controlled publication comparison. R1 requires rerunning all available
backends from the final executable SHA-256
`5a206619d0ab9cf3c99eadfefd57728d706a8b3a3c26659fe1a68f483e82cba4`.

The environment probe found CUDA 12.6 and usable cuBLAS headers/libraries, but
no installed FAISS headers, libraries, or Python GPU API. This establishes only
that FAISS GPU is unavailable in the current environment. An ARM64 source-build
attempt remains required before reporting it as infeasible.
