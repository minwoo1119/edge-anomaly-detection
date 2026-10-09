# Controlled System Ablation

The S0–S6 experiment fixes the bottle dataset, FP16 engine, 10% FP32 memory bank,
`cuda_tiled_async_transpose` retrieval kernel, executable, and MAXN_SUPER_ID2.
Each stage uses all 83 test images for accuracy and three 50-warm-up/200-measured
performance runs. All stages preserve image AUROC 1.0 and pixel AUROC
0.985454531.

| Stage | Implemented change | Total mean (ms) | Run SD (ms) | Mean p95 (ms) | Pipeline interval (ms/image) | Power (W) | Energy (J/image) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | Per-frame device buffers, pageable synchronous copies | 548.333 | 10.038 | 619.183 | 548.333 | 12.989 | 7.121 |
| S1 | Persistent device buffers | 547.476 | 16.431 | 616.182 | 547.476 | 12.993 | 7.111 |
| S2 | Persistent pinned host staging | 502.427 | 1.369 | 507.189 | 502.427 | 13.091 | 6.577 |
| S3 | Asynchronous copies on the default stream | 501.316 | 0.438 | 504.444 | 501.316 | 13.018 | 6.526 |
| S4 | Dedicated non-blocking stream | 508.594 | 11.041 | 512.696 | 508.594 | 14.115 | 7.195 |
| S5 | GPU-resident output-to-NN path | 503.152 | 12.464 | 509.192 | 503.152 | 14.826 | 7.462 |
| S6 | Two-frame preprocessing/runtime overlap | 964.625 | 0.716 | 971.700 | 482.350 | 13.322 | 6.426 |

Pinned host staging is the strongest isolated single-frame improvement in this
matrix, reducing mean latency by 8.2% from S1 to S2. S4 and S5 do not improve
the mean under this fixed, already optimized retrieval kernel, which is a useful
negative result. S6 improves steady-state interval by 12.0% relative to S0 and
reduces energy per image by 9.8%, but it increases the time until an individual
frame completes because two frames are in flight. The paper must call 482.350 ms
a throughput interval, not single-frame latency.

The canonical inputs are `results/processed/system_results.csv` and
`results/metadata/paper_system_matrix_run01.json`. Generated Table 6 and Figures
7/8 are under `results/paper_current/` with input hashes in
`result_coverage.json`.
