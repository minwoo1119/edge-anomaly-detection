# Controlled Nearest-Neighbor Backend Ablation

This experiment fixes the bottle dataset, FP16 feature engine, 10% FP32 memory
bank, S0 data path, Release executable, and MAXN_SUPER_ID2 power mode. Only the
exact nearest-neighbor backend changes. Each configuration uses all 83 bottle
test images for accuracy and three independent runs with 50 warm-up and 200
measured iterations.

| Backend | Image AUROC | Pixel AUROC | NN mean (ms) | Total mean (ms) | Mean p95 (ms) | Power (W) | Energy (J/image) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 1.000000 | 0.985452 | 14434.802 | 14505.975 | 14561.495 | 6.527 | 94.681 |
| Direct CUDA | 1.000000 | 0.985452 | 4314.865 | 4420.934 | 4426.563 | 16.014 | 70.795 |
| Warp CUDA | 1.000000 | 0.985452 | 735.978 | 844.294 | 848.650 | 11.301 | 9.541 |
| Tiled CUDA | 1.000000 | 0.985452 | 508.859 | 613.777 | 631.197 | 16.202 | 9.943 |
| Asynchronous tiled CUDA | 1.000000 | 0.985452 | 404.378 | 510.285 | 516.213 | 14.426 | 7.367 |
| Transpose-assisted asynchronous tiled CUDA | 1.000000 | 0.985452 | 422.996 | 527.955 | 534.227 | 14.715 | 7.772 |

The asynchronous tiled backend is the best S0 configuration. Relative to CPU,
it reduces exact-search latency by 35.70x and end-to-end latency by 28.43x,
while reducing energy per image by 92.2%. Direct CUDA improves total latency by
3.28x over CPU, warp-level reduction adds a 5.24x improvement, tiling adds
1.38x, and asynchronous tiling adds 1.20x.

The transpose-assisted backend is 3.46% slower end to end than asynchronous
tiling under S0. This is an informative negative result: S0 already converts
the device output through the host patch-major path, so the additional
transpose specialization does not provide the GPU-resident advantage it has in
S5. The system-stage experiment separately measures that GPU-resident path.

Canonical evidence:

- `results/processed/nn_backend_results.csv`: 18 hash-joined rows.
- `results/csv/paper_nn_backend_accuracy.csv`: six full-test accuracy rows.
- `results/metadata/paper_nn_backend_bottle_run01.json`: matrix provenance.
- `results/paper_mvtec15_complete/tables/table_4_nn_backend.csv`: generated table.
- `results/paper_mvtec15_complete/figures/figure_6_nn_backend.svg`: generated figure.
