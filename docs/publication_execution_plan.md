# Publication Execution Plan

Target: Journal of Real-Time Image Processing (JRTIP).

The implementation optimization phase for the bottle 10% coreset is frozen at
commit `e2a7d12`. Further kernel work requires evidence that it answers a paper
research question. The publication work now follows these gates.

## Gate 1 — Reproducible bottle baseline (complete)

- Python/C++ preprocessing, embedding, NN, score and anomaly-map checks
- FP32 CPU, FP32 CUDA, FP16 CUDA and CUDA optimization variants
- clean-build final S5 run: three independent 50/200 measurements
- runtime accuracy: image AUROC 1.0, pixel AUROC 0.9854545306151844
- latency, memory, power and energy metadata stored with artifact hashes

The canonical current result is
`results/processed/e9-bottle-fp16-async-transpose-s5-clean01_results.csv`.

## Gate 2 — Coreset trade-off (next)

Train independent bottle artifacts at 1%, 2.5%, 5%, 10% and 20% using seed 42.
Do not derive smaller banks by truncating the 10% bank. For every ratio:

- preserve checkpoint, training/export manifests and bundle checksum;
- build a Jetson-local FP16 TensorRT engine;
- evaluate all bottle test images once without test-set threshold tuning;
- run 50 warm-up + 200 measurements, three independent runs;
- record bank entries/size, image/pixel AUROC, stage latency, power and energy.

Exit criterion: five hash-joined configurations in `coreset_results.csv`, with
Table 3 and Figure 5 generated without `--allow-incomplete`.

## Gate 3 — Precision and bank storage

On the fixed 10% bottle artifacts and final NN backend, compare FP32, FP16 and
INT8 feature engines. INT8 calibration uses train-normal images only. Separately
compare FP32 and FP16 bank storage with FP32 distance accumulation.

Exit criterion: artifact-matched runtime accuracy and three-run measurements for
each reported configuration. If INT8 is unstable or unsupported, report the
negative result and narrow the paper claim rather than silently omitting it.

After building the INT8 engine, the controlled S5 precision plan is
`configs/paper_precision_matrix.json`. Run it with a dedicated accuracy CSV so
earlier development evaluations cannot collide with final publication records.

## Gate 4 — Retrieval and system ablation

Use one fixed artifact chain. Compare CPU brute force, the direct CUDA baseline,
warp, tiled, asynchronous tiled, and transpose-assisted asynchronous tiled
search. Report the direct-NCHW regression as an informative negative result.
System stages must reflect actual implementation changes; stage labels alone are
not evidence of an ablation.

Exit criterion: same accuracy artifacts and fixed experimental conditions, with
three runs per configuration and a clear contribution table.

## Gate 5 — Multi-category generality

Train and evaluate all 15 MVTec AD categories with the selected precision,
coreset ratio and NN backend. At minimum, retain full accuracy for all categories
and performance/power measurements for representative objects and textures. The
preferred submission dataset contains three performance runs for every category.

Exit criterion: category-wise AUROC table covers all 15 categories and the paper
does not generalize from bottle alone.

## Gate 6 — Real-time claim and robustness

Define the target inspection interval before using the term real-time. Measure
p95 latency, a sustained run, temperature and throttling behavior. Current bottle
throughput is approximately 2 FPS, so the supported claim must be tied to an
inspection cycle of at least the observed p95 latency unless later experiments
show otherwise.

## Gate 7 — Manuscript package

Generate all tables/figures from merged CSVs, freeze the result manifest hashes,
write the 12-page double-column manuscript, and perform an internal claim/evidence
audit. The paper must state dataset splits, no-test-tuning policy, hardware/power
mode, software versions, run protocol, uncertainty and limitations.

Submission readiness means Gates 1–6 are complete and
`scripts/generate_paper_results.py` succeeds without `--allow-incomplete`.
