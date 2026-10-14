# Publication Execution Plan

Target: Journal of Real-Time Image Processing (JRTIP).

> **Revision status (2026-10-14):** the package generated on 2026-10-13 is a
> pre-revision draft, not a submission-ready package. External feedback requires
> strong exact-GPU baselines, validation-based multi-seed bank selection,
> native FP16 device storage, and fixed-rate deadline evidence. The binding
> continuation plan is `docs/revision_experiment_plan.md`.

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

## Gate 2 — Coreset trade-off (complete)

Train independent bottle artifacts at 1%, 2.5%, 5%, 10% and 20% using seed 42.
Do not derive smaller banks by truncating the 10% bank. For every ratio:

- preserve checkpoint, training/export manifests and bundle checksum;
- build a Jetson-local FP16 TensorRT engine;
- evaluate all bottle test images once without test-set threshold tuning;
- run 50 warm-up + 200 measurements, three independent runs;
- record bank entries/size, image/pixel AUROC, stage latency, power and energy.
- require the exact runtime executable SHA-256 to match between accuracy and
  benchmark rows, in addition to config, engine and memory-bank hashes.

Exit criterion: five hash-joined configurations in `coreset_results.csv`, with
Table 3 and Figure 5 generated without `--allow-incomplete`.

The five independently trained bottle variants completed at commit `09f6523`.
Ratios 1%, 2.5%, 5%, 10%, and 20% contain 2,140, 5,350, 10,700, 21,401,
and 42,803 bank entries. All retain image AUROC 1.0; pixel AUROC ranges only
from 0.984961 to 0.985500. Mean latency increases from 72.093 ms at 1% to
992.520 ms at 20%, while serialized bank size increases from 12.539 MB to
250.799 MB. The canonical joined input is
`results/processed/coreset_results.csv`.

## Gate 3 — Precision and bank storage (complete)

On the fixed 10% bottle artifacts and final NN backend, compare FP32, FP16 and
INT8 feature engines. INT8 calibration uses train-normal images only. Separately
compare FP32 and FP16 bank storage with FP32 distance accumulation.

Exit criterion: artifact-matched runtime accuracy and three-run measurements for
each reported configuration. If INT8 is unstable or unsupported, report the
negative result and narrow the paper claim rather than silently omitting it.

The controlled precision matrix completed at commit `6985796`. FP32 and FP16
both reached image AUROC 1.0 and pixel AUROC 0.985461/0.985455. Entropy-calibrated
INT8 fell to image AUROC 0.688889 and pixel AUROC 0.761489, while end-to-end
latency improved only from 550.245 ms (FP16) to 523.076 ms because exact NN
search remained dominant. The INT8 path is therefore retained as a negative
result and is not a deployment candidate. FP16 bank serialization preserved
image/pixel AUROC 1.0/0.985455 and halved the artifact payload, but the loader
restores it to FP32; measured runtime bank memory therefore remained 125.396 MB
for both formats. This is reported as a storage-only result.

After building the INT8 engine, the controlled S5 precision plan is
`configs/paper_precision_matrix.json`. Run it with a dedicated accuracy CSV so
earlier development evaluations cannot collide with final publication records.

## Gate 4 — Retrieval and system ablation (complete)

Use one fixed artifact chain. Compare CPU brute force, the direct CUDA baseline,
warp, tiled, asynchronous tiled, and transpose-assisted asynchronous tiled
search. Report the direct-NCHW regression as an informative negative result.
System stages must reflect actual implementation changes; stage labels alone are
not evidence of an ablation.

Exit criterion: same accuracy artifacts and fixed experimental conditions, with
three runs per configuration and a clear contribution table.

The controlled S0–S6 system matrix completed at commit `09f6523`. Every stage
preserved image/pixel AUROC 1.0/0.985455. S2 pinned staging produced the largest
single-frame reduction in the pre-S5 path (547.476 ms at S1 to 502.427 ms at
S2). S6 reached a 482.350 ms steady-state interval and 6.426 J/image. Its
964.625 ms single-frame completion time is reported separately.

The controlled S0 retrieval matrix completed at commit `df3df81` with the same
FP16 engine, 10% FP32 bank, executable, and power mode across all rows. CPU,
direct CUDA, warp, tiled, asynchronous tiled, and transpose-assisted
asynchronous tiled backends each have one full-test accuracy evaluation and
three 50/200 performance runs. Asynchronous tiled search reduced NN latency
from 14434.802 to 404.378 ms and total latency from 14505.975 to 510.285 ms
relative to CPU, corresponding to 35.70x and 28.43x speedups. The S0 transpose
variant measured 527.955 ms and is reported as a 3.46% regression. Canonical
evidence is `results/processed/nn_backend_results.csv`, with Table 4 and Figure
6 under `results/paper_mvtec15_complete/`.

## Gate 5 — Multi-category generality (complete)

Train and evaluate all 15 MVTec AD categories with the selected precision,
coreset ratio and NN backend. At minimum, retain full accuracy for all categories
and performance/power measurements for representative objects and textures. The
preferred submission dataset contains three performance runs for every category.

Exit criterion: category-wise AUROC table covers all 15 categories and the paper
does not generalize from bottle alone.

The selected FP16/S5 configuration was evaluated on all 15 MVTec AD categories
at commit `09f6523`. Each category has one full-test accuracy evaluation and
three independent 50-warm-up/200-measured Jetson runs. The resulting macro image
and pixel AUROC are 0.983314 and 0.979761. The canonical joined input is
`results/processed/multi_category_results.csv`, with the matrix manifest at
`results/metadata/paper_mvtec15_run01.json` and generated Table 5 under
`results/paper_mvtec15_complete/`.

## Gate 6 — Real-time claim and robustness (complete)

Define the target inspection interval before using the term real-time. Measure
p95 latency, a sustained run, temperature and throttling behavior. Current bottle
throughput is approximately 2 FPS, so the supported claim must be tied to an
inspection cycle of at least the observed p95 latency unless later experiments
show otherwise.

The final FP16/S6 bottle runtime completed 4,000 measured frames over 2,036.670
seconds. Mean/p95 pipeline interval was 509.168/517.845 ms, mean/p95 completion
time was 1018.309/1031.660 ms, and maximum GPU temperature was 63.906 C. The
last complete five-minute window was 1.405% faster than the first, passing the
predeclared 5% degradation bound. The p95 interval and completion values are
below reference bounds of 550 ms and 1.1 s. Because this run did not impose
timestamped fixed-rate arrivals or record p99/max, misses, backlog, and drops,
it supports thermal and p95 stability only. Because `tegrastats` did not provide
a hardware throttle flag, the conclusion is also limited to observed
temperature and performance stability. Canonical analysis is
`results/metadata/paper_bottle_s6_sustained_analysis_01.json`.

## Gate 7 — Manuscript package

Generate all tables/figures from merged CSVs, freeze the result manifest hashes,
write the 12-page double-column manuscript, and perform an internal claim/evidence
audit. The paper must state dataset splits, no-test-tuning policy, hardware/power
mode, software versions, run protocol, uncertainty and limitations.

Submission readiness means Gates 1–6 are complete and
`scripts/generate_paper_results.py` succeeds without `--allow-incomplete`.

Gate 7 pre-revision technical production was complete as of 2026-10-13. The JRTIP manuscript
uses the official two-column `svjour3` class, compiles to eight pages, contains
six vector figures, and passes `scripts/audit_jrtip_submission.py`: the
231-word abstract, six keywords, embedded PDF fonts, citations, bibliography,
mandatory declarations, page limit, and flat source archive all validate. The
generated package is `results/submission_jrtip_20261013/`. Minwoo Lee and
corresponding author HyeongTae Ahn, both affiliated with the Department of
Computer Engineering at Kumoh National Institute of Technology, are populated.
Submission is now also blocked on revision work packages R1--R6 in
`docs/revision_experiment_plan.md`, in addition to the blank corresponding-email,
funding, competing-interest, contribution, and cover-letter confirmation fields.

## Gate 8 — Strong exact-GPU baselines (in progress)

Add matched cuBLAS exact L2 and, where feasible on Jetson ARM64, FAISS GPU
`GpuIndexFlatL2`. Separate GPU-resident operator time from S0 transfer-inclusive
NN and end-to-end time. The cuBLAS runtime backend and environment probe are
implemented. Its clean, three-run Bottle S0 measurement and full-test accuracy
evaluation are complete at commit `920959d`; matched reruns of the other
backends, profiling, and the FAISS build attempt remain pending.

## Gates 9--13 — Required revision evidence (pending)

- Gate 9: Nsight Systems/Compute explanation of backend and S-stage behavior.
- Gate 10: multi-category, multi-ratio, multi-seed validation-based selection.
- Gate 11: native FP16 GPU bank with FP32 accumulation and memory verification.
- Gate 12: locked selection followed by held-out test evaluation.
- Gate 13: fixed-rate arrivals with p99/max, miss rate, backlog, drops, and
  sustained thermal/power evidence.
