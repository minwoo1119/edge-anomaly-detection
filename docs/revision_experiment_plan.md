# Revision Experiment Plan: Accuracy-Constrained Edge PatchCore

This document supersedes the earlier submission-ready interpretation of the
Bottle-only optimization study. The existing measurements remain valid within
their recorded configurations, but the revised paper requires stronger GPU
baselines, validation-based selection, native FP16 memory, and fixed-rate
service evidence before submission.

## Research scope

The work studies joint optimization of the PatchCore memory bank, exact GPU
retrieval, data movement, and execution pipeline on Jetson Orin Nano. It does
not claim a new anomaly-detection model. The intended contribution is a
reproducible systems method for choosing a deployment configuration subject to
accuracy, energy, and memory constraints:

\[
\min_{r,b,p} T(r,b,p)\quad\text{s.t.}\quad
A(r,b,p)\geq A_{target},\ E(r,b,p)\leq E_{budget},\
M(r,b,p)\leq M_{budget}.
\]

Here `r` is the coreset ratio, `b` the exact retrieval backend, and `p` the
bank precision. The formulation is an evaluation and selection rule; novelty
must come from measured behavior against strong baselines.

## Research questions

- **RQ1:** Why does a faster TensorRT backbone fail to produce proportional
  end-to-end latency reduction?
- **RQ2:** How do retrieval backend, bank dtype, memory movement, and pipeline
  organization affect kernel time, completion latency, interval, memory, and
  energy?
- **RQ3:** How much resource use can category-specific, accuracy-constrained
  memory-bank selection remove relative to a fixed 10% bank?
- **RQ4:** Under fixed-rate arrivals, which configurations meet response
  deadlines without growing queues, drops, or thermal degradation?

## Mandatory measurement contract

Every comparison must join records by executable, config, engine, export
manifest, memory bank, dataset split, and device-state hashes. Device, power
mode, clocks, warm-up count, measured repetitions, input, and background load
must match. Report mean, median, p95, p99, maximum, power, J/image, and peak
device memory. `device_memory_delta_mb` records the CUDA free-memory difference
between pre-runtime initialization and the post-warm-up measured process; it is
an allocated-memory delta, not a peak-memory claim. Keep these boundaries separate:

1. GPU-resident retrieval operator time;
2. query and result transfer time;
3. single-frame end-to-end completion latency;
4. steady-state output interval under overlap.

Numerical checks include nearest-index agreement, distance error, anomaly-score
error, anomaly-map error, image AUROC, and pixel AUROC. Add AUPRO when its
implementation and integration limit are fixed before testing.

## Work packages and exit criteria

### R1 — Exact GPU retrieval baselines (in progress)

- Compare CPU, direct CUDA, warp, tiled, async tiled, cuBLAS exact L2, and
  FAISS `GpuIndexFlatL2` when available under S0.
- The `cublas` backend uses FP32 SGEMM, FP32 norm/reduction, and pedantic math.
  Its S0 timing includes query H2D and result D2H, matching the existing S0
  boundary. Nsight reports provide the GPU-resident operator breakdown.
- Run `python3 jetson/scripts/probe_gpu_baselines.py --output <new.json>` on the
  target. Absence means unavailable in that environment, not impossible to
  build on ARM64.
- Do not compare cuBLAS S0 with S5/S6 custom CUDA results as though their data
  paths were identical.

Exit: full-test accuracy plus three matched 50/200 runs for every available
backend; documented FAISS source-build attempt if its prebuilt GPU API is absent.

### R2 — GPU bottleneck profiling

Profile representative small, medium, and large banks for direct, warp, tiled,
async, transpose, and cuBLAS paths. Use
`jetson/scripts/profile_nn_backend.sh` to preserve Nsight Systems, Nsight
Compute, config/image hashes, and commit. Extract kernel duration, achieved
occupancy, DRAM throughput, launch count, transfers, and synchronization. Use
these measurements to explain S4 and transpose regressions instead of inferring
causes from end-to-end time alone.

Exit: a profile table and plots indexed by bank rows and embedding dimension,
with at least three bank sizes and the same query count.

### R3 — Accuracy-constrained coreset selection

Run 1%, 2.5%, 5%, 10%, and 20% banks over seeds 42, 123, and 2026. Prefer all
15 MVTec AD categories; the minimum publishable reduction must be declared
before running and include both object and texture categories.

MVTec AD has no official validation split. For category-specific automatic
selection, create one deterministic, defect-type-stratified development/test
split and freeze its file manifest before experiments. Use development only to
select `(ratio, backend, dtype)`; evaluate the selected configuration once on
held-out test. Keep the conventional fixed-policy, full official test result as
a separate benchmark track. Never label a split-selected result as the official
full-test score.

Compare fixed 10%, one global selected ratio, and category-specific selection.
Report per-category and macro accuracy, latency, p99, J/image, and memory with
seed variability.

### R4 — Native FP16 memory bank

Preserve FP16 values after loading instead of expanding them permanently to
FP32. Store FP16 on the GPU and accumulate/reduce squared L2 in FP32. Record
allocated device bytes from the implementation and corroborate them with a
device-memory measurement. Compare against the same learned FP32 bank using
index agreement and numerical errors as well as system and accuracy metrics.

Exit: confirmed device-memory reduction and a complete accuracy/performance
table. No speedup claim is made before measurement.

### R5 — Selection and locked held-out evaluation

Freeze accuracy targets, energy/memory budgets, tie-breaking, and the selection
algorithm before reading held-out results. A valid selected configuration must
meet the development accuracy constraint for every required metric. Apply the
locked choice once to held-out test and retain failures.

### R6 — Fixed-rate industrial simulation

Feed timestamped frames at predeclared periods (initial sweep: 100, 250, 500,
550, and 1000 ms; remove impossible points only with a recorded reason). Record
arrival, service start, completion, output, and drop timestamps for every frame.
Report throughput, completion p95/p99/max, deadline-miss rate, queue length,
drops, temperature, power, and time-window drift. The 550 ms arrival and 1.1 s
response bounds remain experimental scenarios until their industrial origin or
design rationale is documented.

Exit: a sustained duration fixed before execution, zero ambiguity between
completion latency and output interval, and explicit miss-rate/max evidence.

### R7 — Optional model comparison

Compare EfficientAD only after R1–R6. Match device, input size, dataset split,
power mode, and end-to-end measurement boundary, and present model choice as a
trade-off rather than a universal PatchCore speed claim.

## Existing evidence that remains usable

- The 28.43x figure is a fixed 10% Bottle, FP16 engine, FP32 bank, S0 comparison
  of async tiled CUDA against CPU.
- The 482.350 ms S6 value is a pipeline interval; its paired completion latency
  is 964.625 ms.
- The 72.093 ms value belongs to the separate Bottle 1% coreset experiment and
  cannot be combined with the S6 configuration.
- Bottle 1% image AUROC 1.0 is single-category, single-seed evidence.
- S4 and S0 transpose regressions are retained as negative results.
- The sustained trace establishes observed p95 and thermal stability. It does
  not establish a fixed-arrival miss rate or prove that every frame met a
  deadline.

## Planned paper outputs

1. exact-retrieval table: CPU/custom CUDA/cuBLAS/FAISS;
2. bank size versus occupancy, bandwidth, and NN time;
3. category-wise AUROC-latency-energy Pareto plots with seed variation;
4. FP32/FP16 bank accuracy, memory, latency, and energy table;
5. S0–S6 completion latency beside output interval;
6. fixed-rate p99, misses, backlog, temperature, and power;
7. hash-addressed configuration matrix for every reported point.

The expanded title
“Accuracy-Constrained GPU Retrieval and Memory Optimization for Edge Industrial
Anomaly Detection” should be adopted only after R1–R6 support the expanded
claims. Until then, retain the current bottleneck-focused title.
