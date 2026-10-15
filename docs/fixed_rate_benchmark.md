# Fixed-Rate Inspection Benchmark

`run_fixed_rate_experiment.sh` injects frames according to a monotonic clock,
uses a bounded FIFO queue, and records one row for every offered frame. The
runtime is initialized and warmed once before arrivals begin. A producer thread
generates arrivals with `sleep_until`; the inference thread consumes the same
image repeatedly so the experiment isolates service capacity and queueing from
storage I/O variability.

```bash
bash jetson/scripts/run_fixed_rate_experiment.sh \
  configs/jetson_fp16_faiss_gpu_s0.yaml \
  datasets/MVTecAD/bottle/test/good/000.png \
  fixed-bottle-p250-r01 \
  250 200 1100 8
```

The positional values after the run ID are arrival period in milliseconds,
offered frames, response deadline in milliseconds, and waiting-queue capacity.
The current overload policy drops the newest arrival when the waiting queue is
full. Queue capacity excludes the frame currently in service.

The raw trace records scheduled and actual arrival, service start, completion,
queue wait, service time, response time, output interval, queue depth, drop,
deadline status, score, and artifact hashes. The analyzer reports completion
p95/p99/max, miss rate including drops, sustained throughput, maximum backlog,
queue-wait growth, power, utilization, and temperature.

Use unique run IDs; outputs are never overwritten. The initial experimental
sweep is 100, 250, 500, 550, and 1000 ms. The 550 ms arrival period and 1.1 s
deadline are retained as declared experimental scenarios. They must not be
described as requirements from a specific industrial line until an external
source or application requirement establishes that interpretation.

For publication, run a short sweep first, select the boundary conditions
without inspecting held-out accuracy, then execute a predeclared sustained
duration. Report dropped frames separately from completed deadline misses and
include both in the overall service-failure rate.
