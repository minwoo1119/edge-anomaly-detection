# Fixed-Rate Inspection Results

The clean runtime from commit `9879755` was evaluated with the Bottle 10%
FP32 FAISS GPU bank, FP16 feature engine, MAXN_SUPER_ID2, a bounded queue of
eight waiting frames, and a common 1.1 s response deadline. Each short-sweep
condition offered 100 copies of the same input image after one runtime
initialization and 50 warm-up inferences.

| Arrival period (ms) | Completed | Dropped | Failure rate including drops | Response mean (ms) | p99 (ms) | Max (ms) | Max queue | Power (W) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 79 | 21 | 75% | 1037.615 | 1309.113 | 1316.062 | 8 | 11.493 |
| 250 | 100 | 0 | 0% | 188.188 | 217.810 | 222.396 | 0 | 8.165 |
| 500 | 100 | 0 | 0% | 226.391 | 257.785 | 268.755 | 0 | 6.307 |
| 550 | 100 | 0 | 0% | 236.885 | 273.890 | 279.035 | 0 | 6.128 |
| 1000 | 100 | 0 | 0% | 237.015 | 271.448 | 273.111 | 0 | 5.348 |

At 100 ms, offered load exceeds service capacity: queue wait grows by 685.729
ms between the first and last quarters, 21 frames are dropped, and another 54
completed frames miss the deadline. At 250 ms and slower, all short-sweep
frames complete without queue buildup or deadline failure. Idle intervals
allow lower clocks and increase measured service time, so the 250--1000 ms
single-frame responses are longer than the saturated 100 ms service time.

The 550 ms period is an experimental scenario, not a sourced industrial cycle
requirement. Its corresponding 1.1 s deadline is likewise a declared study
bound.

## Predeclared sustained confirmation

Before inspecting the sustained result, the confirmation run is fixed as:

- arrival period: 550 ms;
- offered frames: 3,273 (nominal scheduled span 1,799.6 s);
- deadline: 1,100 ms;
- waiting-queue capacity: 8, drop-newest policy;
- input/config/device state: identical to the short sweep;
- acceptance: zero drops, zero completed deadline misses, response p99 and max
  below 1,100 ms, no positive queue-wait growth exceeding 25 ms, and maximum
  observed temperature below 80 C.

Canonical short-sweep evidence is
`results/processed/r4_fixed_rate_sweep.csv` and the five
`results/metadata/r4-fixed-bottle-*_fixed_rate.json` files.
