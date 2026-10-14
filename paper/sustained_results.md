# Sustained Thermal and Deadline Validation

The final FP16/S6 bottle runtime was exercised continuously for 4,000 measured
frames after 50 warm-up iterations in MAXN_SUPER_ID2. The measured interval was
2,036.670 seconds (33.94 minutes). Telemetry was sampled nominally every 100 ms,
yielding 19,184 samples.

| Metric | Result |
| --- | ---: |
| Frames | 4,000 |
| Duration | 2,036.670 s |
| Mean pipeline interval | 509.168 ms |
| Pipeline interval p95 | 517.845 ms |
| Sustained throughput | 1.964 images/s |
| Mean frame completion time | 1,018.309 ms |
| Frame completion p95 | 1,031.660 ms |
| First 5-minute mean interval | 514.355 ms |
| Last complete 5-minute mean interval | 507.128 ms |
| Last-versus-first drift | -1.405% |
| Mean / peak input power | 14.014 / 15.939 W |
| Maximum GPU temperature | 63.906 C |

The preregistered operational degradation criterion was a greater than 5%
increase in mean pipeline interval from the first complete five-minute window
to the last complete five-minute window. The observed change was -1.405%, so
the run shows no sustained performance degradation under this criterion. This
is a performance-and-temperature assessment from `tegrastats`; the platform log
does not expose a hardware throttle flag, so we do not claim direct observation
of an internal throttling signal. A reported 101% peak GPU-utilization sample is
retained as a telemetry artifact and is not interpreted as utilization above
100%.

For the pre-revision paper, 550 ms interval and 1.1 s completion values were
used as reference bounds. The sustained p95 values are below those bounds, but
the run repeatedly processes one image without timestamped fixed-rate arrivals.
It does not record p99, maximum, deadline misses, queue growth, or dropped
frames. The evidence supports thermal and p95 stability, not all-frame deadline
compliance. A fixed-rate simulation is required before making that claim.

Canonical evidence:

- `results/csv/paper_bottle_s6_sustained_benchmark.csv`: 4,000 frame records.
- `results/csv/paper_bottle_s6_sustained_power.csv`: power summary.
- `results/raw/tegrastats/paper-bottle-s6-sustained-30min-01.log`: raw telemetry.
- `results/metadata/paper_bottle_s6_sustained_30min_01.json`: run provenance.
- `results/metadata/paper_bottle_s6_sustained_analysis_01.json`: stability verdict.
- `results/processed/paper_bottle_s6_sustained_windows_01.csv`: five-minute windows.
