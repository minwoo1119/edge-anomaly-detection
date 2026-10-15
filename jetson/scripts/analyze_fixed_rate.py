#!/usr/bin/env python3
"""Summarize a timestamped fixed-rate inspection trace."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


def parse_telemetry(lines: list[str]) -> tuple[list[float], list[float], list[float], list[float]]:
    power: list[float] = []
    temperature: list[float] = []
    gpu: list[float] = []
    cpu: list[float] = []
    for line in lines:
        power_match = re.search(r"VDD_IN\s+(\d+)mW", line)
        if not power_match:
            continue
        power.append(float(power_match.group(1)) / 1000.0)
        temperatures = [float(value) for value in re.findall(r"@([0-9.]+)C", line)]
        if temperatures:
            temperature.append(max(temperatures))
        gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if gpu_match:
            gpu.append(float(gpu_match.group(1)))
        cpu_values = [float(value) for value in re.findall(r"(\d+)%@", line)]
        if cpu_values:
            cpu.append(sum(cpu_values) / len(cpu_values))
    return power, temperature, gpu, cpu


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        raise RuntimeError("Fixed-rate trace is empty")
    run_ids = {row["run_id"] for row in rows}
    periods = {float(row["arrival_period_ms"]) for row in rows}
    deadlines = {float(row["deadline_ms"]) for row in rows}
    if len(run_ids) != 1 or len(periods) != 1 or len(deadlines) != 1:
        raise RuntimeError("Fixed-rate identity changed within the trace")
    completed = [row for row in rows if row["status"] == "completed"]
    dropped = [row for row in rows if row["status"] == "dropped"]
    if len(completed) + len(dropped) != len(rows) or not completed:
        raise RuntimeError("Trace has invalid statuses or no completed frames")
    response = [float(row["response_ms"]) for row in completed]
    service = [float(row["service_ms"]) for row in completed]
    queue_wait = [float(row["queue_wait_ms"]) for row in completed]
    intervals = [float(row["output_interval_ms"]) for row in completed if float(row["output_interval_ms"]) > 0]
    completed_misses = sum(row["deadline_miss"].lower() == "true" for row in completed)
    deadline_failures = completed_misses + len(dropped)
    wall_ms = max(float(row["completion_ms"]) for row in completed)
    quarter = max(1, len(completed) // 4)
    first_wait = sum(queue_wait[:quarter]) / quarter
    last_wait = sum(queue_wait[-quarter:]) / quarter
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": next(iter(run_ids)),
        "arrival_period_ms": next(iter(periods)),
        "deadline_ms": next(iter(deadlines)),
        "offered_frames": len(rows),
        "completed_frames": len(completed),
        "dropped_frames": len(dropped),
        "drop_rate": len(dropped) / len(rows),
        "completed_deadline_misses": completed_misses,
        "deadline_failure_count_including_drops": deadline_failures,
        "deadline_failure_rate_including_drops": deadline_failures / len(rows),
        "response_mean_ms": sum(response) / len(response),
        "response_p95_ms": percentile(response, 0.95),
        "response_p99_ms": percentile(response, 0.99),
        "response_max_ms": max(response),
        "service_mean_ms": sum(service) / len(service),
        "service_p99_ms": percentile(service, 0.99),
        "queue_wait_mean_ms": sum(queue_wait) / len(queue_wait),
        "queue_wait_p99_ms": percentile(queue_wait, 0.99),
        "queue_wait_max_ms": max(queue_wait),
        "maximum_queue_depth_at_arrival": max(int(row["queue_depth_at_arrival"]) for row in rows),
        "first_quarter_queue_wait_mean_ms": first_wait,
        "last_quarter_queue_wait_mean_ms": last_wait,
        "queue_wait_growth_ms": last_wait - first_wait,
        "output_interval_mean_ms": sum(intervals) / len(intervals) if intervals else None,
        "output_interval_p99_ms": percentile(intervals, 0.99) if intervals else None,
        "wall_duration_seconds": wall_ms / 1000.0,
        "sustained_throughput_fps": len(completed) / (wall_ms / 1000.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--tegrastats", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    with args.trace.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = summarize(rows)
    result["trace"] = str(args.trace.resolve())
    if args.tegrastats is not None:
        power, temperature, gpu, cpu = parse_telemetry(
            args.tegrastats.read_text(encoding="utf-8", errors="replace").splitlines()
        )
        if not power:
            raise RuntimeError("Tegrastats log contains no power samples")
        result["telemetry_samples"] = len(power)
        result["average_power_w"] = sum(power) / len(power)
        result["peak_power_w"] = max(power)
        result["maximum_temperature_c"] = max(temperature) if temperature else None
        result["average_gpu_utilization_pct"] = sum(gpu) / len(gpu) if gpu else None
        result["average_cpu_utilization_pct"] = sum(cpu) / len(cpu) if cpu else None
        result["tegrastats"] = str(args.tegrastats.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
