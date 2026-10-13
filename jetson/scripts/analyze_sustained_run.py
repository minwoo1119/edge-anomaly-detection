#!/usr/bin/env python3
"""Analyze time-window stability for a sustained Jetson benchmark run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark_values(path: Path, run_id: str, field: str) -> list[float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["run_id"] == run_id]
    if not rows:
        raise RuntimeError(f"No benchmark rows for run_id={run_id}")
    return [float(row[field]) for row in rows]


def benchmark_intervals(path: Path, run_id: str) -> list[float]:
    return benchmark_values(path, run_id, "pipeline_interval_ms")


def telemetry(path: Path) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        power = re.search(r"VDD_IN\s+(\d+)mW", line)
        gpu_temp = re.search(r"gpu@([0-9.]+)C", line)
        gpu_util = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if power:
            samples.append(
                {
                    "power_w": float(power.group(1)) / 1000.0,
                    "gpu_temperature_c": float(gpu_temp.group(1)) if gpu_temp else math.nan,
                    "gpu_utilization_pct": float(gpu_util.group(1)) if gpu_util else math.nan,
                }
            )
    if not samples:
        raise RuntimeError("No VDD_IN telemetry samples found")
    return samples


def latency_windows(intervals_ms: list[float], window_seconds: float) -> list[dict[str, float | int]]:
    windows: list[list[float]] = []
    elapsed = 0.0
    for interval in intervals_ms:
        index = int(elapsed // window_seconds)
        while len(windows) <= index:
            windows.append([])
        windows[index].append(interval)
        elapsed += interval / 1000.0
    result: list[dict[str, float | int]] = []
    for index, values in enumerate(windows):
        if values:
            mean = sum(values) / len(values)
            result.append(
                {
                    "window_index": index,
                    "start_seconds": index * window_seconds,
                    "end_seconds": min((index + 1) * window_seconds, elapsed),
                    "frames": len(values),
                    "interval_mean_ms": mean,
                    "interval_p95_ms": percentile(values, 0.95),
                    "fps": 1000.0 / mean,
                }
            )
    return result


def add_telemetry_windows(
    windows: list[dict[str, float | int]],
    samples: list[dict[str, float]],
    interval_ms: int,
    window_seconds: float,
) -> None:
    per_window = max(1, round(window_seconds * 1000 / interval_ms))
    for window in windows:
        start = int(window["window_index"]) * per_window
        values = samples[start : start + per_window]
        if not values:
            continue
        power = [sample["power_w"] for sample in values]
        temperatures = [sample["gpu_temperature_c"] for sample in values if math.isfinite(sample["gpu_temperature_c"])]
        utilization = [sample["gpu_utilization_pct"] for sample in values if math.isfinite(sample["gpu_utilization_pct"])]
        window["power_mean_w"] = sum(power) / len(power)
        window["power_peak_w"] = max(power)
        window["gpu_temperature_mean_c"] = sum(temperatures) / len(temperatures) if temperatures else ""
        window["gpu_temperature_max_c"] = max(temperatures) if temperatures else ""
        window["gpu_utilization_mean_pct"] = sum(utilization) / len(utilization) if utilization else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--tegrastats", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-windows-csv", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=300.0)
    parser.add_argument("--tegrastats-interval-ms", type=int, default=100)
    parser.add_argument("--minimum-duration-seconds", type=float, default=1800.0)
    parser.add_argument("--degradation-limit-pct", type=float, default=5.0)
    args = parser.parse_args()

    for output in (args.output_json, args.output_windows_csv):
        if output.exists():
            parser.error(f"refusing to overwrite output: {output}")
    intervals = benchmark_intervals(args.benchmark_csv, args.run_id)
    completion_times = benchmark_values(args.benchmark_csv, args.run_id, "total_ms")
    if len(completion_times) != len(intervals):
        raise RuntimeError("Completion and interval sample counts differ")
    duration = sum(intervals) / 1000.0
    if duration < args.minimum_duration_seconds:
        raise RuntimeError(
            f"Sustained run is too short: {duration:.3f}s < {args.minimum_duration_seconds:.3f}s"
        )
    samples = telemetry(args.tegrastats)
    windows = latency_windows(intervals, args.window_seconds)
    add_telemetry_windows(windows, samples, args.tegrastats_interval_ms, args.window_seconds)
    complete = [
        window for window in windows
        if float(window["end_seconds"]) - float(window["start_seconds"]) >= args.window_seconds * 0.99
    ]
    if len(complete) < 2:
        raise RuntimeError("Sustained run needs at least two complete analysis windows")
    first = complete[0]
    last = complete[-1]
    drift = (
        (float(last["interval_mean_ms"]) - float(first["interval_mean_ms"]))
        / float(first["interval_mean_ms"])
        * 100.0
    )
    temperatures = [sample["gpu_temperature_c"] for sample in samples if math.isfinite(sample["gpu_temperature_c"])]
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "frames": len(intervals),
        "duration_seconds": duration,
        "window_seconds": args.window_seconds,
        "complete_windows": len(complete),
        "overall_interval_mean_ms": sum(intervals) / len(intervals),
        "overall_interval_p95_ms": percentile(intervals, 0.95),
        "overall_fps": 1000.0 / (sum(intervals) / len(intervals)),
        "overall_completion_mean_ms": sum(completion_times) / len(completion_times),
        "overall_completion_p95_ms": percentile(completion_times, 0.95),
        "first_window_interval_mean_ms": first["interval_mean_ms"],
        "last_window_interval_mean_ms": last["interval_mean_ms"],
        "last_vs_first_interval_drift_pct": drift,
        "degradation_limit_pct": args.degradation_limit_pct,
        "sustained_latency_degradation_over_limit": drift > args.degradation_limit_pct,
        "throttling_assessment_scope": "performance_drift_and_tegrastats_temperature; no_hardware_throttle_flag",
        "telemetry_samples": len(samples),
        "gpu_temperature_max_c": max(temperatures) if temperatures else None,
        "benchmark_csv": str(args.benchmark_csv.resolve()),
        "tegrastats_log": str(args.tegrastats.resolve()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_windows_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for window in windows for key in window))
    with args.output_windows_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(windows)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
