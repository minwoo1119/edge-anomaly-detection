#!/usr/bin/env python3
"""Summarize tegrastats power samples and join them to a benchmark run."""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path


def tegrastats_samples(
    path: Path,
) -> tuple[list[float], list[float], list[float], list[float]]:
    return parse_tegrastats_lines(
        path.read_text(encoding="utf-8", errors="replace").splitlines()
    )


def parse_tegrastats_lines(
    lines: Iterable[str],
) -> tuple[list[float], list[float], list[float], list[float]]:
    samples: list[float] = []
    temperatures: list[float] = []
    gpu_utilization: list[float] = []
    cpu_utilization: list[float] = []
    pattern = re.compile(r"VDD_IN\s+(\d+)mW(?:/(\d+)mW)?")
    rail_pattern = re.compile(r"VDD_(?:GPU_SOC|CPU_CV|VIN_SYS_5V0)\s+(\d+)mW")
    for line in lines:
        temperatures.extend(float(value) for value in re.findall(r"@([0-9.]+)C", line))
        gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if gpu_match:
            gpu_utilization.append(float(gpu_match.group(1)))
        cpu_match = re.search(r"CPU\s*\[([^]]+)]", line)
        if cpu_match:
            core_values: list[float] = []
            for core in cpu_match.group(1).split(","):
                match = re.search(r"(\d+(?:\.\d+)?)%@", core)
                core_values.append(float(match.group(1)) if match else 0.0)
            if core_values:
                cpu_utilization.append(sum(core_values) / len(core_values))
        match = pattern.search(line)
        if match:
            samples.append(float(match.group(1)) / 1000.0)
            continue
        rails = [float(value) for value in rail_pattern.findall(line)]
        if rails:
            samples.append(sum(rails) / 1000.0)
    if not samples:
        raise RuntimeError("No supported power fields were found in the tegrastats log")
    return samples, temperatures, gpu_utilization, cpu_utilization


def benchmark_metadata(path: Path, run_id: str) -> tuple[dict[str, str], float, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Benchmark CSV has no data rows")
    run_rows = [row for row in rows if row["run_id"] == run_id]
    if not run_rows:
        raise RuntimeError(f"Benchmark CSV has no rows for run_id={run_id}")
    interval_field = (
        "pipeline_interval_ms"
        if "pipeline_interval_ms" in run_rows[0]
        else "total_ms"
    )
    mean_interval_ms = sum(float(row[interval_field]) for row in run_rows) / len(run_rows)
    mean_fps = sum(float(row["fps"]) for row in run_rows) / len(run_rows)
    return run_rows[-1], mean_interval_ms, mean_fps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tegrastats", type=Path, required=True)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples, temperatures, gpu_utilization, cpu_utilization = tegrastats_samples(
        args.tegrastats
    )
    metadata, mean_interval_ms, mean_fps = benchmark_metadata(
        args.benchmark_csv, args.run_id
    )
    average_power = sum(samples) / len(samples)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": metadata["git_commit"],
        "device": metadata["device"],
        "power_mode": metadata["power_mode"],
        "category": metadata["category"],
        "precision": metadata["precision"],
        "nn_backend": metadata["nn_backend"],
        "run_id": metadata["run_id"],
        "samples": len(samples),
        "measurement_scope": "measured_iterations_only",
        "tegrastats_interval_ms": 100,
        "avg_power_w": average_power,
        "peak_power_w": max(samples),
        "energy_per_image_mj": average_power * mean_interval_ms,
        "fps_per_w": mean_fps / average_power,
        "avg_gpu_utilization_pct": (
            sum(gpu_utilization) / len(gpu_utilization) if gpu_utilization else ""
        ),
        "peak_gpu_utilization_pct": max(gpu_utilization) if gpu_utilization else "",
        "avg_cpu_utilization_pct": (
            sum(cpu_utilization) / len(cpu_utilization) if cpu_utilization else ""
        ),
        "peak_cpu_utilization_pct": max(cpu_utilization) if cpu_utilization else "",
        "max_temperature_c": max(temperatures) if temperatures else "",
        "raw_tegrastats_path": str(args.tegrastats.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    if not write_header:
        with args.output.open(newline="", encoding="utf-8") as stream:
            existing_header = next(csv.reader(stream), [])
        if existing_header != list(result):
            raise RuntimeError(
                f"Power CSV schema mismatch; use a new output file: {args.output}"
            )
    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
