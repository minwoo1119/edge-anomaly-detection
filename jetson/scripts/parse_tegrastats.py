#!/usr/bin/env python3
"""Summarize tegrastats power samples and join them to a benchmark run."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path


def power_samples(path: Path) -> list[float]:
    samples: list[float] = []
    pattern = re.compile(r"VDD_IN\s+(\d+)mW(?:/(\d+)mW)?")
    rail_pattern = re.compile(r"VDD_(?:GPU_SOC|CPU_CV|VIN_SYS_5V0)\s+(\d+)mW")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            samples.append(float(match.group(1)) / 1000.0)
            continue
        rails = [float(value) for value in rail_pattern.findall(line)]
        if rails:
            samples.append(sum(rails) / 1000.0)
    if not samples:
        raise RuntimeError("No supported power fields were found in the tegrastats log")
    return samples


def benchmark_metadata(path: Path) -> tuple[dict[str, str], float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Benchmark CSV has no data rows")
    latest_key = (rows[-1]["timestamp"], rows[-1]["run_id"])
    run_rows = [
        row for row in rows if (row["timestamp"], row["run_id"]) == latest_key
    ]
    mean_total_ms = sum(float(row["total_ms"]) for row in run_rows) / len(run_rows)
    return run_rows[-1], mean_total_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tegrastats", type=Path, required=True)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = power_samples(args.tegrastats)
    metadata, mean_total_ms = benchmark_metadata(args.benchmark_csv)
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
        "avg_power_w": average_power,
        "peak_power_w": max(samples),
        "energy_per_image_mj": average_power * mean_total_ms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
