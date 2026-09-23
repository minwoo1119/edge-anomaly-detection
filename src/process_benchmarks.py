#!/usr/bin/env python3
"""Join raw benchmark/power CSVs and write one validated summary row per run."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import json
from collections import defaultdict
from pathlib import Path


TIMING_FIELDS = (
    "preprocess_ms",
    "h2d_ms",
    "trt_ms",
    "d2h_ms",
    "reshape_ms",
    "nn_ms",
    "post_ms",
    "total_ms",
    "fps",
)
IDENTITY_FIELDS = (
    "git_commit",
    "git_dirty",
    "build_type",
    "device",
    "jetpack",
    "cuda",
    "tensorrt",
    "opencv",
    "power_mode",
    "category",
    "model",
    "precision",
    "coreset_ratio",
    "bank_precision",
    "nn_backend",
    "decision_enabled",
    "threshold",
    "threshold_space",
    "threshold_source",
    "config_sha256",
    "engine_sha256",
    "memory_bank_sha256",
    "image_sha256",
)
POWER_FIELDS = (
    "avg_power_w",
    "peak_power_w",
    "energy_per_image_mj",
    "max_temperature_c",
    "raw_tegrastats_path",
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"CSV contains no data rows: {path}")
    return rows


def require_fields(rows: list[dict[str, str]], fields: tuple[str, ...], path: Path) -> None:
    missing = [field for field in fields if field not in rows[0]]
    if missing:
        raise RuntimeError(f"CSV {path} is missing columns: {', '.join(missing)}")


def power_by_run(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows = read_rows(path)
    require_fields(rows, ("run_id", *POWER_FIELDS), path)
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        run_id = row["run_id"]
        if run_id in indexed:
            raise RuntimeError(f"Power CSV contains duplicate run_id: {run_id}")
        indexed[run_id] = row
    return indexed


def summarize_run(
    rows: list[dict[str, str]],
    power: dict[str, str] | None,
    lineage: dict[str, str] | None,
) -> dict[str, object]:
    first = rows[0]
    run_id = first["run_id"]
    for row in rows[1:]:
        changed = [field for field in IDENTITY_FIELDS if row[field] != first[field]]
        if changed:
            raise RuntimeError(
                f"Metadata changed within run_id={run_id}: {', '.join(changed)}"
            )

    result: dict[str, object] = {
        "run_id": run_id,
        "timestamp": first["timestamp"],
        **{field: first[field] for field in IDENTITY_FIELDS},
        "sample_count": len(rows),
        "bank_memory_mb": first["bank_memory_mb"],
        "engine_size_mb": first["engine_size_mb"],
    }
    for field in TIMING_FIELDS:
        try:
            values = [float(row[field]) for row in rows]
        except ValueError as error:
            raise RuntimeError(f"Non-numeric {field} in run_id={run_id}") from error
        prefix = field.removesuffix("_ms")
        result[f"{prefix}_mean"] = statistics.fmean(values)
        result[f"{prefix}_median"] = statistics.median(values)
        result[f"{prefix}_std"] = statistics.pstdev(values)
        result[f"{prefix}_min"] = min(values)
        result[f"{prefix}_max"] = max(values)
        result[f"{prefix}_p95"] = percentile(values, 0.95)
    for field in POWER_FIELDS:
        result[field] = power[field] if power is not None else ""
    for field in ("checkpoint_sha256", "export_manifest_sha256", "engine_manifest_sha256"):
        result[field] = lineage.get(field, "") if lineage is not None else ""
    return result


def lineage_by_run(directory: Path | None) -> dict[str, dict[str, str]]:
    if directory is None:
        return {}
    indexed: dict[str, dict[str, str]] = {}
    for path in sorted(directory.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        run_id = manifest.get("run_id")
        lineage = manifest.get("lineage")
        if not isinstance(run_id, str) or not isinstance(lineage, dict):
            continue
        if run_id in indexed:
            raise RuntimeError(f"Duplicate run_id in experiment manifests: {run_id}")
        indexed[run_id] = {
            key: str(lineage[key])
            for key in ("checkpoint_sha256", "export_manifest_sha256", "engine_manifest_sha256")
            if key in lineage
        }
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--power-csv", type=Path)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.benchmark_csv)
    required = (
        "timestamp",
        "run_id",
        "bank_memory_mb",
        "engine_size_mb",
        *IDENTITY_FIELDS,
        *TIMING_FIELDS,
    )
    require_fields(rows, required, args.benchmark_csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["run_id"]].append(row)

    powers = power_by_run(args.power_csv)
    lineages = lineage_by_run(args.metadata_dir)
    unknown_power_runs = sorted(set(powers) - set(grouped))
    if unknown_power_runs:
        raise RuntimeError(
            "Power CSV contains run_id values absent from benchmark CSV: "
            + ", ".join(unknown_power_runs)
        )
    summaries = [
        summarize_run(run_rows, powers.get(run_id), lineages.get(run_id))
        for run_id, run_rows in sorted(grouped.items())
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"processed_runs={len(summaries)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
