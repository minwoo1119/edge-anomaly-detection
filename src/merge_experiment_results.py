#!/usr/bin/env python3
"""Join processed benchmark runs with matching C++/TensorRT accuracy results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ACCURACY_FIELDS = (
    "config_sha256",
    "engine_sha256",
    "memory_bank_sha256",
    "evaluation_dataset_sha256",
    "image_auroc",
    "pixel_auroc",
    "accuracy_source",
    "threshold_tuned_on_test",
)
MATCH_FIELDS = (
    "category",
    "precision",
    "coreset_ratio",
    "bank_precision",
    "nn_backend",
    "optimization_stage",
)


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


def merge_rows(
    benchmark_rows: list[dict[str, str]],
    accuracy_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    accuracy_by_config: dict[str, dict[str, str]] = {}
    for row in accuracy_rows:
        config_hash = row["config_sha256"]
        if config_hash in accuracy_by_config:
            raise RuntimeError(f"Duplicate runtime accuracy for config_sha256={config_hash}")
        if row["accuracy_source"] != "cpp_tensorrt_runtime":
            raise RuntimeError(
                f"Unsupported accuracy_source for config_sha256={config_hash}: "
                f"{row['accuracy_source']}"
            )
        if row["threshold_tuned_on_test"].lower() != "false":
            raise RuntimeError(f"Test-tuned result is not publishable: {config_hash}")
        accuracy_by_config[config_hash] = row

    merged: list[dict[str, str]] = []
    for benchmark in benchmark_rows:
        config_hash = benchmark["config_sha256"]
        accuracy = accuracy_by_config.get(config_hash)
        if accuracy is None:
            raise RuntimeError(
                f"No C++/TensorRT runtime accuracy row for config_sha256={config_hash}"
            )
        mismatches = [
            field for field in MATCH_FIELDS if benchmark[field] != accuracy[field]
        ]
        for field in ("engine_sha256", "memory_bank_sha256"):
            if benchmark[field] != accuracy[field]:
                mismatches.append(field)
        if mismatches:
            raise RuntimeError(
                f"Benchmark/accuracy mismatch for config_sha256={config_hash}: "
                + ", ".join(mismatches)
            )
        merged.append(
            {
                **benchmark,
                "evaluation_dataset_sha256": accuracy["evaluation_dataset_sha256"],
                "image_auroc": accuracy["image_auroc"],
                "pixel_auroc": accuracy["pixel_auroc"],
                "accuracy_source": accuracy["accuracy_source"],
                "threshold_tuned_on_test": accuracy["threshold_tuned_on_test"],
            }
        )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-summary", type=Path, action="append", required=True)
    parser.add_argument("--runtime-accuracy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmark_rows: list[dict[str, str]] = []
    for path in args.benchmark_summary:
        rows = read_rows(path)
        require_fields(
            rows,
            ("config_sha256", "engine_sha256", "memory_bank_sha256", *MATCH_FIELDS),
            path,
        )
        benchmark_rows.extend(rows)
    accuracy_rows = read_rows(args.runtime_accuracy)
    require_fields(accuracy_rows, (*ACCURACY_FIELDS, *MATCH_FIELDS), args.runtime_accuracy)
    merged = merge_rows(benchmark_rows, accuracy_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    print(f"merged_runs={len(merged)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
