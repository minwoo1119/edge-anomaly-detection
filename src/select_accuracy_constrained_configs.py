#!/usr/bin/env python3
"""Lock category-specific deployment choices from development results only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


IDENTITY = ("category", "coreset_ratio", "bank_precision", "nn_backend")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(
    rows: list[dict[str, str]], baseline_ratio: float, image_tolerance: float,
    pixel_tolerance: float, expected_seeds: set[int]
) -> list[dict[str, object]]:
    if not rows:
        raise RuntimeError("Development result table is empty")
    if any(row.get("evaluation_split") != "development" for row in rows):
        raise RuntimeError("Selection input must contain development rows only")
    baselines: dict[tuple[str, int], tuple[float, float]] = {}
    for row in rows:
        seed = int(row["training_seed"])
        if float(row["coreset_ratio"]) == baseline_ratio:
            key = (row["category"], seed)
            candidate = (float(row["image_auroc"]), float(row["pixel_auroc"]))
            previous = baselines.get(key)
            baselines[key] = candidate if previous is None else (
                max(previous[0], candidate[0]), max(previous[1], candidate[1])
            )

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in IDENTITY)].append(row)
    categories = sorted({row["category"] for row in rows})
    selected: list[dict[str, object]] = []
    for category in categories:
        feasible: list[dict[str, object]] = []
        for key, candidates in grouped.items():
            if key[0] != category:
                continue
            seeds = {int(row["training_seed"]) for row in candidates}
            if seeds != expected_seeds:
                continue
            valid = True
            for row in candidates:
                seed = int(row["training_seed"])
                baseline = baselines.get((category, seed))
                if baseline is None or float(row["image_auroc"]) < baseline[0] - image_tolerance or float(row["pixel_auroc"]) < baseline[1] - pixel_tolerance:
                    valid = False
                    break
            if not valid:
                continue
            total = statistics.fmean(float(row["total_mean"]) for row in candidates)
            energy = statistics.fmean(float(row["energy_per_image_mj"]) for row in candidates)
            memory = statistics.fmean(float(row["device_bank_storage_mb"]) for row in candidates)
            feasible.append({
                "category": category,
                "coreset_ratio": float(key[1]),
                "bank_precision": key[2],
                "nn_backend": key[3],
                "training_seeds": sorted(seeds),
                "total_mean_ms": total,
                "energy_per_image_mj": energy,
                "device_bank_storage_mb": memory,
                "worst_image_auroc": min(float(row["image_auroc"]) for row in candidates),
                "worst_pixel_auroc": min(float(row["pixel_auroc"]) for row in candidates),
                "config_sha256_by_seed": {row["training_seed"]: row["config_sha256"] for row in candidates},
                "memory_bank_sha256_by_seed": {row["training_seed"]: row["memory_bank_sha256"] for row in candidates},
            })
        if not feasible:
            raise RuntimeError(f"No feasible development configuration for category={category}")
        feasible.sort(key=lambda item: (
            float(item["total_mean_ms"]), float(item["energy_per_image_mj"]),
            float(item["device_bank_storage_mb"]), float(item["coreset_ratio"]),
            str(item["nn_backend"]), str(item["bank_precision"]),
        ))
        choice = feasible[0]
        choice["feasible_candidates"] = len(feasible)
        selected.append(choice)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ratio", type=float, default=0.10)
    parser.add_argument("--image-auroc-tolerance", type=float, default=0.005)
    parser.add_argument("--pixel-auroc-tolerance", type=float, default=0.005)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 123, 2026))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    with args.development_results.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    choices = select(
        rows, args.baseline_ratio, args.image_auroc_tolerance,
        args.pixel_auroc_tolerance, set(args.seeds)
    )
    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_split": "development",
        "heldout_results_read": False,
        "source": {"path": str(args.development_results.resolve()), "sha256": sha256(args.development_results)},
        "rules": {
            "baseline_ratio": args.baseline_ratio,
            "image_auroc_tolerance": args.image_auroc_tolerance,
            "pixel_auroc_tolerance": args.pixel_auroc_tolerance,
            "required_training_seeds": args.seeds,
            "feasibility": "every seed meets both baseline-relative AUROC constraints",
            "objective_order": ["total_mean_ms", "energy_per_image_mj", "device_bank_storage_mb", "coreset_ratio", "nn_backend", "bank_precision"],
        },
        "choices": choices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"selected_categories={len(choices)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
