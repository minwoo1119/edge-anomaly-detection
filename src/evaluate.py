#!/usr/bin/env python3
"""Evaluate a fixed PatchCore checkpoint without exporting a deployment threshold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import anomalib
import numpy as np
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

from export_preprocessing_reference import positive, sha256


def tree_fingerprint(paths: list[Path], root: Path) -> tuple[str, int]:
    files = sorted(path for path in paths if path.is_file())
    if not files:
        raise RuntimeError("Evaluation dataset contains no files")
    digest = hashlib.sha256()
    for path in files:
        entry = {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        digest.update(json.dumps(entry, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def scalar(value: object) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError("Expected a scalar evaluation metric")
        return float(value.detach().cpu().item())
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-batch-size", type=positive, default=32)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--devices", type=positive, default=1)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not args.dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {args.dataset_root}")
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        parser.error("GPU accelerator was requested but CUDA is not available")
    if args.manifest.exists():
        parser.error(f"refusing to overwrite evaluation manifest: {args.manifest}")

    category_root = args.dataset_root / args.category
    test_root = category_root / "test"
    ground_truth_root = category_root / "ground_truth"
    if not test_root.is_dir():
        parser.error(f"MVTec test directory does not exist: {test_root}")
    fingerprint_paths = list(test_root.rglob("*"))
    if ground_truth_root.is_dir():
        fingerprint_paths.extend(ground_truth_root.rglob("*"))
    dataset_sha256, dataset_file_count = tree_fingerprint(fingerprint_paths, category_root)
    checkpoint_sha256 = sha256(args.checkpoint)

    model = Patchcore.load_from_checkpoint(
        str(args.checkpoint),
        weights_only=False,
        map_location="cpu",
    )
    datamodule = MVTecAD(
        root=args.dataset_root,
        category=args.category,
        train_batch_size=1,
        eval_batch_size=args.eval_batch_size,
    )
    engine = Engine(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    results = engine.test(model=model, datamodule=datamodule, verbose=False)
    if len(results) != 1:
        raise RuntimeError(f"Expected one evaluation result set, got {len(results)}")
    metrics = {name: scalar(value) for name, value in results[0].items()}
    for required in ("image_AUROC", "pixel_AUROC"):
        if required not in metrics or not np.isfinite(metrics[required]):
            raise RuntimeError(f"Missing or non-finite required metric: {required}")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": args.category,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_root": str(args.dataset_root.resolve()),
        "evaluation_dataset_sha256": dataset_sha256,
        "evaluation_file_count": dataset_file_count,
        "anomalib_version": str(anomalib.__version__),
        "torch_version": str(torch.__version__),
        "image_auroc": metrics["image_AUROC"],
        "pixel_auroc": metrics["pixel_AUROC"],
        "image_f1_test_derived": metrics.get("image_F1Score", ""),
        "pixel_f1_test_derived": metrics.get("pixel_F1Score", ""),
        "deployment_threshold_exported": False,
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists() or args.output_csv.stat().st_size == 0
    if not write_header:
        with args.output_csv.open(newline="", encoding="utf-8") as stream:
            existing_rows = list(csv.DictReader(stream))
        if existing_rows and list(existing_rows[0]) != list(row):
            raise RuntimeError(f"Evaluation CSV schema mismatch: {args.output_csv}")
        if any(
            existing["category"] == args.category
            and existing["checkpoint_sha256"] == checkpoint_sha256
            and existing["evaluation_dataset_sha256"] == dataset_sha256
            for existing in existing_rows
        ):
            raise RuntimeError("Evaluation CSV already contains this checkpoint/category/dataset")
    with args.output_csv.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    manifest = {
        "schema_version": 1,
        "evaluation_only": True,
        "test_set_used_for_training_or_threshold_tuning": False,
        "deployment_threshold_exported": False,
        "row": row,
        "raw_metrics": metrics,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"image_auroc={row['image_auroc']:.9g}")
    print(f"pixel_auroc={row['pixel_auroc']:.9g}")
    print(f"output_csv={args.output_csv}")
    print(f"manifest={args.manifest}")


if __name__ == "__main__":
    main()
