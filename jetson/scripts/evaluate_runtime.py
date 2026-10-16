#!/usr/bin/env python3
"""Evaluate one C++/TensorRT configuration on an MVTec AD test split.

This script intentionally evaluates raw scores and anomaly maps without selecting a
threshold. It records the exact runtime artifacts so deployment accuracy is not
silently replaced with the PyTorch checkpoint accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flat_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise RuntimeError(f"Invalid config line {line_number}: {raw_line}")
        key, value = (part.strip() for part in line.split(":", 1))
        if not key or not value or key in values:
            raise RuntimeError(f"Invalid or duplicate config key on line {line_number}")
        values[key] = value
    return values


def resolve_artifact(repository: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def training_seed(export_manifest_path: Path) -> int:
    export = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    lineage = export.get("training_manifest")
    if not isinstance(lineage, dict):
        raise RuntimeError("Export manifest has no training-manifest lineage")
    recorded = Path(str(lineage.get("path", "")))
    candidates = (
        recorded,
        export_manifest_path.parent / "training_manifest.json",
        export_manifest_path.parent.parent / "results" / "training_manifest.json",
    )
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None or sha256(manifest_path) != lineage.get("sha256"):
        raise RuntimeError("Training manifest is unavailable or differs from export lineage")
    training = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = training.get("runtime")
    seed = runtime.get("seed") if isinstance(runtime, dict) else None
    if not isinstance(seed, int):
        raise RuntimeError("Training manifest has no integer runtime seed")
    return seed


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or infinity")
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires at least one positive and one negative")

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    threshold_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), labels.size - 1]
    true_positives = np.cumsum(sorted_labels, dtype=np.int64)[threshold_ends]
    false_positives = (threshold_ends + 1) - true_positives
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) * 0.5))


def dataset_fingerprint(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        resampling = getattr(Image, "Resampling", Image)
        mask = image.convert("L").resize((shape[1], shape[0]), resampling.NEAREST)
        return np.asarray(mask, dtype=np.uint8) > 0


def write_csv_row(path: Path, row: dict[str, object], unique_fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
        if existing and list(existing[0]) != list(row):
            raise RuntimeError(f"CSV schema mismatch; use a new output: {path}")
        if any(all(item[field] == str(row[field]) for field in unique_fields) for item in existing):
            raise RuntimeError(f"CSV already contains this evaluation: {path}")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def images_from_split_manifest(
    manifest_path: Path, split_name: str, category_root: Path, category: str
) -> tuple[list[Path], dict[str, dict[str, object]]]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("category") != category:
        raise RuntimeError("Split manifest schema or category does not match the evaluation")
    records = document.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Split manifest has no records list")
    selected: list[Path] = []
    indexed: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("split") not in {"development", "heldout"}:
            raise RuntimeError("Split manifest contains an invalid record")
        relative_text = record.get("image_path")
        if not isinstance(relative_text, str) or relative_text in indexed:
            raise RuntimeError("Split manifest contains an invalid or duplicate image path")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Split manifest image paths must remain inside the category")
        path = category_root / relative
        if not path.is_file() or sha256(path) != record.get("image_sha256"):
            raise RuntimeError(f"Split image is missing or changed: {path}")
        indexed[relative_text] = record
        if record["split"] == split_name:
            selected.append(path)
    if not selected:
        raise RuntimeError(f"Split manifest contains no {split_name} images")
    return sorted(selected), indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split-name", choices=("development", "heldout"))
    args = parser.parse_args()

    if (args.split_manifest is None) != (args.split_name is None):
        parser.error("--split-manifest and --split-name must be provided together")
    if args.split_manifest is not None and not args.split_manifest.is_file():
        parser.error(f"split manifest does not exist: {args.split_manifest}")

    for path in (args.executable, args.config):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    category_root = args.dataset_root / args.category
    test_root = category_root / "test"
    if not test_root.is_dir():
        parser.error(f"MVTec test split does not exist: {test_root}")
    for output in (args.predictions_csv, args.manifest):
        if output.exists():
            parser.error(f"refusing to overwrite output: {output}")
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        parser.error(f"work directory must be empty: {args.work_dir}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    repository = Path(__file__).resolve().parents[2]
    config = flat_config(args.config)
    required_config = (
        "engine_path",
        "memory_bank_path",
        "engine_manifest_path",
        "export_manifest_path",
        "precision",
        "coreset_ratio",
        "bank_precision",
        "nn_backend",
        "optimization_stage",
        "decision_enabled",
        "threshold",
        "threshold_source",
    )
    missing = [key for key in required_config if key not in config]
    if missing:
        parser.error("config is missing keys: " + ", ".join(missing))
    if config.get("category") != args.category:
        parser.error(
            f"category mismatch: config={config.get('category')}, requested={args.category}"
        )
    engine = resolve_artifact(repository, config["engine_path"])
    memory_bank = resolve_artifact(repository, config["memory_bank_path"])
    engine_manifest = resolve_artifact(repository, config["engine_manifest_path"])
    export_manifest = resolve_artifact(repository, config["export_manifest_path"])
    for path in (engine, memory_bank, engine_manifest, export_manifest):
        if not path.is_file():
            parser.error(f"runtime artifact does not exist: {path}")
    validation_command = [
        sys.executable,
        str(repository / "jetson" / "scripts" / "validate_artifact_chain.py"),
        "--engine",
        str(engine),
        "--memory-bank",
        str(memory_bank),
        "--engine-manifest",
        str(engine_manifest),
        "--export-manifest",
        str(export_manifest),
        "--precision",
        config["precision"],
        "--coreset-ratio",
        config["coreset_ratio"],
        "--category",
        args.category,
        "--decision-enabled",
        config["decision_enabled"],
        "--threshold",
        config["threshold"],
        "--threshold-source",
        config["threshold_source"],
    ]
    threshold_manifest: Path | None = None
    if config["decision_enabled"] == "true":
        if "threshold_manifest_path" not in config:
            parser.error("decision-enabled config has no threshold_manifest_path")
        threshold_manifest = resolve_artifact(repository, config["threshold_manifest_path"])
        if not threshold_manifest.is_file():
            parser.error(f"threshold manifest does not exist: {threshold_manifest}")
        validation_command.extend(["--threshold-manifest", str(threshold_manifest)])
    subprocess.run(
        validation_command,
        check=True,
    )

    split_records: dict[str, dict[str, object]] = {}
    if args.split_manifest is not None:
        images, split_records = images_from_split_manifest(
            args.split_manifest, args.split_name, category_root, args.category
        )
    else:
        images = sorted(
            path
            for path in test_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if not images:
        parser.error(f"test split contains no supported images: {test_root}")

    image_labels: list[bool] = []
    image_scores: list[float] = []
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    prediction_rows: list[dict[str, object]] = []
    fingerprint_paths = list(images)

    for index, image_path in enumerate(images):
        defect_type = image_path.parent.name
        relative_image = str(image_path.relative_to(category_root))
        if split_records:
            record = split_records[relative_image]
            if record.get("defect_type") != defect_type or int(record.get("label", -1)) != int(defect_type != "good"):
                raise RuntimeError(f"Split metadata differs from dataset layout: {relative_image}")
        prefix = f"{index:05d}"
        score_path = args.work_dir / f"{prefix}_score.npy"
        map_path = args.work_dir / f"{prefix}_map.npy"
        command = [
            str(args.executable.resolve()),
            "--config",
            str(args.config.resolve()),
            "--image",
            str(image_path.resolve()),
            "--dump-score",
            str(score_path.resolve()),
            "--dump-map",
            str(map_path.resolve()),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        score = float(np.load(score_path, allow_pickle=False).reshape(-1)[0])
        anomaly_map = np.load(map_path, allow_pickle=False).astype(np.float32, copy=False)
        if anomaly_map.ndim != 2 or not np.isfinite(anomaly_map).all() or not np.isfinite(score):
            raise RuntimeError(f"invalid runtime output for {image_path}")

        is_anomaly = defect_type != "good"
        if is_anomaly:
            mask_path = category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
            if not mask_path.is_file():
                raise RuntimeError(f"ground-truth mask does not exist: {mask_path}")
            if split_records:
                record = split_records[relative_image]
                expected_mask = record.get("mask_path")
                if expected_mask != str(mask_path.relative_to(category_root)) or sha256(mask_path) != record.get("mask_sha256"):
                    raise RuntimeError(f"Split ground-truth mask is missing or changed: {mask_path}")
            mask = load_mask(mask_path, anomaly_map.shape)
            fingerprint_paths.append(mask_path)
        else:
            mask = np.zeros(anomaly_map.shape, dtype=np.bool_)
        image_labels.append(is_anomaly)
        image_scores.append(score)
        pixel_labels.append(mask.reshape(-1))
        pixel_scores.append(anomaly_map.reshape(-1))
        prediction_rows.append(
            {
                "image_path": relative_image,
                "image_sha256": sha256(image_path),
                "defect_type": defect_type,
                "label": int(is_anomaly),
                "raw_score": score,
                "map_path": str(map_path.resolve()),
                "map_sha256": sha256(map_path),
            }
        )

    image_auroc = binary_auc(np.asarray(image_labels), np.asarray(image_scores))
    pixel_auroc = binary_auc(np.concatenate(pixel_labels), np.concatenate(pixel_scores))
    dataset_sha256 = dataset_fingerprint(fingerprint_paths, category_root)
    git_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_dirty = bool(
        subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    row: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "category": args.category,
        "training_seed": training_seed(export_manifest),
        "precision": config["precision"],
        "coreset_ratio": config["coreset_ratio"],
        "bank_precision": config["bank_precision"],
        "nn_backend": config["nn_backend"],
        "optimization_stage": config["optimization_stage"],
        "config_sha256": sha256(args.config),
        "engine_sha256": sha256(engine),
        "memory_bank_sha256": sha256(memory_bank),
        "executable_sha256": sha256(args.executable),
        "evaluation_dataset_sha256": dataset_sha256,
        "evaluation_split": args.split_name or "official_test",
        "split_manifest_sha256": sha256(args.split_manifest) if args.split_manifest else "",
        "image_count": len(images),
        "pixel_count": sum(values.size for values in pixel_labels),
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "threshold_tuned_on_test": False,
        "accuracy_source": "cpp_tensorrt_runtime",
    }
    write_csv_row(
        args.output_csv,
        row,
        ("config_sha256", "evaluation_dataset_sha256"),
    )
    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    manifest = {
        "schema_version": 1,
        "test_set_used_for_threshold_tuning": False,
        "runtime_accuracy": row,
        "artifacts": {
            "config": str(args.config.resolve()),
            "engine": str(engine.resolve()),
            "memory_bank": str(memory_bank.resolve()),
            "predictions_csv": str(args.predictions_csv.resolve()),
            "work_dir": str(args.work_dir.resolve()),
        },
    }
    if args.split_manifest is not None:
        manifest["artifacts"]["split_manifest"] = {
            "path": str(args.split_manifest.resolve()),
            "sha256": sha256(args.split_manifest),
            "split": args.split_name,
        }
    if threshold_manifest is not None:
        manifest["artifacts"]["threshold_manifest"] = {
            "path": str(threshold_manifest.resolve()),
            "sha256": sha256(threshold_manifest),
        }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"image_auroc={image_auroc:.9g}")
    print(f"pixel_auroc={pixel_auroc:.9g}")
    print(f"output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
