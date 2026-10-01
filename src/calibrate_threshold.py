#!/usr/bin/env python3
"""Calibrate a raw PatchCore deployment threshold using train-normal images only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import anomalib
import numpy as np
import torch
from anomalib.models import Patchcore

from export_preprocessing_reference import positive, preprocess_image, sha256
from train import dataset_fingerprint


def raw_score(core: torch.nn.Module, input_tensor: torch.Tensor) -> float:
    """Return the pre-normalization PatchCore image score used by the C++ runtime."""
    features = core.feature_extractor(input_tensor)
    features = {layer: core.feature_pooler(features[layer]) for layer in core.layers}
    embedding = core.generate_embedding(features)
    patches = core.reshape_embedding(embedding)
    patch_scores, locations = core.nearest_neighbors(patches, n_neighbors=1)
    score = core.compute_anomaly_score(
        patch_scores.reshape(1, -1),
        locations.reshape(1, -1),
        patches,
    )
    if score.numel() != 1:
        raise RuntimeError("PatchCore returned a non-scalar image score")
    value = float(score.detach().cpu().item())
    if not np.isfinite(value):
        raise RuntimeError("PatchCore returned a non-finite image score")
    return value


def select_threshold(scores: np.ndarray, method: str, quantile: float) -> float:
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be a non-empty finite vector")
    if method == "max":
        return float(np.max(scores))
    if method == "quantile":
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")
        return float(np.quantile(scores, quantile, method="linear"))
    raise ValueError(f"unsupported threshold method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("quantile", "max"), default="quantile")
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--input-width", type=positive, default=256)
    parser.add_argument("--input-height", type=positive, default=256)
    args = parser.parse_args()

    for path, name in (
        (args.checkpoint, "checkpoint"),
        (args.training_manifest, "training manifest"),
    ):
        if not path.is_file():
            parser.error(f"{name} does not exist: {path}")
    if args.output.exists():
        parser.error(f"refusing to overwrite threshold manifest: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if not 0.0 < args.quantile <= 1.0:
        parser.error("quantile must be in (0, 1]")

    train_normal = args.dataset_root / args.category / "train" / "good"
    if not train_normal.is_dir():
        parser.error(f"train-normal directory does not exist: {train_normal}")
    training = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    training_dataset = training.get("dataset")
    training_checkpoint = training.get("checkpoint")
    training_model = training.get("model")
    if not all(
        isinstance(value, dict)
        for value in (training_dataset, training_checkpoint, training_model)
    ):
        raise RuntimeError("Training manifest is missing dataset/checkpoint/model metadata")
    if training_dataset.get("category") != args.category:
        raise RuntimeError("Training manifest category does not match --category")
    checkpoint_hash = sha256(args.checkpoint)
    if training_checkpoint.get("sha256") != checkpoint_hash:
        raise RuntimeError("Checkpoint SHA-256 does not match the training manifest")
    dataset_hash, dataset_entries = dataset_fingerprint(train_normal)
    if training_dataset.get("train_normal_fingerprint") != dataset_hash:
        raise RuntimeError("Train-normal dataset differs from the training manifest")
    expected_size = training_model.get("input_size_hw")
    if expected_size != [args.input_height, args.input_width]:
        raise RuntimeError(
            "Calibration input size differs from training: "
            f"{[args.input_height, args.input_width]} != {expected_size}"
        )

    device = torch.device(args.device)
    model = Patchcore.load_from_checkpoint(
        str(args.checkpoint), weights_only=False, map_location=device
    ).to(device)
    model.eval()
    core = model.model
    scores: list[float] = []
    score_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for entry in dataset_entries:
            image_path = train_normal / str(entry["path"])
            input_tensor = preprocess_image(
                image_path,
                args.input_width,
                args.input_height,
                args.input_width,
                args.input_height,
            ).to(device=device, dtype=core.memory_bank.dtype)
            score = raw_score(core, input_tensor)
            scores.append(score)
            score_rows.append(
                {
                    "path": str(entry["path"]),
                    "image_sha256": entry["sha256"],
                    "raw_score": score,
                }
            )

    score_array = np.asarray(scores, dtype=np.float64)
    threshold = select_threshold(score_array, args.method, args.quantile)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "category": args.category,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_hash,
        },
        "training_manifest": {
            "path": str(args.training_manifest.resolve()),
            "sha256": sha256(args.training_manifest),
        },
        "source": "train_normal",
        "test_set_used_for_threshold_tuning": False,
        "threshold_space": "raw",
        "method": args.method,
        "quantile": args.quantile if args.method == "quantile" else None,
        "threshold": threshold,
        "sample_count": len(scores),
        "train_normal_fingerprint": dataset_hash,
        "score_summary": {
            "minimum": float(np.min(score_array)),
            "mean": float(np.mean(score_array)),
            "median": float(np.median(score_array)),
            "maximum": float(np.max(score_array)),
        },
        "scores": score_rows,
        "runtime": {
            "device": str(device),
            "anomalib_version": str(anomalib.__version__),
            "torch_version": str(torch.__version__),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"threshold={threshold:.9g}")
    print("threshold_source=train_normal")
    print(f"manifest={args.output}")


if __name__ == "__main__":
    main()
