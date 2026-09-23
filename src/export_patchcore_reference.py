#!/usr/bin/env python3
"""Export raw Anomalib PatchCore tensors for C++ correctness validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anomalib
import numpy as np
import torch
import torchvision
from anomalib.models import Patchcore

from export_preprocessing_reference import positive, preprocess_image, sha256


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export raw embedding, NN, score, and anomaly-map references.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--input-width", type=positive, default=256)
    parser.add_argument("--input-height", type=positive, default=256)
    parser.add_argument("--center-crop-width", type=positive)
    parser.add_argument("--center-crop-height", type=positive)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    crop_width = args.center_crop_width or args.input_width
    crop_height = args.center_crop_height or args.input_height
    if crop_width > args.input_width or crop_height > args.input_height:
        parser.error("center crop must not be larger than the resized image")
    prefix = args.prefix or args.image.stem
    if not prefix or any(character in prefix for character in ("/", "\\")):
        parser.error("prefix must be a non-empty file-name prefix")

    names = (
        "input",
        "embedding",
        "patches",
        "patch_scores",
        "nn_indices",
        "raw_score",
        "anomaly_map",
    )
    output_paths = {name: args.output_dir / f"{prefix}_{name}.npy" for name in names}
    manifest_path = args.output_dir / f"{prefix}_manifest.json"
    existing = [path for path in (*output_paths.values(), manifest_path) if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "reference output already exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )

    device = torch.device(args.device)
    input_tensor = preprocess_image(
        args.image,
        args.input_width,
        args.input_height,
        crop_width,
        crop_height,
    ).to(device)
    model = Patchcore.load_from_checkpoint(
        str(args.checkpoint),
        weights_only=False,
        map_location=device,
    ).to(device)
    model.eval()
    core = model.model

    with torch.inference_mode():
        input_tensor = input_tensor.type(core.memory_bank.dtype)
        features = core.feature_extractor(input_tensor)
        features = {
            layer: core.feature_pooler(features[layer])
            for layer in core.layers
        }
        embedding_nchw = core.generate_embedding(features)
        feature_height, feature_width = embedding_nchw.shape[-2:]
        patches = core.reshape_embedding(embedding_nchw)
        patch_scores, locations = core.nearest_neighbors(patches, n_neighbors=1)
        patch_scores_batch = patch_scores.reshape(1, -1)
        locations_batch = locations.reshape(1, -1)
        raw_score = core.compute_anomaly_score(
            patch_scores_batch,
            locations_batch,
            patches,
        )
        patch_map = patch_scores_batch.reshape(1, 1, feature_height, feature_width)
        anomaly_map = core.anomaly_map_generator(
            patch_map,
            (crop_height, crop_width),
        )

    arrays = {
        "input": input_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
        "embedding": embedding_nchw.detach().cpu().numpy().astype(np.float32, copy=False),
        "patches": patches.detach().cpu().numpy().astype(np.float32, copy=False),
        "patch_scores": patch_scores_batch.reshape(feature_height, feature_width)
        .detach().cpu().numpy().astype(np.float32, copy=False),
        "nn_indices": locations_batch.reshape(feature_height, feature_width)
        .detach().cpu().numpy().astype(np.uint64, copy=False),
        "raw_score": raw_score.detach().cpu().numpy().astype(np.float32, copy=False),
        "anomaly_map": anomaly_map[0, 0].detach().cpu().numpy().astype(np.float32, copy=False),
    }
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise RuntimeError(f"{name} contains NaN or infinity")
        save_array(output_paths[name], array)

    manifest = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_image": str(args.image.resolve()),
        "source_image_sha256": sha256(args.image),
        "device": str(device),
        "input_size_hw": [args.input_height, args.input_width],
        "center_crop_size_hw": [crop_height, crop_width],
        "memory_bank_shape": list(core.memory_bank.shape),
        "memory_bank_dtype": str(core.memory_bank.dtype),
        "num_neighbors": core.num_neighbors,
        "anomalib_version": anomalib.__version__,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "outputs": {
            name: {
                "file": output_paths[name].name,
                "path": str(output_paths[name].resolve()),
                "sha256": sha256(output_paths[name]),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            for name, array in arrays.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")
    for name, array in arrays.items():
        print(f"{name}_shape={array.shape}")


if __name__ == "__main__":
    main()
