#!/usr/bin/env python3
"""Export deterministic train-normal tensors for TensorRT INT8 calibration."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import anomalib
import numpy as np
import torch
import torchvision

from export_preprocessing_reference import positive, preprocess_image, sha256


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", type=positive, default=100)
    parser.add_argument("--input-width", type=positive, default=256)
    parser.add_argument("--input-height", type=positive, default=256)
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    resolved_input = args.input_dir.resolve()
    if resolved_input.name.lower() != "good" or resolved_input.parent.name.lower() != "train":
        parser.error("INT8 calibration source must be an MVTec-style train/good directory")
    manifest_path = args.output_dir / "calibration_manifest.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing to write into non-empty output directory: {args.output_dir}")

    images = sorted(
        path for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(images) < args.images:
        parser.error(f"need {args.images} train-normal images, found {len(images)}")
    selected = images[: args.images]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for index, image_path in enumerate(selected):
        tensor = preprocess_image(
            image_path,
            args.input_width,
            args.input_height,
            args.input_width,
            args.input_height,
        )
        array = np.ascontiguousarray(tensor.cpu().numpy(), dtype=np.float32)
        expected_shape = (1, 3, args.input_height, args.input_width)
        if array.shape != expected_shape or not np.isfinite(array).all():
            raise RuntimeError(
                f"Invalid calibration tensor for {image_path}: shape={array.shape}"
            )
        output_path = args.output_dir / f"{index:05d}.npy"
        np.save(output_path, array, allow_pickle=False)
        entries.append(
            {
                "index": index,
                "source_path": str(image_path.resolve()),
                "source_sha256": sha256(image_path),
                "tensor_path": str(output_path.resolve()),
                "tensor_sha256": sha256(output_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "train_normal_only",
        "selection": "lexicographic_first_n",
        "source_directory": str(resolved_input),
        "image_count": len(entries),
        "tensor_shape": [1, 3, args.input_height, args.input_width],
        "tensor_dtype": "float32",
        "anomalib_version": str(anomalib.__version__),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "entries": entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"calibration_tensors={len(entries)}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
