#!/usr/bin/env python3
"""Export the Anomalib PatchCore input tensor used as the C++ reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anomalib
import numpy as np
import torch
import torchvision
from anomalib.models import Patchcore
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2.functional import to_dtype


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def preprocess_image(
    image_path: Path,
    input_width: int,
    input_height: int,
    crop_width: int,
    crop_height: int,
) -> torch.Tensor:
    """Run the same torchvision transform configured by Anomalib PatchCore."""
    image = decode_image(str(image_path), mode=ImageReadMode.RGB)
    image = to_dtype(image, torch.float32, scale=True)
    preprocessor = Patchcore.configure_pre_processor(
        image_size=(input_height, input_width),
        center_crop_size=(crop_height, crop_width),
    )
    transform = getattr(preprocessor, "transform", None)
    if transform is None:
        raise RuntimeError("Anomalib PreProcessor does not expose its configured transform")
    with torch.inference_mode():
        return transform(image).unsqueeze(0).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the exact PatchCore preprocessing output for one image.",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-width", type=positive, default=256)
    parser.add_argument("--input-height", type=positive, default=256)
    parser.add_argument("--center-crop-width", type=positive)
    parser.add_argument("--center-crop-height", type=positive)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    crop_width = args.center_crop_width or args.input_width
    crop_height = args.center_crop_height or args.input_height

    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if args.output.suffix.lower() != ".npy":
        parser.error("output must use the .npy extension")
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    existing_outputs = [path for path in (args.output, manifest_path) if path.exists()]
    if existing_outputs and not args.overwrite:
        parser.error(
            "output already exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    if crop_width > args.input_width or crop_height > args.input_height:
        parser.error("center crop must not be larger than the resized image")

    tensor = preprocess_image(
        args.image,
        args.input_width,
        args.input_height,
        crop_width,
        crop_height,
    )
    array = tensor.cpu().numpy().astype(np.float32, copy=False)
    if array.shape != (1, 3, crop_height, crop_width):
        raise RuntimeError(f"unexpected preprocessing output shape: {array.shape}")
    if not np.isfinite(array).all():
        raise RuntimeError("preprocessing output contains NaN or infinity")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, array, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "source_image": str(args.image.resolve()),
        "source_image_sha256": sha256(args.image),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "input_size_hw": [args.input_height, args.input_width],
        "center_crop_size_hw": [crop_height, crop_width],
        "resize_antialias": True,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "anomalib_version": anomalib.__version__,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"shape={array.shape}")
    print(f"dtype={array.dtype}")
    print(f"min={float(array.min()):.10g}")
    print(f"max={float(array.max()):.10g}")
    print(f"output={args.output}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
