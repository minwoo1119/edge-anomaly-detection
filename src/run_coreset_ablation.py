#!/usr/bin/env python3
"""Build independently traced PatchCore artifacts for the JRTIP coreset ablation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from decimal import Decimal
from pathlib import Path


DEFAULT_RATIOS = (0.01, 0.025, 0.05, 0.10, 0.20)


def ratio_tag(ratio: float) -> str:
    decimal = Decimal(str(ratio)).normalize()
    digits = format(decimal, "f").replace(".", "")
    return f"c{digits}"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--calibration-images", type=int, default=100)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {args.dataset_root}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error(f"output-root must be empty or absent: {args.output_root}")
    if args.calibration_images <= 0 or args.devices <= 0:
        parser.error("calibration-images and devices must be positive")
    if len(set(args.ratios)) != len(args.ratios):
        parser.error("coreset ratios must be unique")
    if any(ratio <= 0.0 or ratio > 1.0 for ratio in args.ratios):
        parser.error("each coreset ratio must be in (0, 1]")

    source = Path(__file__).resolve().parent
    python = sys.executable
    for ratio in args.ratios:
        tag = ratio_tag(ratio)
        variant_root = args.output_root / args.category / tag
        checkpoint = variant_root / "models" / f"patchcore_{args.category}_{tag}.ckpt"
        training_manifest = variant_root / "results" / "metadata" / "training.json"
        deployment_root = variant_root / "deployment"
        run(
            [
                python,
                str(source / "train.py"),
                "--dataset-root",
                str(args.dataset_root),
                "--category",
                args.category,
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(training_manifest),
                "--coreset-ratio",
                f"{ratio:.12g}",
                "--accelerator",
                args.accelerator,
                "--devices",
                str(args.devices),
                "--seed",
                str(args.seed),
            ]
        )
        run(
            [
                python,
                str(source / "run_colab_pipeline.py"),
                "--checkpoint",
                str(checkpoint),
                "--dataset-root",
                str(args.dataset_root),
                "--category",
                args.category,
                "--output-root",
                str(deployment_root),
                "--calibration-images",
                str(args.calibration_images),
                "--device",
                "cuda" if args.accelerator == "gpu" else "cpu",
            ]
        )
    print(f"coreset_variants={len(args.ratios)}")
    print(f"output_root={args.output_root}")


if __name__ == "__main__":
    main()
