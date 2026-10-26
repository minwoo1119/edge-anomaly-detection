#!/usr/bin/env python3
"""Build independently traced PatchCore artifacts for the JRTIP coreset ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_variant(variant_root: Path, category: str, ratio: float, seed: int) -> bool:
    marker = variant_root / "COMPLETE.json"
    if not marker.is_file():
        return False
    try:
        document = json.loads(marker.read_text(encoding="utf-8"))
        recorded_ratio = Decimal(str(document.get("coreset_ratio")))
    except (json.JSONDecodeError, ValueError):
        return False
    bundle = variant_root / "deployment" / f"{category}_artifacts.zip"
    checksum = bundle.with_suffix(bundle.suffix + ".sha256")
    return (
        document.get("category") == category
        and recorded_ratio == Decimal(str(ratio))
        and document.get("seed") == seed
        and bundle.is_file()
        and checksum.is_file()
        and document.get("bundle_sha256") == sha256(bundle)
    )


def reusable_training_artifacts(
    checkpoint: Path,
    training_manifest: Path,
    category: str,
    ratio: float,
    seed: int,
) -> bool:
    if not checkpoint.is_file() or not training_manifest.is_file():
        return False
    try:
        document = json.loads(training_manifest.read_text(encoding="utf-8"))
        recorded_ratio = Decimal(str(document["model"]["coreset_ratio"]))
        recorded_hash = document["checkpoint"]["sha256"]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return (
        document.get("dataset", {}).get("category") == category
        and recorded_ratio == Decimal(str(ratio))
        and document.get("runtime", {}).get("seed") == seed
        and recorded_hash == sha256(checkpoint)
    )


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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {args.dataset_root}")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
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
        if completed_variant(variant_root, args.category, ratio, args.seed):
            print(f"skip_complete={variant_root}", flush=True)
            continue
        checkpoint = variant_root / "models" / f"patchcore_{args.category}_{tag}.ckpt"
        training_manifest = variant_root / "results" / "metadata" / "training.json"
        deployment_root = variant_root / "deployment"
        reuse_training = False
        if variant_root.exists():
            if not args.resume:
                parser.error(f"variant output already exists: {variant_root}")
            reuse_training = reusable_training_artifacts(
                checkpoint, training_manifest, args.category, ratio, args.seed
            )
            if reuse_training:
                print(f"resume_from_training={variant_root}", flush=True)
                shutil.rmtree(deployment_root, ignore_errors=True)
                (variant_root / "COMPLETE.json").unlink(missing_ok=True)
            else:
                print(f"rebuild_incomplete={variant_root}", flush=True)
                shutil.rmtree(variant_root)
        if not reuse_training:
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
                "--training-manifest",
                str(training_manifest),
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
        bundle = deployment_root / f"{args.category}_artifacts.zip"
        checksum = bundle.with_suffix(bundle.suffix + ".sha256")
        if not bundle.is_file() or not checksum.is_file():
            raise RuntimeError(
                f"Deployment pipeline did not create the expected bundle and checksum: {bundle}"
            )
        marker = {
            "schema_version": 1,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "category": args.category,
            "coreset_ratio": ratio,
            "seed": args.seed,
            "bundle": str(bundle.resolve()),
            "bundle_sha256": sha256(bundle),
        }
        (variant_root / "COMPLETE.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
    print(f"coreset_variants={len(args.ratios)}")
    print(f"output_root={args.output_root}")


if __name__ == "__main__":
    main()
