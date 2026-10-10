#!/usr/bin/env python3
"""Run the complete non-Jetson artifact pipeline with one Colab command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def first_image(directory: Path) -> Path:
    images = image_files(directory)
    if not images:
        raise RuntimeError(f"No image found in {directory}")
    return images[0]


def first_anomalous_image(test_directory: Path) -> Path:
    defect_directories = sorted(
        path for path in test_directory.iterdir()
        if path.is_dir() and path.name.lower() != "good"
    )
    if not defect_directories:
        raise RuntimeError(f"No anomalous test directory found in {test_directory}")
    for directory in defect_directories:
        try:
            return first_image(directory)
        except RuntimeError:
            continue
    raise RuntimeError(f"No anomalous test image found in {test_directory}")


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--export-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not args.training_manifest.is_file():
        parser.error(f"training manifest does not exist: {args.training_manifest}")
    training = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    training_dataset = training.get("dataset")
    if not isinstance(training_dataset, dict):
        parser.error("training manifest has no dataset metadata")
    if training_dataset.get("category") != args.category:
        parser.error(
            "training manifest category does not match --category: "
            f"{training_dataset.get('category')} != {args.category}"
        )
    if args.calibration_images <= 0:
        parser.error("calibration-images must be positive")
    category_root = args.dataset_root / args.category
    train_normal = category_root / "train" / "good"
    test_normal = category_root / "test" / "good"
    test_root = category_root / "test"
    for directory in (train_normal, test_normal, test_root):
        if not directory.is_dir():
            parser.error(f"required MVTec directory does not exist: {directory}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error(f"output-root must be empty or absent: {args.output_root}")

    validation_image = first_image(train_normal)
    normal_reference = first_image(test_normal)
    anomalous_reference = first_anomalous_image(test_root)
    available_calibration_images = len(image_files(train_normal))
    calibration_images = min(args.calibration_images, available_calibration_images)
    if calibration_images < args.calibration_images:
        print(
            "calibration_images_adjusted="
            f"{calibration_images} requested={args.calibration_images} "
            f"available={available_calibration_images}",
            flush=True,
        )
    models = args.output_root / "models"
    metadata = args.output_root / "results" / "metadata"
    csv_directory = args.output_root / "results" / "csv"
    reference = args.output_root / "results" / "reference"
    calibration = models / "calibration" / args.category
    bundle = args.output_root / f"{args.category}_artifacts.zip"
    onnx_path = models / "patchcore_feature_extractor.onnx"
    memory_bank = models / "patchcore_memory_bank.npy"
    export_manifest = models / "patchcore_export_manifest.json"
    evaluation_manifest = metadata / f"evaluate_{args.category}.json"
    threshold_manifest = metadata / f"threshold_{args.category}.json"
    accuracy_csv = csv_directory / "accuracy.csv"
    python = sys.executable
    source = Path(__file__).resolve().parent

    run(
        [
            python,
            str(source / "evaluate.py"),
            "--checkpoint",
            str(args.checkpoint),
            "--dataset-root",
            str(args.dataset_root),
            "--category",
            args.category,
            "--output-csv",
            str(accuracy_csv),
            "--manifest",
            str(evaluation_manifest),
            "--accelerator",
            "gpu" if args.device == "cuda" else "cpu",
        ]
    )
    run(
        [
            python,
            str(source / "export_onnx.py"),
            "--checkpoint",
            str(args.checkpoint),
            "--training-manifest",
            str(args.training_manifest),
            "--onnx",
            str(onnx_path),
            "--memory-bank",
            str(memory_bank),
            "--manifest",
            str(export_manifest),
            "--validation-image",
            str(validation_image),
            "--device",
            args.export_device,
            "--exporter",
            "torchscript",
            "--atol",
            "3e-4",
            "--max-mean-error",
            "1e-5",
        ]
    )
    run(
        [
            python,
            str(source / "calibrate_threshold.py"),
            "--checkpoint",
            str(args.checkpoint),
            "--training-manifest",
            str(args.training_manifest),
            "--dataset-root",
            str(args.dataset_root),
            "--category",
            args.category,
            "--output",
            str(threshold_manifest),
            "--device",
            args.device,
        ]
    )
    run(
        [
            python,
            str(source / "export_int8_calibration.py"),
            "--input-dir",
            str(train_normal),
            "--output-dir",
            str(calibration),
            "--images",
            str(calibration_images),
        ]
    )
    for image, prefix in (
        (normal_reference, f"{args.category}_good"),
        (anomalous_reference, f"{args.category}_{anomalous_reference.parent.name}"),
    ):
        run(
            [
                python,
                str(source / "export_patchcore_reference.py"),
                "--checkpoint",
                str(args.checkpoint),
                "--image",
                str(image),
                "--output-dir",
                str(reference),
                "--prefix",
                prefix,
                "--device",
                args.device,
            ]
        )
    run(
        [
            python,
            str(source / "package_colab_artifacts.py"),
            "--export-manifest",
            str(export_manifest),
            "--training-manifest",
            str(args.training_manifest),
            "--threshold-manifest",
            str(threshold_manifest),
            "--calibration-dir",
            str(calibration),
            "--reference-dir",
            str(reference),
            "--evaluation-manifest",
            str(evaluation_manifest),
            "--accuracy-csv",
            str(accuracy_csv),
            "--output",
            str(bundle),
        ]
    )
    print(f"pipeline_complete=true")
    print(f"bundle={bundle}")
    print(f"checksum={bundle.with_suffix(bundle.suffix + '.sha256')}")


if __name__ == "__main__":
    main()
