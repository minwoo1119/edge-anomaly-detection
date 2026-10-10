#!/usr/bin/env python3
"""Prepare all 15 MVTec AD bundles and generate the Jetson experiment plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from decimal import Decimal
from pathlib import Path


MVTEC_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
)
EXPECTED_RATIO = Decimal("0.1")


def identify(bundle: Path) -> tuple[str, Decimal]:
    with zipfile.ZipFile(bundle) as archive:
        training = json.loads(archive.read("results/training_manifest.json"))
    dataset = training.get("dataset")
    model = training.get("model")
    if not isinstance(dataset, dict) or not isinstance(model, dict):
        raise RuntimeError(f"Invalid training manifest in {bundle}")
    category = dataset.get("category")
    ratio = model.get("coreset_ratio")
    if not isinstance(category, str) or not isinstance(ratio, (int, float)):
        raise RuntimeError(f"Invalid category/coreset metadata in {bundle}")
    return category, Decimal(str(ratio))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/mvtec15"))
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--skip-engine-build", action="store_true")
    args = parser.parse_args()
    if not args.bundle_root.is_dir():
        parser.error(f"bundle-root does not exist: {args.bundle_root}")
    if not args.dataset_root.is_dir():
        parser.error(f"dataset-root does not exist: {args.dataset_root}")

    bundles: dict[str, Path] = {}
    for bundle in sorted(args.bundle_root.rglob("*.zip")):
        category, ratio = identify(bundle)
        if category not in MVTEC_CATEGORIES:
            raise RuntimeError(f"Unexpected MVTec category in {bundle}: {category}")
        if ratio != EXPECTED_RATIO:
            raise RuntimeError(f"Expected 10% coreset in {bundle}, got {ratio}")
        if category in bundles:
            raise RuntimeError(f"Duplicate bundle for {category}: {bundles[category]}, {bundle}")
        bundles[category] = bundle
    missing = [category for category in MVTEC_CATEGORIES if category not in bundles]
    if missing:
        raise RuntimeError(f"Missing category bundles: {', '.join(missing)}")

    repository = Path(__file__).resolve().parents[2]
    prepare = repository / "jetson" / "scripts" / "prepare_colab_bundle.py"
    experiments: list[dict[str, object]] = []
    for category in MVTEC_CATEGORIES:
        benchmark_image = args.dataset_root / category / "test" / "good" / "000.png"
        if not benchmark_image.is_file():
            raise RuntimeError(f"Benchmark image is missing: {benchmark_image}")
        artifact_root = args.artifact_root / category
        config = args.config_dir / f"jetson_{category}_fp16_c01_s5.yaml"
        command = [
            sys.executable, str(prepare), "--bundle", str(bundles[category]),
            "--artifact-root", str(artifact_root), "--config-output", str(config),
            "--expected-category", category, "--expected-coreset-ratio", "0.1",
        ]
        if args.skip_engine_build:
            command.append("--skip-engine-build")
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        if not args.skip_engine_build:
            try:
                benchmark = str(benchmark_image.resolve().relative_to(repository.resolve()))
                config_path = str(config.resolve().relative_to(repository.resolve()))
            except ValueError as error:
                raise RuntimeError("dataset-root and config-dir must be inside the repository") from error
            experiments.append({
                "experiment_id": f"paper-mvtec15-{category}-fp16-s5",
                "groups": ["multi_category"],
                "config": config_path,
                "benchmark_image": benchmark,
                "category": category,
            })

    if not args.skip_engine_build:
        if args.plan_output.exists():
            parser.error(f"refusing to overwrite plan: {args.plan_output}")
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(
            json.dumps({"schema_version": 1, "experiments": experiments}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"plan={args.plan_output}")
    print(f"prepared_categories={len(MVTEC_CATEGORIES)}")


if __name__ == "__main__":
    main()
