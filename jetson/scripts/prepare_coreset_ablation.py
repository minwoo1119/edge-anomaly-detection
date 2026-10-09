#!/usr/bin/env python3
"""Prepare the five publication coreset bundles and their Jetson experiment plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from decimal import Decimal
from pathlib import Path

from prepare_colab_bundle import ratio_tag


EXPECTED_RATIOS = tuple(Decimal(value) for value in ("0.01", "0.025", "0.05", "0.1", "0.2"))


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
    parser.add_argument("--category", required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/coreset"))
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--benchmark-image", required=True)
    parser.add_argument("--skip-engine-build", action="store_true")
    args = parser.parse_args()

    if not args.bundle_root.is_dir():
        parser.error(f"bundle-root does not exist: {args.bundle_root}")
    bundles: dict[Decimal, Path] = {}
    for bundle in sorted(args.bundle_root.rglob("*.zip")):
        category, ratio = identify(bundle)
        if category != args.category:
            continue
        if ratio in bundles:
            raise RuntimeError(f"Duplicate bundle for ratio {ratio}: {bundles[ratio]}, {bundle}")
        bundles[ratio] = bundle
    missing = [ratio for ratio in EXPECTED_RATIOS if ratio not in bundles]
    extra = [ratio for ratio in bundles if ratio not in EXPECTED_RATIOS]
    if missing or extra:
        raise RuntimeError(
            "Coreset bundle coverage mismatch; "
            f"missing={[str(value) for value in missing]}, extra={[str(value) for value in extra]}"
        )

    repository = Path(__file__).resolve().parents[2]
    prepare = repository / "jetson" / "scripts" / "prepare_colab_bundle.py"
    for ratio in EXPECTED_RATIOS:
        tag = ratio_tag(float(ratio))
        artifact_root = args.artifact_root / args.category / tag
        config = args.config_dir / f"jetson_{args.category}_fp16_{tag}_s5.yaml"
        command = [
            sys.executable, str(prepare),
            "--bundle", str(bundles[ratio]),
            "--artifact-root", str(artifact_root),
            "--config-output", str(config),
            "--expected-category", args.category,
            "--expected-coreset-ratio", str(ratio),
        ]
        if args.skip_engine_build:
            command.append("--skip-engine-build")
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    if not args.skip_engine_build:
        command = [
            sys.executable,
            str(repository / "src" / "generate_coreset_experiment_plan.py"),
            "--config-dir", str(args.config_dir),
            "--category", args.category,
            "--benchmark-image", args.benchmark_image,
            "--output", str(args.plan_output),
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    print(f"prepared_variants={len(EXPECTED_RATIOS)}")


if __name__ == "__main__":
    main()
