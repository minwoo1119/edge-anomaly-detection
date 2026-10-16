#!/usr/bin/env python3
"""Prepare category x seed x ratio Colab bundles for Jetson evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from decimal import Decimal
from pathlib import Path

RATIOS = tuple(Decimal(value) for value in ("0.01", "0.025", "0.05", "0.1", "0.2"))
SEEDS = (42, 123, 2026)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio_tag(value: float) -> str:
    return "c" + format(Decimal(str(value)).normalize(), "f").replace(".", "")


def identify(bundle: Path) -> tuple[str, int, Decimal]:
    with zipfile.ZipFile(bundle) as archive:
        training = json.loads(archive.read("results/training_manifest.json"))
    dataset = training.get("dataset")
    model = training.get("model")
    runtime = training.get("runtime")
    if not all(isinstance(value, dict) for value in (dataset, model, runtime)):
        raise RuntimeError(f"Invalid training manifest in {bundle}")
    category, ratio, seed = dataset.get("category"), model.get("coreset_ratio"), runtime.get("seed")
    if not isinstance(category, str) or not isinstance(ratio, (int, float)) or not isinstance(seed, int):
        raise RuntimeError(f"Invalid category/ratio/seed in {bundle}")
    return category, seed, Decimal(str(ratio))


def already_prepared(root: Path, config: Path, bundle: Path) -> bool:
    manifest = root / "jetson_preparation.json"
    if not manifest.is_file() or not config.is_file():
        return False
    document = json.loads(manifest.read_text(encoding="utf-8"))
    packaged = document.get("bundle")
    return isinstance(packaged, dict) and packaged.get("sha256") == sha256(bundle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/revision_coreset"))
    parser.add_argument("--config-dir", type=Path, default=Path("results/configs/revision_coreset"))
    parser.add_argument("--config-template", type=Path, default=Path("configs/jetson_fp16_faiss_gpu_s0.yaml"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.bundle_root.is_dir():
        parser.error(f"bundle root does not exist: {args.bundle_root}")
    bundles: dict[tuple[str, int, Decimal], Path] = {}
    for bundle in sorted(args.bundle_root.rglob("*.zip")):
        key = identify(bundle)
        if key in bundles:
            raise RuntimeError(f"Duplicate variant {key}: {bundles[key]}, {bundle}")
        if key[1] not in SEEDS or key[2] not in RATIOS:
            raise RuntimeError(f"Unexpected seed or ratio in {bundle}: {key}")
        bundles[key] = bundle
    categories = sorted({key[0] for key in bundles})
    expected = {(category, seed, ratio) for category in categories for seed in SEEDS for ratio in RATIOS}
    missing = sorted(expected - set(bundles))
    if missing and not args.allow_partial:
        raise RuntimeError(f"Matrix is incomplete; missing {len(missing)} variants, first={missing[:10]}")
    if not bundles:
        raise RuntimeError("No artifact bundles found")

    repository = Path(__file__).resolve().parents[2]
    prepare = Path(__file__).with_name("prepare_colab_bundle.py")
    reusable_engine: Path | None = None
    prepared = skipped = 0
    for (category, seed, ratio), bundle in sorted(bundles.items()):
        tag = ratio_tag(float(ratio))
        artifact = args.artifact_root / f"seed{seed}" / category / tag
        config = args.config_dir / f"jetson_{category}_seed{seed}_{tag}_faiss_s0.yaml"
        if args.resume and already_prepared(artifact, config, bundle):
            if reusable_engine is None:
                candidate = artifact / "models" / "patchcore_feature_extractor_fp16.engine"
                if candidate.is_file():
                    reusable_engine = candidate
            skipped += 1
            print(f"skip_complete={artifact}")
            continue
        if artifact.exists() and any(artifact.iterdir()):
            parser.error(f"artifact root is non-empty; use a clean path or --resume: {artifact}")
        if config.exists():
            parser.error(f"config already exists; use a clean path or --resume: {config}")
        command = [
            sys.executable, str(prepare), "--bundle", str(bundle),
            "--artifact-root", str(artifact), "--config-output", str(config),
            "--config-template", str(args.config_template),
            "--expected-category", category, "--expected-coreset-ratio", str(ratio),
        ]
        if reusable_engine is not None:
            command.extend(["--reuse-engine", str(reusable_engine)])
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=repository)
        if reusable_engine is None:
            reusable_engine = artifact / "models" / "patchcore_feature_extractor_fp16.engine"
        prepared += 1
    print(f"categories={len(categories)}")
    print(f"bundles={len(bundles)}")
    print(f"prepared={prepared}")
    print(f"skipped={skipped}")
    print(f"missing={len(missing)}")


if __name__ == "__main__":
    main()
