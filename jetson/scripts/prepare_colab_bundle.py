#!/usr/bin/env python3
"""Prepare one traced Colab artifact bundle for Jetson experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import zipfile
from decimal import Decimal
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_sha256(stream: object) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def ratio_tag(value: float) -> str:
    digits = format(Decimal(str(value)).normalize(), "f").replace(".", "")
    return f"c{digits}"


def verify_members(archive: zipfile.ZipFile) -> list[dict[str, object]]:
    names = archive.namelist()
    if len(names) != len(set(names)) or "bundle_manifest.json" not in names:
        raise RuntimeError("Bundle has duplicate entries or no bundle manifest")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe archive path: {name}")
        info = archive.getinfo(name)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise RuntimeError(f"Bundle contains a symbolic link: {name}")
    manifest = json.loads(archive.read("bundle_manifest.json"))
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("Bundle manifest has no entries")
    expected = {"bundle_manifest.json"}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("Invalid bundle manifest entry")
        name = entry.get("archive_path")
        expected_hash = entry.get("sha256")
        if not isinstance(name, str) or not isinstance(expected_hash, str):
            raise RuntimeError("Invalid bundle manifest entry")
        expected.add(name)
        with archive.open(name) as stream:
            digest = stream_sha256(stream)
        if digest != expected_hash:
            raise RuntimeError(f"Bundle artifact hash mismatch: {name}")
    if set(names) != expected:
        raise RuntimeError("Bundle contains files not declared in its manifest")
    return entries


def find_trtexec() -> Path:
    command = shutil.which("trtexec")
    candidates = [Path(command)] if command else []
    candidates.append(Path("/usr/src/tensorrt/bin/trtexec"))
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise RuntimeError("trtexec was not found")
    return path


def relative_to_repository(path: Path, repository: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository.resolve()))
    except ValueError:
        return str(resolved)


def render_config(
    template: Path,
    destination: Path,
    repository: Path,
    artifact_root: Path,
    category: str,
    coreset_ratio: float,
) -> None:
    replacements = {
        "engine_path": artifact_root / "models" / "patchcore_feature_extractor_fp16.engine",
        "memory_bank_path": artifact_root / "models" / "patchcore_memory_bank.npy",
        "engine_manifest_path": artifact_root / "models" / "patchcore_feature_extractor_fp16.engine.json",
        "export_manifest_path": artifact_root / "models" / "patchcore_export_manifest.json",
    }
    output: list[str] = []
    seen: set[str] = set()
    for raw in template.read_text(encoding="utf-8").splitlines():
        key = raw.split(":", 1)[0].strip() if ":" in raw else ""
        if key in replacements:
            output.append(f"{key}: {relative_to_repository(replacements[key], repository)}")
            seen.add(key)
        elif key == "category":
            output.append(f"category: {category}")
            seen.add(key)
        elif key == "coreset_ratio":
            output.append(f"coreset_ratio: {coreset_ratio:.12g}")
            seen.add(key)
        else:
            output.append(raw)
    required = {*replacements, "category", "coreset_ratio"}
    if seen != required:
        raise RuntimeError(f"Config template is missing keys: {', '.join(sorted(required - seen))}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument(
        "--config-template",
        type=Path,
        default=Path("configs/jetson_fp16_cuda_tiled_async_transpose_s5.yaml"),
    )
    parser.add_argument("--expected-category")
    parser.add_argument("--expected-coreset-ratio", type=float)
    parser.add_argument("--skip-engine-build", action="store_true")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    bundle = args.bundle.resolve()
    artifact_root = args.artifact_root.resolve()
    config_output = args.config_output.resolve()
    template = (repository / args.config_template).resolve() if not args.config_template.is_absolute() else args.config_template
    if not bundle.is_file() or not template.is_file():
        parser.error("bundle and config template must exist")
    if artifact_root.exists() and any(artifact_root.iterdir()):
        parser.error(f"artifact-root must be empty or absent: {artifact_root}")
    if config_output.exists():
        parser.error(f"refusing to overwrite config: {config_output}")

    with zipfile.ZipFile(bundle) as archive:
        entries = verify_members(archive)
        archive.extractall(artifact_root)

    export_path = artifact_root / "models" / "patchcore_export_manifest.json"
    training_path = artifact_root / "results" / "training_manifest.json"
    for path in (export_path, training_path):
        if not path.is_file():
            raise RuntimeError(f"Bundle is missing required artifact: {path}")
    export = json.loads(export_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    dataset = training.get("dataset")
    model = training.get("model")
    if not isinstance(dataset, dict) or not isinstance(model, dict):
        raise RuntimeError("Training manifest has no dataset/model metadata")
    category = dataset.get("category")
    ratio = model.get("coreset_ratio")
    if not isinstance(category, str) or not isinstance(ratio, (int, float)):
        raise RuntimeError("Training manifest has invalid category or coreset ratio")
    if export.get("coreset_ratio") != ratio:
        raise RuntimeError("Training/export coreset ratios differ")
    if args.expected_category is not None and category != args.expected_category:
        raise RuntimeError(f"Category mismatch: expected {args.expected_category}, got {category}")
    if args.expected_coreset_ratio is not None and Decimal(str(ratio)) != Decimal(str(args.expected_coreset_ratio)):
        raise RuntimeError(
            f"Coreset ratio mismatch: expected {args.expected_coreset_ratio}, got {ratio}"
        )

    models = artifact_root / "models"
    onnx = models / "patchcore_feature_extractor.onnx"
    engine = models / "patchcore_feature_extractor_fp16.engine"
    engine_manifest = engine.with_suffix(engine.suffix + ".json")
    if not args.skip_engine_build:
        trtexec = find_trtexec()
        subprocess.run(
            [str(trtexec), f"--onnx={onnx}", f"--saveEngine={engine}", "--fp16", "--skipInference"],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(repository / "jetson" / "scripts" / "write_engine_manifest.py"),
                "--onnx", str(onnx), "--engine", str(engine), "--precision", "fp16",
                "--trtexec", str(trtexec), "--output", str(engine_manifest),
                "--export-manifest", str(export_path),
            ],
            check=True,
        )

    if not args.skip_engine_build:
        for path in (onnx, engine, engine_manifest):
            if not path.is_file():
                raise RuntimeError(f"Prepared runtime artifact is missing: {path}")
        render_config(template, config_output, repository, artifact_root, category, float(ratio))

    preparation = {
        "schema_version": 1,
        "bundle": {"path": str(bundle), "sha256": sha256(bundle), "files": len(entries)},
        "category": category,
        "coreset_ratio": float(ratio),
        "ratio_tag": ratio_tag(float(ratio)),
        "artifact_root": str(artifact_root),
        "config": str(config_output) if not args.skip_engine_build else None,
        "engine_built": not args.skip_engine_build,
    }
    (artifact_root / "jetson_preparation.json").write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("bundle_valid=true")
    print(f"category={category}")
    print(f"coreset_ratio={float(ratio):.12g}")
    print(f"artifact_root={artifact_root}")
    if not args.skip_engine_build:
        print(f"config={config_output}")


if __name__ == "__main__":
    main()
