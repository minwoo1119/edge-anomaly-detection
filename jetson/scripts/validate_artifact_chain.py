#!/usr/bin/env python3
"""Validate that a benchmark config uses artifacts from one recorded export chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hash(manifest: dict[str, object], name: str) -> str:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(name), dict):
        raise RuntimeError(f"Engine manifest has no {name} artifact")
    value = artifacts[name].get("sha256")
    if not isinstance(value, str):
        raise RuntimeError(f"Engine manifest {name} artifact has no SHA-256")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--memory-bank", type=Path, required=True)
    parser.add_argument("--engine-manifest", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), required=True)
    args = parser.parse_args()

    for path in (args.engine, args.memory_bank, args.engine_manifest, args.export_manifest):
        if not path.is_file():
            parser.error(f"artifact does not exist: {path}")
    engine_manifest = json.loads(args.engine_manifest.read_text(encoding="utf-8"))
    export_manifest = json.loads(args.export_manifest.read_text(encoding="utf-8"))
    if engine_manifest.get("precision") != args.precision:
        raise RuntimeError(
            f"Engine precision mismatch: config={args.precision}, "
            f"manifest={engine_manifest.get('precision')}"
        )
    if artifact_hash(engine_manifest, "engine") != sha256(args.engine):
        raise RuntimeError("Engine SHA-256 does not match its build manifest")
    if artifact_hash(engine_manifest, "export_manifest") != sha256(args.export_manifest):
        raise RuntimeError("Engine was not built from the selected export manifest")

    export_artifacts = export_manifest.get("artifacts")
    if not isinstance(export_artifacts, list):
        raise RuntimeError("Export manifest has no artifact list")
    hashes = {
        item.get("sha256")
        for item in export_artifacts
        if isinstance(item, dict) and isinstance(item.get("sha256"), str)
    }
    memory_bank_hash = sha256(args.memory_bank)
    if memory_bank_hash not in hashes:
        conversion_path = args.memory_bank.with_suffix(args.memory_bank.suffix + ".json")
        if not conversion_path.is_file():
            raise RuntimeError("Memory bank does not belong to the selected export manifest")
        conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
        source = conversion.get("source")
        output = conversion.get("output")
        source_manifest = conversion.get("source_export_manifest")
        if (
            not isinstance(source, dict)
            or source.get("sha256") not in hashes
            or not isinstance(output, dict)
            or output.get("sha256") != memory_bank_hash
            or not isinstance(source_manifest, dict)
            or source_manifest.get("sha256") != sha256(args.export_manifest)
        ):
            raise RuntimeError("Derived memory-bank provenance is invalid")
    if artifact_hash(engine_manifest, "onnx") not in hashes:
        raise RuntimeError("Engine ONNX does not belong to the selected export manifest")
    print("artifact_chain=valid")


if __name__ == "__main__":
    main()
