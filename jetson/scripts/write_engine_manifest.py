#!/usr/bin/env python3
"""Record provenance for a TensorRT engine produced by trtexec."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    combined = "\n".join(item.strip() for item in (result.stdout, result.stderr) if item.strip())
    return combined or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--trtexec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path)
    args = parser.parse_args()

    for path in (args.onnx, args.engine, args.trtexec):
        if not path.is_file():
            parser.error(f"required artifact does not exist: {path}")
    if args.output.exists():
        parser.error(f"refusing to overwrite engine manifest: {args.output}")
    if args.export_manifest is not None and not args.export_manifest.is_file():
        parser.error(f"export manifest does not exist: {args.export_manifest}")

    repository = Path(__file__).resolve().parents[2]
    external_data = args.onnx.with_name(args.onnx.name + ".data")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "precision": args.precision,
        "git_commit": output(["git", "-C", str(repository), "rev-parse", "HEAD"]),
        "git_dirty": bool(output(["git", "-C", str(repository), "status", "--porcelain"])),
        "trtexec_version": output([str(args.trtexec), "--version"]),
        "builder_flags": [
            f"--onnx={args.onnx}",
            f"--saveEngine={args.engine}",
            *(["--fp16"] if args.precision == "fp16" else []),
            "--skipInference",
        ],
        "artifacts": {
            "onnx": artifact(args.onnx),
            "onnx_external_data": artifact(external_data) if external_data.is_file() else None,
            "engine": artifact(args.engine),
            "export_manifest": artifact(args.export_manifest)
            if args.export_manifest is not None
            else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"engine_manifest={args.output}")


if __name__ == "__main__":
    main()
