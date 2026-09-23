#!/usr/bin/env python3
"""Convert a PatchCore memory bank to a deployment storage precision."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    output_manifest = args.output.with_suffix(args.output.suffix + ".json")
    existing = [path for path in (args.output, output_manifest) if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "output already exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )
    if not args.input.is_file() or not args.source_manifest.is_file():
        parser.error("input memory bank and source export manifest must exist")
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    source_hash = sha256(args.input)
    source_artifacts = source_manifest.get("artifacts")
    if not isinstance(source_artifacts, list) or source_hash not in {
        item.get("sha256") for item in source_artifacts if isinstance(item, dict)
    }:
        parser.error("input memory bank does not belong to source export manifest")

    memory_bank = np.load(args.input, allow_pickle=False)
    if memory_bank.ndim != 2 or 0 in memory_bank.shape:
        raise SystemExit(f"Memory bank must be a non-empty 2D array, got {memory_bank.shape}")
    if not np.issubdtype(memory_bank.dtype, np.floating):
        raise SystemExit(f"Memory bank must contain floating-point values, got {memory_bank.dtype}")
    if not np.all(np.isfinite(memory_bank)):
        raise SystemExit("Memory bank contains NaN or infinity")

    target_dtype = np.float16 if args.dtype == "fp16" else np.float32
    converted = np.ascontiguousarray(memory_bank, dtype=target_dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, converted, allow_pickle=False)

    source_fp32 = memory_bank.astype(np.float32, copy=False)
    restored = converted.astype(np.float32)
    error = np.abs(restored - source_fp32)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "conversion": "memory_bank_storage_precision",
        "source": {
            "path": str(args.input.resolve()),
            "sha256": source_hash,
            "dtype": str(memory_bank.dtype),
        },
        "source_export_manifest": {
            "path": str(args.source_manifest.resolve()),
            "sha256": sha256(args.source_manifest),
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "dtype": str(converted.dtype),
            "shape": list(converted.shape),
        },
        "error_after_fp32_restore": {
            "mae": float(error.mean()),
            "max_abs_error": float(error.max()),
        },
    }
    output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"shape={converted.shape}")
    print(f"source_dtype={memory_bank.dtype}")
    print(f"target_dtype={converted.dtype}")
    print(f"size_bytes={converted.nbytes}")
    print(f"mae={float(error.mean()):.10g}")
    print(f"max_error={float(error.max()):.10g}")
    print(f"output={args.output}")
    print(f"manifest={output_manifest}")


if __name__ == "__main__":
    main()
