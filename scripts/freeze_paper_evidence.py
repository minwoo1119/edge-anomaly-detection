#!/usr/bin/env python3
"""Freeze hashes and basic metadata for publication evidence artifacts."""

from __future__ import annotations

import argparse
import csv
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


def describe(path: Path) -> dict[str, object]:
    item: dict[str, object] = {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            rows = list(reader)
        item["data_rows"] = max(0, len(rows) - 1)
        item["columns"] = rows[0] if rows else []
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            item["schema_version"] = value.get("schema_version")
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("artifacts", type=Path, nargs="+")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite output: {args.output}")
    missing = [path for path in args.artifacts if not path.is_file()]
    if missing:
        parser.error("missing artifacts: " + ", ".join(str(path) for path in missing))
    duplicates = [
        path for path in args.artifacts
        if sum(other.resolve() == path.resolve() for other in args.artifacts) > 1
    ]
    if duplicates:
        parser.error("artifact paths must be unique")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_git_commit": commit,
        "artifacts": [describe(path) for path in args.artifacts],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"artifacts={len(args.artifacts)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
