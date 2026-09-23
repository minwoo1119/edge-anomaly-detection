#!/usr/bin/env python3
"""Verify a Colab artifact bundle without extracting untrusted paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import PurePosixPath, Path


def stream_sha256(stream: object) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if not args.bundle.is_file():
        parser.error(f"bundle does not exist: {args.bundle}")

    with zipfile.ZipFile(args.bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "bundle_manifest.json" not in names:
            raise RuntimeError("Bundle has duplicate entries or no bundle manifest")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Unsafe archive path: {name}")
        manifest = json.loads(archive.read("bundle_manifest.json"))
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise RuntimeError("Bundle manifest has no entries")
        expected_names = {"bundle_manifest.json"}
        for entry in entries:
            archive_path = entry.get("archive_path")
            expected_hash = entry.get("sha256")
            if not isinstance(archive_path, str) or not isinstance(expected_hash, str):
                raise RuntimeError("Invalid bundle manifest entry")
            expected_names.add(archive_path)
            with archive.open(archive_path) as stream:
                if stream_sha256(stream) != expected_hash:
                    raise RuntimeError(f"Bundle artifact hash mismatch: {archive_path}")
        if set(names) != expected_names:
            raise RuntimeError("Bundle contains files not declared in its manifest")
    print("bundle_valid=true")
    print(f"files={len(entries)}")


if __name__ == "__main__":
    main()
