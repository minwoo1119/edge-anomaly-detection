#!/usr/bin/env python3
"""Package verified Colab artifacts for local analysis and Jetson transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file(
    archive: zipfile.ZipFile,
    source: Path,
    archive_path: str,
    entries: list[dict[str, object]],
    expected_sha256: str | None = None,
) -> None:
    if not source.is_file():
        raise RuntimeError(f"Artifact does not exist: {source}")
    actual_hash = sha256(source)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise RuntimeError(f"Artifact hash mismatch: {source}")
    archive.write(source, archive_path)
    entries.append(
        {
            "archive_path": archive_path,
            "source_path": str(source.resolve()),
            "size_bytes": source.stat().st_size,
            "sha256": actual_hash,
        }
    )


def add_directory(
    archive: zipfile.ZipFile,
    directory: Path,
    archive_root: str,
    entries: list[dict[str, object]],
) -> None:
    if not directory.is_dir():
        raise RuntimeError(f"Artifact directory does not exist: {directory}")
    for source in sorted(path for path in directory.rglob("*") if path.is_file()):
        relative = source.relative_to(directory).as_posix()
        add_file(archive, source, f"{archive_root}/{relative}", entries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--accuracy-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.suffix.lower() != ".zip":
        parser.error("output must use the .zip extension")
    checksum_path = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or checksum_path.exists():
        parser.error("refusing to overwrite artifact bundle or checksum")
    if not args.export_manifest.is_file():
        parser.error(f"export manifest does not exist: {args.export_manifest}")

    export = json.loads(args.export_manifest.read_text(encoding="utf-8"))
    checkpoint = export.get("checkpoint")
    artifacts = export.get("artifacts")
    if not isinstance(checkpoint, dict) or not isinstance(artifacts, list):
        raise RuntimeError("Invalid export manifest structure")

    entries: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as archive:
        add_file(
            archive,
            Path(str(checkpoint["path"])),
            "models/checkpoint.ckpt",
            entries,
            str(checkpoint["sha256"]),
        )
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise RuntimeError("Invalid artifact entry in export manifest")
            source = Path(str(artifact["path"]))
            add_file(
                archive,
                source,
                f"models/{source.name}",
                entries,
                str(artifact["sha256"]),
            )
        add_file(
            archive,
            args.export_manifest,
            "models/patchcore_export_manifest.json",
            entries,
        )
        if args.calibration_dir is not None:
            add_directory(archive, args.calibration_dir, "calibration", entries)
        if args.reference_dir is not None:
            add_directory(archive, args.reference_dir, "reference", entries)
        for source, archive_path in (
            (args.training_manifest, "results/training_manifest.json"),
            (args.evaluation_manifest, "results/evaluation_manifest.json"),
            (args.accuracy_csv, "results/accuracy.csv"),
        ):
            if source is not None:
                add_file(archive, source, archive_path, entries)

        bundle_manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "export_manifest_sha256": sha256(args.export_manifest),
            "entries": entries,
        }
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n",
        )

    bundle_hash = sha256(args.output)
    checksum_path.write_text(f"{bundle_hash}  {args.output.name}\n", encoding="utf-8")
    print(f"bundle={args.output}")
    print(f"bundle_sha256={bundle_hash}")
    print(f"files={len(entries)}")


if __name__ == "__main__":
    main()
