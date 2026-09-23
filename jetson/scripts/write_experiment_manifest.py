#!/usr/bin/env python3
"""Write immutable artifact and environment metadata for one benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return output or None


def flat_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise RuntimeError(f"Invalid flat config entry: {raw_line}")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def artifact(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "config",
        "image",
        "engine",
        "engine_manifest",
        "export_manifest",
        "memory_bank",
        "executable",
        "benchmark_csv",
        "power_csv",
        "tegrastats",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    git_commit = command_output(["git", "-C", str(repository), "rev-parse", "HEAD"])
    git_status = command_output(["git", "-C", str(repository), "status", "--porcelain"])
    release = Path("/etc/nv_tegra_release")
    export_metadata = json.loads(args.export_manifest.read_text(encoding="utf-8"))
    engine_metadata = json.loads(args.engine_manifest.read_text(encoding="utf-8"))
    checkpoint_metadata = export_metadata.get("checkpoint")
    if not isinstance(checkpoint_metadata, dict) or not isinstance(
        checkpoint_metadata.get("sha256"), str
    ):
        raise RuntimeError("Export manifest has no checkpoint SHA-256 lineage")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "git": {"commit": git_commit, "dirty": bool(git_status), "status": git_status},
        "config": flat_config(args.config),
        "lineage": {
            "checkpoint_sha256": checkpoint_metadata["sha256"],
            "export_manifest_sha256": sha256(args.export_manifest),
            "engine_manifest_sha256": sha256(args.engine_manifest),
            "engine_precision": engine_metadata.get("precision"),
        },
        "artifacts": {
            "config": artifact(args.config),
            "image": artifact(args.image),
            "engine": artifact(args.engine),
            "engine_manifest": artifact(args.engine_manifest),
            "export_manifest": artifact(args.export_manifest),
            "memory_bank": artifact(args.memory_bank),
            "executable": artifact(args.executable),
            "benchmark_csv": artifact(args.benchmark_csv),
            "power_csv": artifact(args.power_csv),
            "tegrastats": artifact(args.tegrastats),
        },
        "environment": {
            "platform": platform.platform(),
            "uname": platform.uname()._asdict(),
            "nv_tegra_release": release.read_text(encoding="utf-8", errors="replace").strip()
            if release.exists()
            else None,
            "nvcc": command_output(["nvcc", "--version"]),
            "nvpmodel": command_output(["nvpmodel", "-q"]),
            "jetson_clocks": command_output(["jetson_clocks", "--show"]),
            "opencv": command_output(["pkg-config", "--modversion", "opencv4"]),
            "tensorrt_packages": command_output(
                ["dpkg-query", "-W", "-f=${Package}=${Version}\n", "libnvinfer*"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment manifest: {args.output}")
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest={args.output}")


if __name__ == "__main__":
    main()
