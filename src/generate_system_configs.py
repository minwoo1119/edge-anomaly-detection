#!/usr/bin/env python3
"""Generate traceable S0-S6 configs from one environment-specific baseline config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IMPLEMENTED_STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def with_stage(config_text: str, stage: str) -> str:
    if stage not in IMPLEMENTED_STAGES:
        raise ValueError(f"System stage is not implemented: {stage}")
    lines = config_text.splitlines()
    matches = [index for index, line in enumerate(lines) if line.strip().startswith("optimization_stage:")]
    if len(matches) != 1:
        raise ValueError("Baseline config must contain exactly one optimization_stage key")
    prefix = lines[matches[0]][: len(lines[matches[0]]) - len(lines[matches[0]].lstrip())]
    lines[matches[0]] = f"{prefix}optimization_stage: {stage}"
    if stage in {"S5", "S6"}:
        backend_matches = [
            index for index, line in enumerate(lines) if line.strip().startswith("nn_backend:")
        ]
        if len(backend_matches) != 1:
            raise ValueError("Baseline config must contain exactly one nn_backend key")
        backend_prefix = lines[backend_matches[0]][
            : len(lines[backend_matches[0]]) - len(lines[backend_matches[0]].lstrip())
        ]
        backend = lines[backend_matches[0]].split(":", 1)[1].split("#", 1)[0].strip()
        cuda_backends = {"cuda", "cuda_warp", "cuda_tiled", "cuda_tiled_cached", "cuda_tiled_async"}
        # Preserve the selected kernel during system-stage comparisons.
        selected = backend if backend in cuda_backends else "cuda"
        lines[backend_matches[0]] = f"{backend_prefix}nn_backend: {selected}"
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=IMPLEMENTED_STAGES)
    args = parser.parse_args()

    if not args.baseline.is_file():
        parser.error(f"baseline config does not exist: {args.baseline}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output-dir must be empty or absent: {args.output_dir}")
    if len(set(args.stages)) != len(args.stages):
        parser.error("stages must be unique")
    unsupported = [stage for stage in args.stages if stage not in IMPLEMENTED_STAGES]
    if unsupported:
        parser.error("stages are not implemented: " + ", ".join(unsupported))

    baseline_text = args.baseline.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, str]] = []
    for stage in args.stages:
        path = args.output_dir / f"{args.baseline.stem}_{stage.lower()}.yaml"
        path.write_text(with_stage(baseline_text, stage), encoding="utf-8")
        outputs.append({"stage": stage, "path": str(path.resolve()), "sha256": sha256(path)})
    manifest = {
        "schema_version": 1,
        "baseline": {
            "path": str(args.baseline.resolve()),
            "sha256": sha256(args.baseline),
        },
        "implemented_stages": list(IMPLEMENTED_STAGES),
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "system_configs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"configs={len(outputs)}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
