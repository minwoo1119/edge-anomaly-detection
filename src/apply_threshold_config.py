#!/usr/bin/env python3
"""Create a decision-enabled runtime config from a traced threshold manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def replace_one(config_text: str, key: str, value: str) -> str:
    lines = config_text.splitlines()
    matches = [
        index for index, line in enumerate(lines)
        if line.strip().startswith(f"{key}:")
    ]
    if len(matches) != 1:
        raise ValueError(f"Config must contain exactly one {key} key")
    index = matches[0]
    prefix = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    lines[index] = f"{prefix}{key}: {value}"
    return "\n".join(lines) + "\n"


def config_value(config_text: str, key: str) -> str:
    matches = [
        line.split(":", 1)[1].strip().strip("\"'")
        for line in config_text.splitlines()
        if line.strip().startswith(f"{key}:")
    ]
    if len(matches) != 1:
        raise ValueError(f"Config must contain exactly one {key} key")
    return matches[0]


def apply_threshold(config_text: str, threshold: float, manifest_path: Path) -> str:
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    updated = replace_one(config_text, "decision_enabled", "true")
    updated = replace_one(updated, "threshold", format(threshold, ".17g"))
    updated = replace_one(updated, "threshold_space", "raw")
    updated = replace_one(updated, "threshold_source", "train_normal")
    if any(
        line.strip().startswith("threshold_manifest_path:")
        for line in updated.splitlines()
    ):
        return replace_one(updated, "threshold_manifest_path", str(manifest_path))
    return updated + f"threshold_manifest_path: {manifest_path}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--threshold-manifest", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.config, args.threshold_manifest, args.export_manifest):
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    if args.output.exists():
        parser.error(f"refusing to overwrite output config: {args.output}")
    config_text = args.config.read_text(encoding="utf-8")
    threshold = json.loads(args.threshold_manifest.read_text(encoding="utf-8"))
    export = json.loads(args.export_manifest.read_text(encoding="utf-8"))
    if threshold.get("source") != "train_normal":
        raise RuntimeError("Threshold source must be train_normal")
    if threshold.get("threshold_space") != "raw":
        raise RuntimeError("Threshold must use raw PatchCore score space")
    if threshold.get("test_set_used_for_threshold_tuning") is not False:
        raise RuntimeError("Threshold manifest does not prohibit test-set tuning")
    if threshold.get("category") != config_value(config_text, "category"):
        raise RuntimeError("Threshold category does not match runtime config")
    threshold_checkpoint = threshold.get("checkpoint")
    export_checkpoint = export.get("checkpoint")
    if not isinstance(threshold_checkpoint, dict) or not isinstance(export_checkpoint, dict):
        raise RuntimeError("Checkpoint lineage is absent from threshold/export manifests")
    if threshold_checkpoint.get("sha256") != export_checkpoint.get("sha256"):
        raise RuntimeError("Threshold and export checkpoint SHA-256 values differ")
    threshold_value = float(threshold.get("threshold"))
    output = apply_threshold(config_text, threshold_value, args.threshold_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"config={args.output}")
    print(f"threshold={threshold_value:.9g}")


if __name__ == "__main__":
    main()
