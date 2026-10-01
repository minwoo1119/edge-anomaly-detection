#!/usr/bin/env python3
"""Validate that a benchmark config uses artifacts from one recorded export chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def validate_coreset_lineage(
    export_manifest: dict[str, object],
    export_manifest_path: Path,
    coreset_ratio: float,
) -> None:
    """Validate training, checkpoint, and coreset provenance for an export."""
    training_manifest = export_manifest.get("training_manifest")
    if not isinstance(training_manifest, dict):
        raise RuntimeError("Export manifest has no training-manifest lineage")
    training_path = Path(str(training_manifest.get("path", "")))
    if not training_path.is_file():
        candidates = (
            export_manifest_path.parent / "training_manifest.json",
            export_manifest_path.parent.parent / "results" / "training_manifest.json",
        )
        training_path = next((path for path in candidates if path.is_file()), training_path)
    if not training_path.is_file():
        raise RuntimeError("Training manifest referenced by export is unavailable")
    if training_manifest.get("sha256") != sha256(training_path):
        raise RuntimeError("Training manifest SHA-256 does not match export lineage")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    validate_coreset_metadata(export_manifest, training, coreset_ratio)
    export_checkpoint = export_manifest["checkpoint"]
    packaged_checkpoint = export_manifest_path.parent / "checkpoint.ckpt"
    if packaged_checkpoint.is_file() and sha256(packaged_checkpoint) != export_checkpoint.get("sha256"):
        raise RuntimeError("Packaged checkpoint SHA-256 does not match export lineage")


def validate_coreset_metadata(
    export_manifest: dict[str, object],
    training_manifest: dict[str, object],
    coreset_ratio: float,
) -> None:
    """Validate coreset and checkpoint metadata without filesystem assumptions."""
    if not math.isclose(
        float(export_manifest.get("coreset_ratio", -1.0)),
        coreset_ratio,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"Coreset ratio mismatch: config={coreset_ratio}, "
            f"export={export_manifest.get('coreset_ratio')}"
        )
    training_model = training_manifest.get("model")
    if not isinstance(training_model, dict) or not math.isclose(
        float(training_model.get("coreset_ratio", -1.0)),
        coreset_ratio,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Training-manifest coreset ratio does not match the config")
    export_checkpoint = export_manifest.get("checkpoint")
    training_checkpoint = training_manifest.get("checkpoint")
    if not isinstance(export_checkpoint, dict) or not isinstance(training_checkpoint, dict):
        raise RuntimeError("Checkpoint lineage is missing from training/export manifests")
    if export_checkpoint.get("sha256") != training_checkpoint.get("sha256"):
        raise RuntimeError("Training and export checkpoint SHA-256 values differ")


def validate_threshold_metadata(
    threshold_manifest: dict[str, object],
    export_manifest: dict[str, object],
    category: str,
    threshold: float,
    threshold_source: str,
) -> None:
    if threshold_manifest.get("test_set_used_for_threshold_tuning") is not False:
        raise RuntimeError("Threshold manifest does not prohibit test-set tuning")
    if threshold_source != "train_normal" or threshold_manifest.get("source") != threshold_source:
        raise RuntimeError("Enabled decision must use a train-normal threshold")
    if threshold_manifest.get("threshold_space") != "raw":
        raise RuntimeError("Threshold manifest must use raw PatchCore score space")
    if threshold_manifest.get("category") != category:
        raise RuntimeError("Threshold category does not match the config")
    if not math.isclose(
        float(threshold_manifest.get("threshold", math.nan)),
        threshold,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Threshold value does not match the config")
    threshold_checkpoint = threshold_manifest.get("checkpoint")
    export_checkpoint = export_manifest.get("checkpoint")
    if not isinstance(threshold_checkpoint, dict) or not isinstance(export_checkpoint, dict):
        raise RuntimeError("Threshold/export checkpoint lineage is missing")
    if threshold_checkpoint.get("sha256") != export_checkpoint.get("sha256"):
        raise RuntimeError("Threshold and export checkpoint SHA-256 values differ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--memory-bank", type=Path, required=True)
    parser.add_argument("--engine-manifest", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), required=True)
    parser.add_argument("--coreset-ratio", type=float, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--decision-enabled", choices=("true", "false"), required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--threshold-source", required=True)
    parser.add_argument("--threshold-manifest", type=Path)
    args = parser.parse_args()

    for path in (args.engine, args.memory_bank, args.engine_manifest, args.export_manifest):
        if not path.is_file():
            parser.error(f"artifact does not exist: {path}")
    engine_manifest = json.loads(args.engine_manifest.read_text(encoding="utf-8"))
    export_manifest = json.loads(args.export_manifest.read_text(encoding="utf-8"))
    validate_coreset_lineage(export_manifest, args.export_manifest, args.coreset_ratio)
    if args.decision_enabled == "true":
        if args.threshold_manifest is None or not args.threshold_manifest.is_file():
            raise RuntimeError("Enabled decision requires a threshold manifest")
        threshold_manifest = json.loads(args.threshold_manifest.read_text(encoding="utf-8"))
        validate_threshold_metadata(
            threshold_manifest,
            export_manifest,
            args.category,
            args.threshold,
            args.threshold_source,
        )
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
