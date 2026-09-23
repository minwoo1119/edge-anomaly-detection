#!/usr/bin/env python3
"""Compare a reference NumPy tensor with a C++/TensorRT runtime dump."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-mae", type=float, default=1e-4)
    parser.add_argument("--max-error", type=float, default=1e-3)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    reference_raw = np.load(args.reference, allow_pickle=False)
    candidate_raw = np.load(args.candidate, allow_pickle=False)
    if reference_raw.shape != candidate_raw.shape:
        raise SystemExit(f"FAIL shape: reference={reference_raw.shape}, candidate={candidate_raw.shape}")
    if not np.all(np.isfinite(reference_raw)) or not np.all(np.isfinite(candidate_raw)):
        raise SystemExit("FAIL: inputs contain NaN or infinity")
    reference = reference_raw.astype(np.float64, copy=False)
    candidate = candidate_raw.astype(np.float64, copy=False)

    difference = candidate - reference
    absolute = np.abs(difference)
    mae = float(absolute.mean())
    maximum = float(absolute.max())
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_norm = float(np.linalg.norm(reference.ravel()))
    relative_l2 = float(np.linalg.norm(difference.ravel()) / max(reference_norm, np.finfo(float).eps))
    denominator = reference_norm * float(np.linalg.norm(candidate.ravel()))
    cosine = float(np.dot(reference.ravel(), candidate.ravel()) / max(denominator, np.finfo(float).eps))
    passed = (
        bool(np.array_equal(reference_raw, candidate_raw))
        if args.exact
        else mae <= args.max_mae and maximum <= args.max_error and cosine >= args.min_cosine
    )

    if args.report is not None:
        report = {
            "schema_version": 1,
            "reference": str(args.reference.resolve()),
            "reference_sha256": sha256(args.reference),
            "candidate": str(args.candidate.resolve()),
            "candidate_sha256": sha256(args.candidate),
            "shape": list(reference.shape),
            "metrics": {
                "mae": mae,
                "max_error": maximum,
                "rmse": rmse,
                "relative_l2": relative_l2,
                "cosine_similarity": cosine,
            },
            "tolerances": {
                "exact": args.exact,
                "max_mae": args.max_mae,
                "max_error": args.max_error,
                "min_cosine": args.min_cosine,
            },
            "passed": passed,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"shape={reference.shape}")
    print(f"mae={mae:.10g}")
    print(f"max_error={maximum:.10g}")
    print(f"rmse={rmse:.10g}")
    print(f"relative_l2={relative_l2:.10g}")
    print(f"cosine_similarity={cosine:.10g}")
    if not passed:
        raise SystemExit("FAIL: numerical tolerance exceeded")
    print("PASS")


if __name__ == "__main__":
    main()
