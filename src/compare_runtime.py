#!/usr/bin/env python3
"""Compare a reference NumPy tensor with a C++/TensorRT runtime dump."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-mae", type=float, default=1e-4)
    parser.add_argument("--max-error", type=float, default=1e-3)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    args = parser.parse_args()

    reference = np.load(args.reference).astype(np.float64, copy=False)
    candidate = np.load(args.candidate).astype(np.float64, copy=False)
    if reference.shape != candidate.shape:
        raise SystemExit(f"FAIL shape: reference={reference.shape}, candidate={candidate.shape}")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise SystemExit("FAIL: inputs contain NaN or infinity")

    difference = candidate - reference
    absolute = np.abs(difference)
    mae = float(absolute.mean())
    maximum = float(absolute.max())
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_norm = float(np.linalg.norm(reference.ravel()))
    relative_l2 = float(np.linalg.norm(difference.ravel()) / max(reference_norm, np.finfo(float).eps))
    denominator = reference_norm * float(np.linalg.norm(candidate.ravel()))
    cosine = float(np.dot(reference.ravel(), candidate.ravel()) / max(denominator, np.finfo(float).eps))

    print(f"shape={reference.shape}")
    print(f"mae={mae:.10g}")
    print(f"max_error={maximum:.10g}")
    print(f"rmse={rmse:.10g}")
    print(f"relative_l2={relative_l2:.10g}")
    print(f"cosine_similarity={cosine:.10g}")
    if mae > args.max_mae or maximum > args.max_error or cosine < args.min_cosine:
        raise SystemExit("FAIL: numerical tolerance exceeded")
    print("PASS")


if __name__ == "__main__":
    main()
