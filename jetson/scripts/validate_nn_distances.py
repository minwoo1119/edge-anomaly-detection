#!/usr/bin/env python3
"""Check dumped NN distances against direct FP64 distances at selected indices.

This validates distance arithmetic, not global neighbor optimality. The caller
must also compare selected indices exactly against the Python reference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    bank = np.load(args.bank, mmap_mode="r", allow_pickle=False)
    patches = np.load(args.patches, allow_pickle=False).astype(np.float64)
    indices = np.load(args.indices, allow_pickle=False).reshape(-1)
    scores = np.load(args.scores, allow_pickle=False).reshape(-1).astype(np.float64)
    if (bank.ndim != 2 or patches.ndim != 2
            or patches.shape != (scores.size, bank.shape[1])
            or indices.size != scores.size or scores.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= bank.shape[0])):
        raise SystemExit("FAIL: invalid NN dump shapes or indices")
    distances = np.linalg.norm(patches - bank[indices].astype(np.float64), axis=1)
    errors = np.abs(scores - distances)
    mae, maximum = float(errors.mean()), float(errors.max())
    passed = bool(np.all(np.isfinite(errors)) and mae <= 1e-4 and maximum <= 1e-3)
    hashes = {}
    for name in ("bank", "patches", "indices", "scores"):
        path = getattr(args, name)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = {"path": str(path.resolve()), "sha256": digest.hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "reference_method": "direct_fp64_distance_at_selected_indices",
        "artifacts": hashes, "mae": mae, "max_error": maximum,
        "tolerances": {"max_mae": 1e-4, "max_error": 1e-3}, "passed": passed,
    }, indent=2) + "\n")
    print(f"FP64 distance check: mae={mae:.10g}, max_error={maximum:.10g}")
    if not passed:
        raise SystemExit("FAIL: direct FP64 distance tolerance exceeded")
    print("PASS")


if __name__ == "__main__":
    main()
