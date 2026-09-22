#!/usr/bin/env python3
"""Convert a PatchCore memory bank to a deployment storage precision."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; pass --overwrite to replace it")

    memory_bank = np.load(args.input, allow_pickle=False)
    if memory_bank.ndim != 2 or 0 in memory_bank.shape:
        raise SystemExit(f"Memory bank must be a non-empty 2D array, got {memory_bank.shape}")
    if not np.issubdtype(memory_bank.dtype, np.floating):
        raise SystemExit(f"Memory bank must contain floating-point values, got {memory_bank.dtype}")
    if not np.all(np.isfinite(memory_bank)):
        raise SystemExit("Memory bank contains NaN or infinity")

    target_dtype = np.float16 if args.dtype == "fp16" else np.float32
    converted = np.ascontiguousarray(memory_bank, dtype=target_dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, converted, allow_pickle=False)

    source_fp32 = memory_bank.astype(np.float32, copy=False)
    restored = converted.astype(np.float32)
    error = np.abs(restored - source_fp32)
    print(f"shape={converted.shape}")
    print(f"source_dtype={memory_bank.dtype}")
    print(f"target_dtype={converted.dtype}")
    print(f"size_bytes={converted.nbytes}")
    print(f"mae={float(error.mean()):.10g}")
    print(f"max_error={float(error.max()):.10g}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
