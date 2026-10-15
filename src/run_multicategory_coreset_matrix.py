#!/usr/bin/env python3
"""Run a resumable category x seed x coreset-ratio artifact matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


MVTEC_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
)
DEFAULT_RATIOS = (0.01, 0.025, 0.05, 0.10, 0.20)
DEFAULT_SEEDS = (42, 123, 2026)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=MVTEC_CATEGORIES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--calibration-images", type=int, default=100)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()
    if not args.dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {args.dataset_root}")
    if len(set(args.categories)) != len(args.categories):
        parser.error("categories must be unique")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.ratios)) != len(args.ratios):
        parser.error("seeds and ratios must be unique")
    unknown = sorted(set(args.categories) - set(MVTEC_CATEGORIES))
    if unknown:
        parser.error("unknown MVTec categories: " + ", ".join(unknown))

    source = Path(__file__).resolve().parent
    runner = source / "run_coreset_ablation.py"
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "categories": args.categories,
        "seeds": args.seeds,
        "ratios": args.ratios,
        "variant_count": len(args.categories) * len(args.seeds) * len(args.ratios),
        "policy": "resume_completed_variants_and_rebuild_incomplete_variants",
    }
    plan_path = args.output_root / "MATRIX_PLAN.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        for field in ("categories", "seeds", "ratios", "variant_count", "policy"):
            if existing.get(field) != plan[field]:
                parser.error(f"existing matrix plan differs in {field}: {plan_path}")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    for category in args.categories:
        for seed in args.seeds:
            output = args.output_root / f"seed{seed}"
            command = [
                sys.executable, str(runner),
                "--dataset-root", str(args.dataset_root),
                "--category", category,
                "--output-root", str(output),
                "--ratios", *(f"{ratio:.12g}" for ratio in args.ratios),
                "--calibration-images", str(args.calibration_images),
                "--accelerator", args.accelerator,
                "--devices", str(args.devices),
                "--seed", str(seed),
                "--resume",
            ]
            print("+", " ".join(command), flush=True)
            subprocess.run(command, check=True)
    print(f"matrix_complete=true")
    print(f"variants={plan['variant_count']}")
    print(f"output_root={args.output_root}")


if __name__ == "__main__":
    main()
