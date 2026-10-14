#!/usr/bin/env python3
"""Generate controlled nearest-neighbor backend configs and an experiment plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BACKENDS = (
    "cpu",
    "cublas",
    "cuda",
    "cuda_warp",
    "cuda_tiled",
    "cuda_tiled_async",
    "cuda_tiled_async_transpose",
)
FIXED_KEYS = (
    "engine_path",
    "memory_bank_path",
    "engine_manifest_path",
    "export_manifest_path",
    "category",
    "model",
    "precision",
    "bank_precision",
    "coreset_ratio",
    "input_width",
    "input_height",
    "num_neighbors",
    "gaussian_sigma",
    "warmup",
    "repeats",
)


def replace_value(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines) if line.split("#", 1)[0].strip().startswith(f"{key}:")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {key} entry, found {len(matches)}")
    lines[matches[0]] = f"{key}: {value}"
    return "\n".join(lines) + "\n"


def flat_config(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise RuntimeError(f"Invalid config line {line_number}")
        key, value = (part.strip() for part in line.split(":", 1))
        if not key or not value or key in values:
            raise RuntimeError(f"Invalid or duplicate config key {key}")
        values[key] = value
    return values


def backend_tag(backend: str) -> str:
    return backend.replace("_", "-")


def generate(base_text: str) -> dict[str, str]:
    base = flat_config(base_text)
    missing = [key for key in FIXED_KEYS if key not in base]
    if missing:
        raise RuntimeError(f"Base config is missing required keys: {missing}")
    generated: dict[str, str] = {}
    for backend in BACKENDS:
        text = replace_value(base_text, "nn_backend", backend)
        text = replace_value(text, "optimization_stage", "S0")
        values = flat_config(text)
        if any(values[key] != base[key] for key in FIXED_KEYS):
            raise RuntimeError(f"Fixed condition changed for backend {backend}")
        generated[backend] = text
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-config-dir", type=Path, required=True)
    parser.add_argument("--benchmark-image", required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--experiment-prefix", default="paper-nn-bottle")
    args = parser.parse_args()

    if not args.base_config.is_file():
        parser.error(f"base config does not exist: {args.base_config}")
    if args.output_plan.exists():
        parser.error(f"refusing to overwrite plan: {args.output_plan}")
    if args.output_config_dir.exists() and any(args.output_config_dir.iterdir()):
        parser.error(f"refusing to write into non-empty directory: {args.output_config_dir}")

    repository = Path(__file__).resolve().parents[1]
    generated = generate(args.base_config.read_text(encoding="utf-8"))
    args.output_config_dir.mkdir(parents=True, exist_ok=True)
    experiments = []
    for backend, text in generated.items():
        config = args.output_config_dir / f"jetson_bottle_fp16_{backend}_s0.yaml"
        config.write_text(text, encoding="utf-8")
        try:
            config_path = str(config.resolve().relative_to(repository))
        except ValueError:
            config_path = str(config.resolve())
        experiments.append(
            {
                "experiment_id": f"{args.experiment_prefix}-{backend_tag(backend)}",
                "groups": ["nn_backend"],
                "config": config_path,
                "benchmark_image": args.benchmark_image,
                "category": "bottle",
            }
        )
    document = {"schema_version": 1, "experiments": experiments}
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"experiments={len(experiments)}")
    print(f"output={args.output_plan}")


if __name__ == "__main__":
    main()
