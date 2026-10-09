#!/usr/bin/env python3
"""Generate a publication coreset experiment plan from prepared Jetson configs."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path


EXPECTED_RATIOS = tuple(Decimal(value) for value in ("0.01", "0.025", "0.05", "0.1", "0.2"))


def flat_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise RuntimeError(f"Invalid config line {line_number}: {path}")
        key, value = (part.strip() for part in line.split(":", 1))
        if not key or not value or key in values:
            raise RuntimeError(f"Invalid or duplicate config key {key}: {path}")
        values[key] = value
    return values


def ratio_tag(value: Decimal) -> str:
    return "c" + format(value.normalize(), "f").replace(".", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--benchmark-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-prefix", default="paper-coreset")
    args = parser.parse_args()

    if not args.config_dir.is_dir():
        parser.error(f"config directory does not exist: {args.config_dir}")
    if args.output.exists():
        parser.error(f"refusing to overwrite plan: {args.output}")
    configs: dict[Decimal, Path] = {}
    for path in sorted(args.config_dir.glob("*.yaml")):
        values = flat_config(path)
        if values.get("category") != args.category:
            continue
        if values.get("precision") != "fp16" or values.get("bank_precision") != "fp32":
            continue
        if values.get("nn_backend") != "cuda_tiled_async_transpose":
            continue
        ratio = Decimal(values["coreset_ratio"])
        if ratio in configs:
            raise RuntimeError(f"Duplicate coreset ratio {ratio}: {configs[ratio]}, {path}")
        configs[ratio] = path
    missing = [ratio for ratio in EXPECTED_RATIOS if ratio not in configs]
    extra = [ratio for ratio in configs if ratio not in EXPECTED_RATIOS]
    if missing or extra:
        raise RuntimeError(
            "Coreset config coverage mismatch; "
            f"missing={[str(value) for value in missing]}, extra={[str(value) for value in extra]}"
        )

    repository = Path(__file__).resolve().parents[1]
    experiments = []
    for ratio in EXPECTED_RATIOS:
        config = configs[ratio].resolve()
        try:
            config_text = str(config.relative_to(repository))
        except ValueError:
            config_text = str(config)
        experiments.append(
            {
                "experiment_id": f"{args.experiment_prefix}-{args.category}-{ratio_tag(ratio)}",
                "groups": ["coreset"],
                "config": config_text,
                "benchmark_image": args.benchmark_image,
                "category": args.category,
            }
        )
    document = {"schema_version": 1, "experiments": experiments}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"experiments={len(experiments)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
