#!/usr/bin/env python3
"""Copy complete MVTec AD category trees to persistent storage."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MVTEC_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
)


def required_directories(category_root: Path) -> tuple[Path, Path, Path]:
    return (
        category_root / "train" / "good",
        category_root / "test" / "good",
        category_root / "ground_truth",
    )


def validate_category(category_root: Path) -> None:
    missing = [path for path in required_directories(category_root) if not path.is_dir()]
    if missing:
        raise RuntimeError(
            f"Incomplete MVTec category {category_root.name}: "
            + ", ".join(str(path) for path in missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=list(MVTEC_CATEGORIES))
    args = parser.parse_args()

    unknown = sorted(set(args.categories) - set(MVTEC_CATEGORIES))
    if unknown:
        parser.error(f"unknown MVTec categories: {', '.join(unknown)}")
    if args.source_root.resolve() == args.destination_root.resolve():
        parser.error("source-root and destination-root must differ")

    copied = 0
    skipped = 0
    for category in args.categories:
        source = args.source_root / category
        destination = args.destination_root / category
        validate_category(source)
        if all(path.is_dir() for path in required_directories(destination)):
            print(f"SKIP cached: {category}", flush=True)
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        validate_category(destination)
        print(f"CACHED: {category} -> {destination}", flush=True)
        copied += 1

    print(f"cached_categories={copied}")
    print(f"skipped_categories={skipped}")


if __name__ == "__main__":
    main()
