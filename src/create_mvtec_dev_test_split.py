#!/usr/bin/env python3
"""Freeze a deterministic defect-stratified development/held-out MVTec split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, category: str, defect_type: str) -> int:
    value = f"{seed}:{category}:{defect_type}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def split_category(
    dataset_root: Path, category: str, development_fraction: float, seed: int
) -> dict[str, object]:
    category_root = dataset_root / category
    test_root = category_root / "test"
    if not test_root.is_dir():
        raise RuntimeError(f"MVTec test split does not exist: {test_root}")
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(test_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            grouped[path.parent.name].append(path)
    if "good" not in grouped or len(grouped) < 2:
        raise RuntimeError(f"Category requires good and anomalous test images: {category}")

    records: list[dict[str, object]] = []
    counts: dict[str, dict[str, int]] = {}
    for defect_type, paths in sorted(grouped.items()):
        if len(paths) < 2:
            raise RuntimeError(
                f"Defect type requires at least two images for a split: {category}/{defect_type}"
            )
        shuffled = list(paths)
        random.Random(stable_seed(seed, category, defect_type)).shuffle(shuffled)
        development_count = min(
            len(paths) - 1, max(1, round(len(paths) * development_fraction))
        )
        development = set(shuffled[:development_count])
        counts[defect_type] = {
            "development": development_count,
            "heldout": len(paths) - development_count,
        }
        for path in paths:
            relative = path.relative_to(category_root)
            record: dict[str, object] = {
                "image_path": str(relative),
                "image_sha256": sha256(path),
                "defect_type": defect_type,
                "label": int(defect_type != "good"),
                "split": "development" if path in development else "heldout",
            }
            if defect_type != "good":
                mask = category_root / "ground_truth" / defect_type / f"{path.stem}_mask.png"
                if not mask.is_file():
                    raise RuntimeError(f"Ground-truth mask does not exist: {mask}")
                record["mask_path"] = str(mask.relative_to(category_root))
                record["mask_sha256"] = sha256(mask)
            records.append(record)

    fingerprint = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["image_path"])):
        fingerprint.update(json.dumps(record, sort_keys=True).encode("utf-8"))
        fingerprint.update(b"\n")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "MVTec AD",
        "category": category,
        "policy": "defect_type_stratified_deterministic",
        "seed": seed,
        "development_fraction": development_fraction,
        "test_set_used_for_model_training": False,
        "development_use": "configuration_selection_only",
        "heldout_use": "single_locked_final_evaluation",
        "record_count": len(records),
        "counts_by_defect_type": counts,
        "records_sha256": fingerprint.hexdigest(),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--development-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.development_fraction < 1.0:
        parser.error("development-fraction must be in (0, 1)")
    if not args.dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {args.dataset_root}")
    if len(set(args.categories)) != len(args.categories):
        parser.error("categories must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for category in args.categories:
        output = args.output_dir / f"{category}_dev_heldout_seed{args.seed}.json"
        if output.exists():
            parser.error(f"refusing to overwrite split manifest: {output}")
        document = split_category(
            args.dataset_root, category, args.development_fraction, args.seed
        )
        output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        written.append(output)
        print(f"split_manifest={output}")
    print(f"categories={len(written)}")


if __name__ == "__main__":
    main()
