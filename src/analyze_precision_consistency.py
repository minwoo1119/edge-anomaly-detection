#!/usr/bin/env python3
"""Aggregate FP32-vs-candidate runtime consistency metrics across images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REQUIRED_SUFFIXES = (
    "embedding",
    "patch_scores",
    "nn_indices",
    "raw_score",
    "anomaly_map",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prefixes(directory: Path) -> list[str]:
    suffix = "_embedding.npy"
    return sorted(path.name[: -len(suffix)] for path in directory.glob(f"*{suffix}"))


def load_pair(reference_dir: Path, candidate_dir: Path, prefix: str, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    reference_path = reference_dir / f"{prefix}_{suffix}.npy"
    candidate_path = candidate_dir / f"{prefix}_{suffix}.npy"
    for path in (reference_path, candidate_path):
        if not path.is_file():
            raise RuntimeError(f"required tensor does not exist: {path}")
    reference = np.load(reference_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"shape mismatch for {prefix}_{suffix}: {reference.shape} != {candidate.shape}"
        )
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise RuntimeError(f"non-finite tensor values in {prefix}_{suffix}")
    return reference, candidate


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:], values.size]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) * 0.5 + 1.0
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size < 2:
        raise ValueError("correlation requires equally sized arrays with at least two values")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        if np.array_equal(left, right):
            return 1.0
        raise ValueError("correlation is undefined for unequal constant arrays")
    return float(np.dot(left_centered, right_centered) / denominator)


def topk_overlap(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("top-k indices must have equal [query, k] shapes")
    overlaps = [
        len(set(reference[row].tolist()) & set(candidate[row].tolist())) / reference.shape[1]
        for row in range(reference.shape[0])
    ]
    return float(np.mean(overlaps))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-label", default="fp32")
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    reference_prefixes = prefixes(args.reference_dir)
    candidate_prefixes = prefixes(args.candidate_dir)
    if not reference_prefixes:
        parser.error(f"reference directory contains no embedding dumps: {args.reference_dir}")
    if reference_prefixes != candidate_prefixes:
        raise RuntimeError("reference and candidate dump prefixes differ")
    if args.manifest.exists():
        parser.error(f"refusing to overwrite manifest: {args.manifest}")

    embeddings_reference: list[np.ndarray] = []
    embeddings_candidate: list[np.ndarray] = []
    distances_reference: list[np.ndarray] = []
    distances_candidate: list[np.ndarray] = []
    scores_reference: list[float] = []
    scores_candidate: list[float] = []
    maps_reference: list[np.ndarray] = []
    maps_candidate: list[np.ndarray] = []
    matched_indices = 0
    index_count = 0
    topk_values: list[float] = []
    artifacts: list[dict[str, str]] = []

    for prefix in reference_prefixes:
        for suffix in REQUIRED_SUFFIXES:
            reference, candidate = load_pair(
                args.reference_dir, args.candidate_dir, prefix, suffix
            )
            if suffix == "embedding":
                embeddings_reference.append(reference.astype(np.float64, copy=False).reshape(-1))
                embeddings_candidate.append(candidate.astype(np.float64, copy=False).reshape(-1))
            elif suffix == "patch_scores":
                distances_reference.append(reference.astype(np.float64, copy=False).reshape(-1))
                distances_candidate.append(candidate.astype(np.float64, copy=False).reshape(-1))
            elif suffix == "nn_indices":
                matched_indices += int(np.count_nonzero(reference == candidate))
                index_count += reference.size
            elif suffix == "raw_score":
                scores_reference.append(float(reference.reshape(-1)[0]))
                scores_candidate.append(float(candidate.reshape(-1)[0]))
            elif suffix == "anomaly_map":
                maps_reference.append(reference.astype(np.float64, copy=False).reshape(-1))
                maps_candidate.append(candidate.astype(np.float64, copy=False).reshape(-1))
            for directory, role in (
                (args.reference_dir, "reference"),
                (args.candidate_dir, "candidate"),
            ):
                path = directory / f"{prefix}_{suffix}.npy"
                artifacts.append({"role": role, "path": str(path.resolve()), "sha256": sha256(path)})

        reference_topk = args.reference_dir / f"{prefix}_nn_topk_indices.npy"
        candidate_topk = args.candidate_dir / f"{prefix}_nn_topk_indices.npy"
        if reference_topk.is_file() != candidate_topk.is_file():
            raise RuntimeError(f"top-k dump availability differs for prefix={prefix}")
        if reference_topk.is_file():
            reference = np.load(reference_topk, allow_pickle=False)
            candidate = np.load(candidate_topk, allow_pickle=False)
            topk_values.append(topk_overlap(reference, candidate))
            for path, role in (
                (reference_topk, "reference"),
                (candidate_topk, "candidate"),
            ):
                artifacts.append(
                    {"role": role, "path": str(path.resolve()), "sha256": sha256(path)}
                )

    embedding_reference = np.concatenate(embeddings_reference)
    embedding_candidate = np.concatenate(embeddings_candidate)
    embedding_error = embedding_candidate - embedding_reference
    reference_norm = float(np.linalg.norm(embedding_reference))
    cosine_denominator = reference_norm * float(np.linalg.norm(embedding_candidate))
    distance_reference = np.concatenate(distances_reference)
    distance_candidate = np.concatenate(distances_candidate)
    score_reference = np.asarray(scores_reference)
    score_candidate = np.asarray(scores_candidate)
    map_error = np.concatenate(maps_candidate) - np.concatenate(maps_reference)
    row: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reference_label": args.reference_label,
        "candidate_label": args.candidate_label,
        "image_count": len(reference_prefixes),
        "embedding_mae": float(np.mean(np.abs(embedding_error))),
        "embedding_rmse": float(np.sqrt(np.mean(np.square(embedding_error)))),
        "embedding_max_error": float(np.max(np.abs(embedding_error))),
        "embedding_relative_l2": float(np.linalg.norm(embedding_error) / max(reference_norm, np.finfo(float).eps)),
        "embedding_cosine_similarity": float(
            np.dot(embedding_reference, embedding_candidate)
            / max(cosine_denominator, np.finfo(float).eps)
        ),
        "top1_nn_agreement": matched_indices / index_count,
        "topk_available": bool(topk_values),
        "topk_overlap": float(np.mean(topk_values)) if topk_values else "",
        "distance_pearson": correlation(distance_reference, distance_candidate),
        "distance_spearman": correlation(rankdata(distance_reference), rankdata(distance_candidate)),
        "score_mae": float(np.mean(np.abs(score_candidate - score_reference))),
        "score_pearson": correlation(score_reference, score_candidate),
        "score_spearman": correlation(rankdata(score_reference), rankdata(score_candidate)),
        "anomaly_map_mae": float(np.mean(np.abs(map_error))),
        "anomaly_map_rmse": float(np.sqrt(np.mean(np.square(map_error)))),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists() or args.output_csv.stat().st_size == 0
    if not write_header:
        with args.output_csv.open(newline="", encoding="utf-8") as stream:
            existing_header = next(csv.reader(stream), [])
        if existing_header != list(row):
            raise RuntimeError(f"CSV schema mismatch; use a new output: {args.output_csv}")
    with args.output_csv.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    manifest = {
        "schema_version": 1,
        "row": row,
        "prefixes": reference_prefixes,
        "artifacts": artifacts,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for key, value in row.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
