#!/usr/bin/env python3
"""Run traced runtime-accuracy and benchmark experiments from a JSON matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_GROUPS = {"baseline", "precision", "coreset", "nn_backend", "system", "multi_category"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def load_plan(path: Path) -> list[dict[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("experiments"), list):
        raise RuntimeError("Experiment matrix must use schema_version=1 and contain experiments")
    required = ("experiment_id", "config", "benchmark_image", "category")
    experiments: list[dict[str, object]] = []
    for index, raw in enumerate(document["experiments"]):
        if not isinstance(raw, dict) or any(not isinstance(raw.get(key), str) for key in required):
            raise RuntimeError(f"Invalid experiment entry at index {index}")
        groups = raw.get("groups")
        if not isinstance(groups, list) or not groups or any(
            not isinstance(group, str) or group not in ALLOWED_GROUPS for group in groups
        ):
            raise RuntimeError(f"Invalid experiment groups at index {index}")
        if len(groups) != len(set(groups)):
            raise RuntimeError(f"Duplicate experiment groups at index {index}")
        experiment: dict[str, object] = {key: raw[key] for key in required}
        experiment["groups"] = groups
        if not experiment["experiment_id"] or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in experiment["experiment_id"]
        ):
            raise RuntimeError(f"Invalid experiment_id: {experiment['experiment_id']}")
        experiments.append(experiment)
    identifiers = [item["experiment_id"] for item in experiments]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Experiment IDs must be unique")
    if not experiments:
        raise RuntimeError("Experiment matrix contains no experiments")
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--runtime-accuracy-csv", type=Path, default=Path("results/csv/runtime_accuracy.csv"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--independent-runs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.plan.is_file():
        parser.error(f"plan does not exist: {args.plan}")
    if args.independent_runs < 3:
        parser.error("publication experiment requires at least three independent runs")
    if args.manifest.exists() and not args.dry_run:
        parser.error(f"refusing to overwrite matrix manifest: {args.manifest}")
    repository = Path(__file__).resolve().parents[2]
    experiments = load_plan(args.plan)
    git_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if git_status and not args.dry_run:
        raise RuntimeError("Experiment matrix requires a clean git worktree")

    commands: list[list[str]] = []
    summaries_by_group: dict[str, list[Path]] = {}
    python = sys.executable
    for experiment in experiments:
        experiment_id = str(experiment["experiment_id"])
        config = (repository / str(experiment["config"])).resolve()
        image = (repository / str(experiment["benchmark_image"])).resolve()
        if not args.dry_run:
            for path in (config, image):
                if not path.is_file():
                    raise RuntimeError(f"experiment input does not exist: {path}")
        accuracy_command = [
            python,
            str(repository / "jetson" / "scripts" / "evaluate_runtime.py"),
            "--executable",
            str(repository / "jetson" / "build" / "edge_anomaly"),
            "--config",
            str(config),
            "--dataset-root",
            str(args.dataset_root.resolve()),
            "--category",
            str(experiment["category"]),
            "--work-dir",
            str(repository / "results" / "raw" / "runtime_accuracy" / experiment_id),
            "--output-csv",
            str(args.runtime_accuracy_csv.resolve()),
            "--predictions-csv",
            str(repository / "results" / "csv" / f"{experiment_id}_predictions.csv"),
            "--manifest",
            str(repository / "results" / "metadata" / f"{experiment_id}_accuracy.json"),
        ]
        benchmark_command = [
            "bash",
            str(repository / "jetson" / "scripts" / "run_experiment.sh"),
            str(config),
            str(image),
            experiment_id,
            str(args.independent_runs),
        ]
        commands.extend((accuracy_command, benchmark_command))
        run(accuracy_command, args.dry_run)
        run(benchmark_command, args.dry_run)
        summary = repository / "results" / "processed" / f"{experiment_id}_run_summary.csv"
        for group in experiment["groups"]:
            summaries_by_group.setdefault(str(group), []).append(summary)

    group_outputs: dict[str, str] = {}
    for group, summaries in summaries_by_group.items():
        output = repository / "results" / "processed" / f"{group}_results.csv"
        merge_command = [
            python,
            str(repository / "src" / "merge_experiment_results.py"),
        ]
        for summary in summaries:
            merge_command.extend(("--benchmark-summary", str(summary)))
        merge_command.extend(
            (
                "--runtime-accuracy",
                str(args.runtime_accuracy_csv.resolve()),
                "--output",
                str(output),
            )
        )
        commands.append(merge_command)
        run(merge_command, args.dry_run)
        group_outputs[group] = str(output)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "plan": {"path": str(args.plan.resolve()), "sha256": sha256(args.plan)},
        "dataset_root": str(args.dataset_root.resolve()),
        "independent_runs": args.independent_runs,
        "experiments": experiments,
        "commands": commands,
        "group_outputs": group_outputs,
    }
    if not args.dry_run:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"manifest={args.manifest}")


if __name__ == "__main__":
    main()
