#!/usr/bin/env python3
"""Build a flat JRTIP LaTeX submission bundle from frozen experiment results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, format="eps", bbox_inches="tight")
    plt.close(fig)
    pdf_path = path.with_suffix(".pdf")
    subprocess.run(
        [
            "gs", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dEPSCrop",
            "-dAutoRotatePages=/None", "-dCompatibilityLevel=1.5",
            "-sDEVICE=pdfwrite", f"-sOutputFile={pdf_path}", str(path),
        ],
        check=True,
        capture_output=True,
    )


def pipeline_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 1.6))
    labels = ["RGB input", "Preprocess", "TensorRT\nfeatures", "Exact GPU\nretrieval", "Score + map"]
    xs = range(len(labels))
    for x, label in zip(xs, labels):
        ax.text(x, 0, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "black"})
        if x:
            ax.annotate("", xy=(x - 0.28, 0), xytext=(x - 0.72, 0),
                        arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(-0.7, 0.7)
    ax.axis("off")
    save(fig, path)


def tradeoff_figure(path: Path) -> None:
    precision = read_csv(ROOT / "results/paper_mvtec15_complete/processed/precision_aggregate.csv")
    coreset = read_csv(ROOT / "results/paper_mvtec15_complete/processed/coreset_aggregate.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    axes[0].bar([row["precision"].upper() for row in precision], [float(row["total_mean"]) for row in precision], color="#4c78a8")
    axes[0].set_ylabel("End-to-end latency (ms)")
    axes[0].set_title("Feature precision")
    ax2 = axes[0].twinx()
    ax2.plot(range(len(precision)), [float(row["image_auroc"]) for row in precision], "o-", color="#e45756")
    ax2.set_ylabel("Image AUROC")
    axes[1].plot([100 * float(row["coreset_ratio"]) for row in coreset], [float(row["total_mean"]) for row in coreset], "o-", label="Latency")
    axes[1].set_xlabel("Coreset ratio (%)")
    axes[1].set_ylabel("End-to-end latency (ms)")
    axes[1].set_title("Coreset scaling")
    save(fig, path)


def backend_figure(path: Path) -> None:
    raw = read_csv(ROOT / "results/processed/revision_r1_d042_nn_backend_results.csv")
    order = ["cpu", "cuda", "cuda_warp", "cuda_tiled", "cuda_tiled_async", "cuda_tiled_async_transpose", "cublas"]
    rows = []
    for backend in order:
        group = [row for row in raw if row["nn_backend"] == backend]
        if not group:
            continue
        rows.append({
            "nn_backend": backend,
            "nn_mean": statistics.fmean(float(row["nn_mean"]) for row in group),
            "total_mean": statistics.fmean(float(row["total_mean"]) for row in group),
        })
    labels = [row["nn_backend"].replace("cuda_tiled_async_transpose", "async+T").replace("cuda_tiled_async", "async").replace("cuda_tiled", "tiled").replace("cuda_warp", "warp").replace("cuda", "direct") for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    x = range(len(rows))
    nn = [float(row["nn_mean"]) for row in rows]
    other = [float(row["total_mean"]) - value for row, value in zip(rows, nn)]
    ax.bar(x, nn, color="#4c78a8", label="NN search")
    ax.bar(x, other, bottom=nn, color="#72b7b2", label="Other stages")
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_xticks(list(x), labels, rotation=20, ha="right")
    ax.legend(frameon=False, ncol=2)
    save(fig, path)


def system_figure(path: Path) -> None:
    rows = read_csv(ROOT / "results/paper_mvtec15_complete/processed/system_aggregate.csv")
    rows.sort(key=lambda row: int(row["optimization_stage"][1:]))
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    stages = [row["optimization_stage"] for row in rows]
    ax.plot(stages, [float(row["pipeline_interval_mean"]) for row in rows], "o-", label="Pipeline interval")
    ax.plot(stages, [float(row["total_mean"]) for row in rows], "s--", label="Frame completion")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Optimization stage")
    ax.legend(frameon=False)
    save(fig, path)


def category_figure(path: Path) -> None:
    rows = read_csv(ROOT / "results/paper_mvtec15_complete/tables/table_5_category_accuracy.csv")
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.4), sharex=True)
    x = range(len(rows))
    axes[0].plot(x, [float(row["image_auroc"]) for row in rows], "o-", label="Image")
    axes[0].plot(x, [float(row["pixel_auroc"]) for row in rows], "s-", label="Pixel")
    axes[0].set_ylim(0.9, 1.005)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].bar(x, [float(row["total_mean"]) for row in rows], color="#4c78a8")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_xticks(list(x), [row["category"].replace("_", " ") for row in rows], rotation=55, ha="right", fontsize=7)
    save(fig, path)


def sustained_figure(path: Path) -> None:
    rows = read_csv(ROOT / "results/processed/r4_fixed_rate_sweep.csv")
    x = [float(row["arrival_period_ms"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.plot(x, [float(row["response_p99_ms"]) for row in rows], "o-", color="#4c78a8", label="Response p99")
    ax.plot(x, [float(row["response_max_ms"]) for row in rows], "s--", color="#f58518", label="Response max")
    ax.axhline(1100, color="#777777", linestyle=":", label="Deadline")
    ax.set_xlabel("Arrival period (ms)")
    ax.set_ylabel("Response time (ms)")
    ax2 = ax.twinx()
    ax2.plot(x, [100 * float(row["deadline_failure_rate_including_drops"]) for row in rows], "^-", color="#e45756", label="Failure rate")
    ax2.set_ylabel("Deadline failure rate (\%)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, ncol=2, fontsize=8)
    save(fig, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/submission_jrtip")
    args = parser.parse_args()
    if not args.template_zip.is_file():
        parser.error(f"template zip not found: {args.template_zip}")
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    with zipfile.ZipFile(args.template_zip) as archive:
        names = archive.namelist()
        for filename in ("svjour3.cls", "svglov3.clo", "spmpsci.bst"):
            source = next((name for name in names if name.endswith("/" + filename)), None)
            if source is None:
                raise RuntimeError(f"official template is missing {filename}")
            (args.output_dir / filename).write_bytes(archive.read(source))

    shutil.copy2(ROOT / "paper/jrtip_manuscript.tex", args.output_dir / "manuscript.tex")
    shutil.copy2(ROOT / "paper/references.bib", args.output_dir / "references.bib")
    shutil.copy2(ROOT / "paper/jrtip_cover_letter.md", args.output_dir / "cover_letter.md")
    shutil.copy2(ROOT / "paper/jrtip_submission_checklist.md", args.output_dir / "submission_checklist.md")
    pipeline_figure(args.output_dir / "Fig1.eps")
    tradeoff_figure(args.output_dir / "Fig2.eps")
    backend_figure(args.output_dir / "Fig3.eps")
    system_figure(args.output_dir / "Fig4.eps")
    category_figure(args.output_dir / "Fig5.eps")
    sustained_figure(args.output_dir / "Fig6.eps")

    source_names = [
        "manuscript.tex", "references.bib", "svjour3.cls", "svglov3.clo", "spmpsci.bst",
        "Fig1.eps", "Fig2.eps", "Fig3.eps", "Fig4.eps", "Fig5.eps", "Fig6.eps",
        "Fig1.pdf", "Fig2.pdf", "Fig3.pdf", "Fig4.pdf", "Fig5.pdf", "Fig6.pdf",
    ]
    archive_path = args.output_dir / "jrtip_submission_sources.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source_names:
            archive.write(args.output_dir / name, arcname=name)
    manifest = {
        "schema_version": 1,
        "official_template_sha256": sha256(args.template_zip),
        "files": [{"path": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in sorted(args.output_dir.iterdir()) if path.is_file()],
    }
    (args.output_dir / "submission_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"output_dir={args.output_dir}")
    print(f"source_archive={archive_path}")


if __name__ == "__main__":
    main()
