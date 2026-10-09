#!/usr/bin/env python3
"""Generate JRTIP tables and data-driven SVG figures from merged result CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "image_auroc",
    "pixel_auroc",
    "preprocess_mean",
    "h2d_mean",
    "trt_mean",
    "d2h_mean",
    "embedding_transform_mean",
    "nn_mean",
    "post_mean",
    "total_mean",
    "total_p95",
    "pipeline_interval_mean",
    "pipeline_interval_p95",
    "fps_mean",
    "bank_size_mb",
    "engine_size_mb",
    "host_memory_mb",
    "avg_power_w",
    "peak_power_w",
    "energy_per_image_mj",
    "fps_per_w",
)
STAGES = (
    ("Preprocess", "preprocess_mean"),
    ("H2D", "h2d_mean"),
    ("TensorRT", "trt_mean"),
    ("D2H", "d2h_mean"),
    ("Embedding", "embedding_transform_mean"),
    ("NN", "nn_mean"),
    ("Postprocess", "post_mean"),
)
MVTEC_CATEGORIES = {
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
}
ENVIRONMENT_FIELDS = (
    "device", "jetpack", "cuda", "tensorrt", "opencv", "compiler", "build_type", "power_mode"
)
COLORS = ("#386cb0", "#fdb462", "#7fc97f", "#ef3b2c", "#beaed4", "#fdc086", "#666666")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"CSV contains no data rows: {path}")
    return rows


def require_fields(rows: list[dict[str, str]], fields: tuple[str, ...], name: str) -> None:
    missing = [field for field in fields if field not in rows[0]]
    if missing:
        raise RuntimeError(f"{name} is missing columns: {', '.join(missing)}")


def ensure_fixed(rows: list[dict[str, str]], fields: tuple[str, ...], name: str) -> None:
    changed = [field for field in fields if len({row[field] for row in rows}) != 1]
    if changed:
        raise RuntimeError(
            f"{name} violates fixed-condition comparison: {', '.join(changed)}"
        )


def aggregate(rows: list[dict[str, str]], key: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    result: list[dict[str, object]] = []
    for value, members in grouped.items():
        item: dict[str, object] = {key: value, "independent_runs": len(members)}
        for metric in METRICS:
            values = [float(member[metric]) for member in members if member.get(metric, "") != ""]
            item[metric] = statistics.fmean(values) if values else ""
            if metric == "total_mean" and values:
                item["total_run_std"] = statistics.pstdev(values)
        if "bank_entries" in members[0]:
            entries = [float(member["bank_entries"]) for member in members]
            item["bank_entries"] = statistics.fmean(entries)
        result.append(item)
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        cells = []
        for field in fields:
            value = row[field]
            cells.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_document(title: str, body: str, width: int = 900, height: int = 540) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" '
        f'font-size="20">{html.escape(title)}</text>\n{body}\n</svg>\n'
    )


def bar_svg(
    path: Path,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float]]],
    y_label: str,
    *,
    stacked: bool = False,
) -> None:
    width, height = 900, 540
    left, top, plot_width, plot_height = 90, 65, 760, 390
    totals = [sum(values[index] for _, values in series) for index in range(len(labels))]
    maximum = max(totals if stacked else [value for _, values in series for value in values])
    maximum = maximum if maximum > 0 else 1.0
    body = [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" '
        f'text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>',
    ]
    group_width = plot_width / max(len(labels), 1)
    for label_index, label in enumerate(labels):
        base_y = top + plot_height
        for series_index, (name, values) in enumerate(series):
            value = values[label_index]
            bar_height = value / maximum * plot_height
            if stacked:
                x = left + label_index * group_width + group_width * 0.2
                bar_width = group_width * 0.6
                y = base_y - bar_height
                base_y = y
            else:
                bar_width = group_width * 0.7 / len(series)
                x = left + label_index * group_width + group_width * 0.15 + series_index * bar_width
                y = top + plot_height - bar_height
            body.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
                f'fill="{COLORS[series_index % len(COLORS)]}"><title>{html.escape(name)}: {value:.6g}</title></rect>'
            )
        body.append(
            f'<text x="{left + (label_index + 0.5) * group_width:.2f}" y="{top + plot_height + 22}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(label)}</text>'
        )
    for index, (name, _) in enumerate(series):
        x = left + index * 125
        body.append(f'<rect x="{x}" y="490" width="14" height="14" fill="{COLORS[index % len(COLORS)]}"/>')
        body.append(f'<text x="{x + 20}" y="502" font-family="sans-serif" font-size="12">{html.escape(name)}</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_document(title, "\n".join(body), width, height), encoding="utf-8")


def scatter_svg(
    path: Path,
    title: str,
    points: list[tuple[str, float, float, bool]],
    x_label: str,
    y_label: str,
) -> None:
    width, height = 900, 540
    left, top, plot_width, plot_height = 90, 65, 760, 390
    xs = [point[1] for point in points]
    ys = [point[2] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = (x_max - x_min) * 0.08 or 1.0
    y_pad = (y_max - y_min) * 0.08 or 0.01
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad
    body = [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<text x="{left + plot_width / 2}" y="515" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(x_label)}</text>',
        f'<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>',
    ]
    for label, x_value, y_value, highlighted in points:
        x = left + (x_value - x_min) / (x_max - x_min) * plot_width
        y = top + (y_max - y_value) / (y_max - y_min) * plot_height
        color = "#ef3b2c" if highlighted else "#386cb0"
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{7 if highlighted else 5}" fill="{color}"/>')
        body.append(f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" font-family="sans-serif" font-size="11">{html.escape(label)}</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_document(title, "\n".join(body), width, height), encoding="utf-8")


def panel_bars_svg(
    path: Path,
    title: str,
    labels: list[str],
    panels: list[tuple[str, list[float]]],
) -> None:
    width, height = 1050, 500
    panel_width = 300
    plot_height = 320
    body: list[str] = []
    for panel_index, (panel_title, values) in enumerate(panels):
        left = 60 + panel_index * 340
        top = 75
        maximum = max(values) if max(values) > 0 else 1.0
        body.append(f'<text x="{left + panel_width / 2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(panel_title)}</text>')
        body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>')
        body.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + panel_width}" y2="{top + plot_height}" stroke="black"/>')
        slot = panel_width / len(labels)
        for index, (label, value) in enumerate(zip(labels, values)):
            bar_height = value / maximum * plot_height
            x = left + index * slot + slot * 0.2
            y = top + plot_height - bar_height
            body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{slot * 0.6:.2f}" height="{bar_height:.2f}" fill="{COLORS[panel_index]}"><title>{value:.6g}</title></rect>')
            body.append(f'<text x="{left + (index + 0.5) * slot:.2f}" y="{top + plot_height + 20}" text-anchor="middle" font-family="sans-serif" font-size="10">{html.escape(label)}</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_document(title, "\n".join(body), width, height), encoding="utf-8")


def check_coverage(found: set[str], expected: set[str], name: str, allow: bool) -> list[str]:
    missing = sorted(expected - found)
    if missing and not allow:
        raise RuntimeError(f"{name} is incomplete; missing: {', '.join(missing)}")
    return missing


def pareto(points: list[tuple[str, float, float]]) -> set[str]:
    frontier: set[str] = set()
    for name, latency, accuracy in points:
        dominated = any(
            other_latency <= latency
            and other_accuracy >= accuracy
            and (other_latency < latency or other_accuracy > accuracy)
            for other_name, other_latency, other_accuracy in points
            if other_name != name
        )
        if not dominated:
            frontier.add(name)
    return frontier


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "baseline", "precision", "bank_storage", "coreset", "nn_backend", "system",
        "multi_category",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    inputs = {
        name: getattr(args, name)
        for name in (
            "baseline", "precision", "bank_storage", "coreset", "nn_backend", "system",
            "multi_category",
        )
        if getattr(args, name) is not None
    }
    if not inputs:
        parser.error("at least one experiment CSV is required")
    rows_by_name = {name: read_rows(path) for name, path in inputs.items()}
    for name, rows in rows_by_name.items():
        require_fields(
            rows,
            (
                "category", "precision", "coreset_ratio", "bank_precision",
                "nn_backend", "optimization_stage", "image_auroc", "pixel_auroc",
                "total_mean", "accuracy_source", "threshold_tuned_on_test",
                "sample_count", "warmup_iterations", "measurement_iterations",
                *ENVIRONMENT_FIELDS,
            ),
            name,
        )
        invalid_accuracy = [
            row for row in rows
            if row["accuracy_source"] != "cpp_tensorrt_runtime"
            or row["threshold_tuned_on_test"].lower() != "false"
        ]
        if invalid_accuracy:
            raise RuntimeError(f"{name} contains non-runtime or test-tuned accuracy")
        if not args.allow_incomplete and any(int(row["sample_count"]) < 200 for row in rows):
            raise RuntimeError(f"{name} contains runs with fewer than 200 measurements")
        if not args.allow_incomplete and any(int(row["warmup_iterations"]) < 50 for row in rows):
            raise RuntimeError(f"{name} contains runs with fewer than 50 warm-up iterations")
        if any(row["sample_count"] != row["measurement_iterations"] for row in rows):
            raise RuntimeError(f"{name} contains incomplete measurement rows")

    figures = args.output_root / "figures"
    tables = args.output_root / "tables"
    processed = args.output_root / "processed"
    coverage: dict[str, object] = {
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()},
        "allow_incomplete": args.allow_incomplete,
        "missing": {},
    }
    pareto_points: list[tuple[str, float, float]] = []

    if "baseline" in rows_by_name:
        rows = rows_by_name["baseline"]
        ensure_fixed(
            rows,
            ("precision", "coreset_ratio", "bank_precision", "nn_backend", "optimization_stage", *ENVIRONMENT_FIELDS),
            "baseline",
        )
        if rows[0]["optimization_stage"] != "S0":
            raise RuntimeError("baseline must use optimization_stage=S0")
        aggregated = aggregate(rows, "category")
        if not args.allow_incomplete and any(int(row["independent_runs"]) < 3 for row in aggregated):
            raise RuntimeError("baseline requires at least three independent runs per category")
        bar_svg(
            figures / "figure_3_stage_latency.svg",
            "Stage-Level Latency Breakdown",
            [str(row["category"]) for row in aggregated],
            [(label, [float(row[field]) for row in aggregated]) for label, field in STAGES],
            "Latency (ms)",
            stacked=True,
        )
        environments = [{field: row[field] for field in ENVIRONMENT_FIELDS} for row in rows]
        unique_environments = list({tuple(item.items()): item for item in environments}.values())
        write_csv(tables / "table_1_environment.csv", unique_environments)
        write_markdown(tables / "table_1_environment.md", unique_environments)

    analyses = (
        ("precision", "precision", {"fp32", "fp16", "int8"}, ("category", "coreset_ratio", "bank_precision", "nn_backend", "optimization_stage", "power_mode")),
        ("bank_storage", "bank_precision", {"fp32", "fp16"}, ("category", "precision", "coreset_ratio", "nn_backend", "optimization_stage", "power_mode")),
        ("coreset", "coreset_ratio", {"0.01", "0.025", "0.05", "0.1", "0.10", "0.2", "0.20"}, ("category", "precision", "bank_precision", "nn_backend", "optimization_stage", "power_mode")),
        ("nn_backend", "nn_backend", {"cpu", "cuda"}, ("category", "precision", "coreset_ratio", "bank_precision", "optimization_stage", "power_mode")),
        ("system", "optimization_stage", {f"S{index}" for index in range(7)}, ("category", "precision", "coreset_ratio", "bank_precision", "power_mode")),
    )
    table_numbers = {"precision": 2, "bank_storage": "2b", "coreset": 3, "nn_backend": 4, "system": 6}
    for name, key, expected, fixed in analyses:
        if name not in rows_by_name:
            continue
        rows = rows_by_name[name]
        ensure_fixed(rows, (*fixed, *ENVIRONMENT_FIELDS), name)
        aggregated = aggregate(rows, key)
        if not args.allow_incomplete and any(int(row["independent_runs"]) < 3 for row in aggregated):
            raise RuntimeError(f"{name} requires at least three independent runs per configuration")
        found = {str(row[key]) for row in aggregated}
        if name == "coreset":
            found_numeric = {f"{float(value):g}" for value in found}
            expected_numeric = {"0.01", "0.025", "0.05", "0.1", "0.2"}
            coverage["missing"][name] = check_coverage(found_numeric, expected_numeric, name, args.allow_incomplete)
            aggregated.sort(key=lambda row: float(row[key]))
        else:
            coverage["missing"][name] = check_coverage(found, expected, name, args.allow_incomplete)
            aggregated.sort(key=lambda row: str(row[key]))
        if name == "precision":
            order = {"fp32": 0, "fp16": 1, "int8": 2}
            aggregated.sort(key=lambda row: order.get(str(row[key]), 99))
        write_csv(processed / f"{name}_aggregate.csv", aggregated)
        write_csv(tables / f"table_{table_numbers[name]}_{name}.csv", aggregated)
        write_markdown(tables / f"table_{table_numbers[name]}_{name}.md", aggregated)
        for row in aggregated:
            latency_field = "pipeline_interval_mean" if name == "system" else "total_mean"
            pareto_points.append(
                (f"{name}:{row[key]}", float(row[latency_field]), float(row["image_auroc"]))
            )

        labels = [str(row[key]) for row in aggregated]
        if name == "precision":
            scatter_svg(
                figures / "figure_4_precision_tradeoff.svg",
                "Precision Accuracy-Latency Trade-off",
                [(label, float(row["total_mean"]), float(row["image_auroc"]), False) for label, row in zip(labels, aggregated)],
                "End-to-end latency (ms)",
                "Image AUROC",
            )
            has_power = all(
                row["avg_power_w"] != "" and row["energy_per_image_mj"] != ""
                for row in aggregated
            )
            if not has_power and not args.allow_incomplete:
                raise RuntimeError("precision results are missing power/energy measurements")
            if has_power:
                panel_bars_svg(
                    figures / "figure_9_power_energy.svg",
                    "Power and Energy Comparison",
                    labels,
                    [
                        ("Average power (W)", [float(row["avg_power_w"]) for row in aggregated]),
                        ("Energy (mJ/image)", [float(row["energy_per_image_mj"]) for row in aggregated]),
                    ],
                )
        elif name == "coreset":
            panel_bars_svg(
                figures / "figure_5_coreset_tradeoff.svg",
                "Coreset Accuracy, Latency, and Memory",
                labels,
                [
                    ("Image AUROC", [float(row["image_auroc"]) for row in aggregated]),
                    ("Total latency (ms)", [float(row["total_mean"]) for row in aggregated]),
                    ("Bank size (MB)", [float(row["bank_size_mb"]) for row in aggregated]),
                ],
            )
        elif name == "nn_backend":
            bar_svg(
                figures / "figure_6_nn_backend.svg",
                "Nearest-Neighbor Backend Comparison",
                labels,
                [
                    ("NN latency", [float(row["nn_mean"]) for row in aggregated]),
                    ("Total latency", [float(row["total_mean"]) for row in aggregated]),
                ],
                "Latency (ms)",
            )
        elif name == "system":
            bar_svg(
                figures / "figure_7_system_optimization.svg",
                "System Optimization Steady-State Interval",
                labels,
                [("Pipeline interval", [float(row["pipeline_interval_mean"]) for row in aggregated])],
                "Interval (ms/image)",
            )

    if "multi_category" in rows_by_name:
        rows = rows_by_name["multi_category"]
        ensure_fixed(
            rows,
            ("precision", "coreset_ratio", "bank_precision", "nn_backend", "optimization_stage", *ENVIRONMENT_FIELDS),
            "multi_category",
        )
        aggregated = aggregate(rows, "category")
        if not args.allow_incomplete and any(int(row["independent_runs"]) < 3 for row in aggregated):
            raise RuntimeError("multi_category requires at least three independent runs per category")
        aggregated.sort(key=lambda row: str(row["category"]))
        found = {str(row["category"]) for row in aggregated}
        coverage["missing"]["multi_category"] = check_coverage(found, MVTEC_CATEGORIES, "multi_category", args.allow_incomplete)
        write_csv(tables / "table_5_category_accuracy.csv", aggregated)
        write_markdown(tables / "table_5_category_accuracy.md", aggregated)

    if pareto_points:
        frontier = pareto(pareto_points)
        scatter_svg(
            figures / "figure_8_pareto_frontier.svg",
            "Accuracy-Latency Pareto Frontier",
            [(name, latency, accuracy, name in frontier) for name, latency, accuracy in pareto_points],
            "End-to-end latency (ms)",
            "Image AUROC",
        )
        write_csv(
            processed / "pareto_frontier.csv",
            [
                {"configuration": name, "total_latency_ms": latency, "image_auroc": accuracy, "pareto_optimal": name in frontier}
                for name, latency, accuracy in pareto_points
            ],
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "result_coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"output_root={args.output_root}")


if __name__ == "__main__":
    main()
