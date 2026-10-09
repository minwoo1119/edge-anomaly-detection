"""Regression tests for JRTIP benchmark result-flow processing."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


process_benchmarks = load_module(
    "process_benchmarks", ROOT / "src" / "process_benchmarks.py"
)
merge_experiment_results = load_module(
    "merge_experiment_results", ROOT / "src" / "merge_experiment_results.py"
)
generate_paper_results = load_module(
    "generate_paper_results", ROOT / "scripts" / "generate_paper_results.py"
)
run_coreset_ablation = load_module(
    "run_coreset_ablation", ROOT / "src" / "run_coreset_ablation.py"
)
prepare_colab_bundle = load_module(
    "prepare_colab_bundle", ROOT / "jetson" / "scripts" / "prepare_colab_bundle.py"
)
generate_coreset_plan = load_module(
    "generate_coreset_experiment_plan", ROOT / "src" / "generate_coreset_experiment_plan.py"
)
generate_system_configs = load_module(
    "generate_system_configs", ROOT / "src" / "generate_system_configs.py"
)
run_experiment_matrix = load_module(
    "run_experiment_matrix", ROOT / "jetson" / "scripts" / "run_experiment_matrix.py"
)
parse_tegrastats = load_module(
    "parse_tegrastats", ROOT / "jetson" / "scripts" / "parse_tegrastats.py"
)
validate_artifact_chain = load_module(
    "validate_artifact_chain",
    ROOT / "jetson" / "scripts" / "validate_artifact_chain.py",
)
apply_threshold_config = load_module(
    "apply_threshold_config", ROOT / "src" / "apply_threshold_config.py"
)
try:
    evaluate_runtime = load_module(
        "evaluate_runtime", ROOT / "jetson" / "scripts" / "evaluate_runtime.py"
    )
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "PIL"}:
        raise
    evaluate_runtime = None
try:
    compare_runtime = load_module(
        "compare_runtime", ROOT / "src" / "compare_runtime.py"
    )
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    compare_runtime = None
try:
    analyze_precision_consistency = load_module(
        "analyze_precision_consistency",
        ROOT / "src" / "analyze_precision_consistency.py",
    )
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    analyze_precision_consistency = None
try:
    calibrate_threshold = load_module(
        "calibrate_threshold", ROOT / "src" / "calibrate_threshold.py"
    )
except ModuleNotFoundError as error:
    if error.name not in {"anomalib", "numpy", "torch"}:
        raise
    calibrate_threshold = None


class TegrastatsTest(unittest.TestCase):
    def test_power_and_utilization_samples(self) -> None:
        log = (
            "RAM 100/1000MB CPU [10%@1000,20%@1000,off,off] "
            "GR3D_FREQ 40% VDD_IN 5000mW/6000mW GPU@45.5C\n"
            "RAM 120/1000MB CPU [30%@1000,10%@1000,20%@1000,off] "
            "GR3D_FREQ 60% VDD_IN 7000mW/8000mW CPU@50C\n"
        )
        power, temperature, gpu, cpu = parse_tegrastats.parse_tegrastats_lines(
            log.splitlines()
        )

        self.assertEqual(power, [5.0, 7.0])
        self.assertEqual(temperature, [45.5, 50.0])
        self.assertEqual(gpu, [40.0, 60.0])
        self.assertEqual(cpu, [7.5, 15.0])


class BenchmarkSummaryTest(unittest.TestCase):
    @staticmethod
    def row(iteration: int) -> dict[str, str]:
        row = {field: "fixed" for field in process_benchmarks.IDENTITY_FIELDS}
        row.update(
            {
                "run_id": "run-01",
                "timestamp": "2026-01-01T00:00:00Z",
                "iteration": str(iteration),
                "bank_entries": "100",
                "bank_size_mb": "1.5",
                "engine_size_mb": "2.5",
                "host_memory_mb": "32.0",
                "warmup_iterations": "50",
                "measurement_iterations": "2",
                "pipeline_priming_iterations": "0",
                "pipeline_interval_ms": "100",
            }
        )
        row.update({field: "10" for field in process_benchmarks.TIMING_FIELDS})
        row["total_ms"] = "100"
        row["fps"] = "10"
        return row

    def test_resource_fields_and_stage_shares(self) -> None:
        summary = process_benchmarks.summarize_run(
            [self.row(0), self.row(1)], None, None
        )

        self.assertEqual(summary["bank_entries"], "100")
        self.assertEqual(summary["bank_size_mb"], "1.5")
        self.assertEqual(summary["host_memory_mb"], "32.0")
        self.assertEqual(summary["embedding_transform_share_pct"], 10.0)
        self.assertEqual(summary["nn_share_pct"], 10.0)


@unittest.skipIf(evaluate_runtime is None, "NumPy/Pillow are not installed")
class RuntimeEvaluationTest(unittest.TestCase):
    def test_binary_auc_with_ties(self) -> None:
        labels = evaluate_runtime.np.asarray([False, True, False, True])
        scores = evaluate_runtime.np.asarray([0.0, 0.5, 0.5, 1.0])
        self.assertAlmostEqual(evaluate_runtime.binary_auc(labels, scores), 0.875)

    def test_binary_auc_rejects_single_class(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_runtime.binary_auc(
                evaluate_runtime.np.asarray([True, True]),
                evaluate_runtime.np.asarray([0.1, 0.2]),
            )


@unittest.skipIf(compare_runtime is None, "NumPy is not installed")
class RuntimeComparisonTest(unittest.TestCase):
    def test_tensor_statistics(self) -> None:
        values = compare_runtime.np.asarray([1.0, 2.0, 3.0])
        statistics = compare_runtime.tensor_statistics(values)
        self.assertEqual(statistics["min"], 1.0)
        self.assertEqual(statistics["max"], 3.0)
        self.assertEqual(statistics["mean"], 2.0)


@unittest.skipIf(analyze_precision_consistency is None, "NumPy is not installed")
class PrecisionConsistencyTest(unittest.TestCase):
    def test_rankdata_and_topk_overlap(self) -> None:
        np = analyze_precision_consistency.np
        ranks = analyze_precision_consistency.rankdata(np.asarray([3.0, 1.0, 1.0]))
        np.testing.assert_allclose(ranks, np.asarray([3.0, 1.5, 1.5]))
        overlap = analyze_precision_consistency.topk_overlap(
            np.asarray([[1, 2, 3], [4, 5, 6]]),
            np.asarray([[3, 2, 7], [4, 8, 9]]),
        )
        self.assertAlmostEqual(overlap, 0.5)


@unittest.skipIf(calibrate_threshold is None, "Threshold dependencies are not installed")
class ThresholdCalibrationTest(unittest.TestCase):
    def test_quantile_and_max_thresholds(self) -> None:
        scores = calibrate_threshold.np.asarray([0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(
            calibrate_threshold.select_threshold(scores, "quantile", 0.5), 1.5
        )
        self.assertEqual(
            calibrate_threshold.select_threshold(scores, "max", 0.5), 3.0
        )


class MergeExperimentResultsTest(unittest.TestCase):
    @staticmethod
    def benchmark() -> dict[str, str]:
        return {
            "config_sha256": "config",
            "engine_sha256": "engine",
            "memory_bank_sha256": "bank",
            "category": "bottle",
            "precision": "fp16",
            "coreset_ratio": "0.1",
            "bank_precision": "fp32",
            "nn_backend": "cpu",
            "optimization_stage": "S0",
            "total_mean": "10.0",
        }

    @staticmethod
    def accuracy() -> dict[str, str]:
        return {
            **MergeExperimentResultsTest.benchmark(),
            "evaluation_dataset_sha256": "dataset",
            "image_auroc": "0.99",
            "pixel_auroc": "0.98",
            "accuracy_source": "cpp_tensorrt_runtime",
            "threshold_tuned_on_test": "False",
        }

    def test_exact_artifact_join(self) -> None:
        merged = merge_experiment_results.merge_rows(
            [self.benchmark()], [self.accuracy()]
        )
        self.assertEqual(merged[0]["image_auroc"], "0.99")
        self.assertEqual(merged[0]["accuracy_source"], "cpp_tensorrt_runtime")

    def test_coreset_numeric_format_and_mismatch(self) -> None:
        accuracy = self.accuracy()
        accuracy["coreset_ratio"] = "0.10"
        self.assertEqual(len(merge_experiment_results.merge_rows([self.benchmark()], [accuracy])), 1)
        accuracy["coreset_ratio"] = "0.2"
        with self.assertRaises(RuntimeError):
            merge_experiment_results.merge_rows([self.benchmark()], [accuracy])

    def test_artifact_mismatch_is_rejected(self) -> None:
        accuracy = self.accuracy()
        accuracy["engine_sha256"] = "different"
        with self.assertRaises(RuntimeError):
            merge_experiment_results.merge_rows([self.benchmark()], [accuracy])


class PaperResultGenerationTest(unittest.TestCase):
    def test_pareto_frontier(self) -> None:
        points = [
            ("slow-accurate", 10.0, 0.99),
            ("fast-accurate", 5.0, 0.99),
            ("fast-less-accurate", 5.0, 0.95),
            ("fastest", 3.0, 0.90),
        ]
        self.assertEqual(
            generate_paper_results.pareto(points), {"fast-accurate", "fastest"}
        )

    def test_fixed_condition_violation_is_rejected(self) -> None:
        rows = [{"power_mode": "15W"}, {"power_mode": "MAXN"}]
        with self.assertRaises(RuntimeError):
            generate_paper_results.ensure_fixed(rows, ("power_mode",), "precision")


class CoresetAblationTest(unittest.TestCase):
    def test_ratio_tags_are_stable(self) -> None:
        self.assertEqual(
            [run_coreset_ablation.ratio_tag(value) for value in (0.01, 0.025, 0.05, 0.1, 0.2)],
            ["c001", "c0025", "c005", "c01", "c02"],
        )

    def test_jetson_ratio_tags_match_colab_tags(self) -> None:
        for ratio in (0.01, 0.025, 0.05, 0.1, 0.2):
            self.assertEqual(
                prepare_colab_bundle.ratio_tag(ratio),
                run_coreset_ablation.ratio_tag(ratio),
            )

    def test_coreset_plan_ratio_tags_are_stable(self) -> None:
        self.assertEqual(
            [generate_coreset_plan.ratio_tag(value) for value in generate_coreset_plan.EXPECTED_RATIOS],
            ["c001", "c0025", "c005", "c01", "c02"],
        )


class ArtifactLineageTest(unittest.TestCase):
    def test_matching_coreset_and_checkpoint_metadata_is_valid(self) -> None:
        validate_artifact_chain.validate_coreset_metadata(
            {
                "coreset_ratio": 0.1,
                "checkpoint": {"sha256": "checkpoint-hash"},
            },
            {
                "model": {"coreset_ratio": 0.1},
                "checkpoint": {"sha256": "checkpoint-hash"},
            },
            0.1,
        )

    def test_mismatched_coreset_ratio_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_artifact_chain.validate_coreset_metadata(
                {"coreset_ratio": 0.2},
                {},
                0.1,
            )

    def test_train_normal_threshold_metadata_is_valid(self) -> None:
        validate_artifact_chain.validate_threshold_metadata(
            {
                "test_set_used_for_threshold_tuning": False,
                "source": "train_normal",
                "threshold_space": "raw",
                "category": "bottle",
                "threshold": 1.25,
                "checkpoint": {"sha256": "checkpoint-hash"},
            },
            {"checkpoint": {"sha256": "checkpoint-hash"}},
            "bottle",
            1.25,
            "train_normal",
        )


class ApplyThresholdConfigTest(unittest.TestCase):
    def test_enables_decision_and_records_manifest(self) -> None:
        baseline = (
            "category: bottle\n"
            "decision_enabled: false\n"
            "threshold: 0.0\n"
            "threshold_space: raw\n"
            "threshold_source: unset\n"
        )
        updated = apply_threshold_config.apply_threshold(
            baseline, 1.25, Path("results/threshold_manifest.json")
        )
        self.assertIn("decision_enabled: true", updated)
        self.assertIn("threshold: 1.25", updated)
        self.assertIn("threshold_source: train_normal", updated)
        self.assertIn("threshold_manifest_path:", updated)


class SystemConfigGenerationTest(unittest.TestCase):
    def test_cuda_kernel_is_preserved_across_stages(self) -> None:
        for backend in ("cuda", "cuda_warp", "cuda_tiled", "cuda_tiled_cached", "cuda_tiled_async", "cuda_tiled_async_transpose"):
            for stage in ("S4", "S5", "S6"):
                with self.subTest(backend=backend, stage=stage):
                    baseline = f"optimization_stage: S0\nnn_backend: {backend}\n"
                    generated = generate_system_configs.with_stage(baseline, stage)
                    self.assertIn(f"nn_backend: {backend}\n", generated)

    def test_stage_replacement(self) -> None:
        baseline = "precision: fp32\noptimization_stage: S0\nnn_backend: cpu\n"
        generated = generate_system_configs.with_stage(baseline, "S3")
        self.assertIn("optimization_stage: S3\n", generated)

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_system_configs.with_stage("optimization_stage: S0\n", "S7")

    def test_s5_selects_cuda_backend(self) -> None:
        baseline = "optimization_stage: S0\nnn_backend: cpu\n"
        generated = generate_system_configs.with_stage(baseline, "S5")
        self.assertIn("optimization_stage: S5", generated)
        self.assertIn("nn_backend: cuda", generated)

    def test_s6_selects_cuda_backend(self) -> None:
        baseline = "optimization_stage: S0\nnn_backend: cpu\n"
        generated = generate_system_configs.with_stage(baseline, "S6")
        self.assertIn("optimization_stage: S6", generated)
        self.assertIn("nn_backend: cuda", generated)


class ExperimentMatrixTest(unittest.TestCase):
    def test_example_matrix_is_valid(self) -> None:
        experiments = run_experiment_matrix.load_plan(
            ROOT / "configs" / "experiment_matrix.example.json"
        )
        self.assertEqual(len(experiments), 2)
        self.assertIn("baseline", experiments[0]["groups"])
        self.assertIn("system", experiments[0]["groups"])

    def test_resume_output_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = tuple(root / name for name in ("work", "predictions", "manifest", "summary"))
            self.assertEqual(run_experiment_matrix.completed_output_state(paths), "absent")
            paths[0].mkdir()
            self.assertEqual(run_experiment_matrix.completed_output_state(paths), "absent")
            (paths[0] / "map.npy").write_bytes(b"data")
            self.assertEqual(run_experiment_matrix.completed_output_state(paths), "partial")
            for path in paths[1:]:
                path.write_text("complete", encoding="utf-8")
            self.assertEqual(run_experiment_matrix.completed_output_state(paths), "complete")


if __name__ == "__main__":
    unittest.main()
