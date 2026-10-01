"""Regression tests for JRTIP benchmark result-flow processing."""

from __future__ import annotations

import importlib.util
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
parse_tegrastats = load_module(
    "parse_tegrastats", ROOT / "jetson" / "scripts" / "parse_tegrastats.py"
)
try:
    evaluate_runtime = load_module(
        "evaluate_runtime", ROOT / "jetson" / "scripts" / "evaluate_runtime.py"
    )
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "PIL"}:
        raise
    evaluate_runtime = None


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


if __name__ == "__main__":
    unittest.main()
