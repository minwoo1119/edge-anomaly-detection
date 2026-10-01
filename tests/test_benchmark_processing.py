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
parse_tegrastats = load_module(
    "parse_tegrastats", ROOT / "jetson" / "scripts" / "parse_tegrastats.py"
)


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


if __name__ == "__main__":
    unittest.main()
