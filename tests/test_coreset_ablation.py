from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_coreset_ablation", ROOT / "src" / "run_coreset_ablation.py"
)
assert SPEC and SPEC.loader
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


class CoresetAblationResumeTests(unittest.TestCase):
    def test_training_artifacts_are_reused_only_when_identity_and_hash_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.ckpt"
            manifest = root / "training.json"
            checkpoint.write_bytes(b"checkpoint")
            document = {
                "dataset": {"category": "bottle"},
                "model": {"coreset_ratio": 0.025},
                "runtime": {"seed": 42},
                "checkpoint": {"sha256": ablation.sha256(checkpoint)},
            }
            manifest.write_text(json.dumps(document), encoding="utf-8")

            self.assertTrue(
                ablation.reusable_training_artifacts(
                    checkpoint, manifest, "bottle", 0.025, 42
                )
            )
            self.assertFalse(
                ablation.reusable_training_artifacts(
                    checkpoint, manifest, "bottle", 0.05, 42
                )
            )
            checkpoint.write_bytes(b"changed")
            self.assertFalse(
                ablation.reusable_training_artifacts(
                    checkpoint, manifest, "bottle", 0.025, 42
                )
            )

    def test_complete_variant_requires_bundle_checksum_and_matching_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deployment = root / "deployment"
            deployment.mkdir()
            bundle = deployment / "bottle_artifacts.zip"
            bundle.write_bytes(b"bundle")
            bundle.with_suffix(".zip.sha256").write_text("digest\n", encoding="utf-8")
            marker = {
                "category": "bottle",
                "coreset_ratio": 0.1,
                "seed": 123,
                "bundle_sha256": ablation.sha256(bundle),
            }
            (root / "COMPLETE.json").write_text(json.dumps(marker), encoding="utf-8")

            self.assertTrue(ablation.completed_variant(root, "bottle", 0.1, 123))
            self.assertFalse(ablation.completed_variant(root, "bottle", 0.1, 42))


if __name__ == "__main__":
    unittest.main()
