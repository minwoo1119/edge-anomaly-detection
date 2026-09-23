#!/usr/bin/env python3
"""Train PatchCore on MVTec train-normal data and persist a verified checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import anomalib
import numpy as np
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from lightning.pytorch import seed_everything

from export_preprocessing_reference import positive, sha256


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def dataset_fingerprint(directory: Path) -> tuple[str, list[dict[str, object]]]:
    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No train-normal images found in {directory}")
    entries = [
        {
            "path": str(path.relative_to(directory)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(json.dumps(entry, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--coreset-ratio", type=float, default=0.10)
    parser.add_argument("--num-neighbors", type=positive, default=9)
    parser.add_argument("--input-width", type=positive, default=256)
    parser.add_argument("--input-height", type=positive, default=256)
    parser.add_argument("--train-batch-size", type=positive, default=32)
    parser.add_argument("--eval-batch-size", type=positive, default=32)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--devices", type=positive, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.coreset_ratio <= 1.0:
        parser.error("coreset-ratio must be in (0, 1]")
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        parser.error("GPU accelerator was requested but CUDA is not available")
    if not args.dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {args.dataset_root}")
    train_normal = args.dataset_root / args.category / "train" / "good"
    if not train_normal.is_dir():
        parser.error(f"MVTec train-normal directory does not exist: {train_normal}")
    if args.checkpoint.exists() or args.manifest.exists():
        parser.error("refusing to overwrite checkpoint or training manifest")
    if args.checkpoint.resolve() == args.manifest.resolve():
        parser.error("checkpoint and manifest paths must differ")

    dataset_sha256, dataset_entries = dataset_fingerprint(train_normal)
    seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    preprocessor = Patchcore.configure_pre_processor(
        image_size=(args.input_height, args.input_width),
        center_crop_size=(args.input_height, args.input_width),
    )
    model = Patchcore(
        backbone=args.backbone,
        layers=args.layers,
        pre_trained=True,
        coreset_sampling_ratio=args.coreset_ratio,
        num_neighbors=args.num_neighbors,
        pre_processor=preprocessor,
    )
    datamodule = MVTecAD(
        root=args.dataset_root,
        category=args.category,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    engine = Engine(
        accelerator=args.accelerator,
        devices=args.devices,
        deterministic=True,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    engine.fit(model=model, datamodule=datamodule)

    memory_bank = model.model.memory_bank.detach().cpu()
    if memory_bank.ndim != 2 or memory_bank.shape[0] == 0:
        raise RuntimeError(f"Training produced an invalid memory bank: {memory_bank.shape}")
    if not torch.isfinite(memory_bank).all():
        raise RuntimeError("Training produced a memory bank containing NaN or infinity")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    engine.trainer.save_checkpoint(str(args.checkpoint))

    reloaded = Patchcore.load_from_checkpoint(
        str(args.checkpoint),
        weights_only=False,
        map_location="cpu",
    )
    reloaded_bank = reloaded.model.memory_bank.detach().cpu()
    if not torch.equal(memory_bank, reloaded_bank):
        raise RuntimeError("Checkpoint reload changed the trained memory bank")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "train_normal_only",
        "test_set_used_for_training_or_tuning": False,
        "dataset": {
            "root": str(args.dataset_root.resolve()),
            "category": args.category,
            "train_normal_path": str(train_normal.resolve()),
            "train_normal_count": len(dataset_entries),
            "train_normal_fingerprint": dataset_sha256,
            "files": dataset_entries,
        },
        "model": {
            "name": "patchcore",
            "backbone": args.backbone,
            "layers": args.layers,
            "pre_trained": True,
            "coreset_ratio": args.coreset_ratio,
            "num_neighbors": args.num_neighbors,
            "input_size_hw": [args.input_height, args.input_width],
            "memory_bank_shape": list(memory_bank.shape),
            "memory_bank_dtype": str(memory_bank.dtype),
        },
        "runtime": {
            "accelerator": args.accelerator,
            "devices": args.devices,
            "seed": args.seed,
            "deterministic": True,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "anomalib_version": str(anomalib.__version__),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "size_bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint),
            "reload_memory_bank_exact": True,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"checkpoint={args.checkpoint}")
    print(f"manifest={args.manifest}")
    print(f"memory_bank_shape={tuple(memory_bank.shape)}")
    print(f"train_normal_fingerprint={dataset_sha256}")


if __name__ == "__main__":
    main()
