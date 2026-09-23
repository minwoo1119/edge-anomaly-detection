#!/usr/bin/env python3
"""Export the PatchCore feature boundary and memory bank reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import anomalib
import numpy as np
import onnx
import torch
import torch.nn.functional as functional
from anomalib.models import Patchcore

from export_preprocessing_reference import preprocess_image


class PatchcoreFeatureExtractor(torch.nn.Module):
    """Exportable equivalent of PatchcoreModel.generate_embedding()."""

    def __init__(self, patchcore_model: torch.nn.Module) -> None:
        super().__init__()
        self.feature_extractor = patchcore_model.feature_extractor
        self.feature_pooler = patchcore_model.feature_pooler
        self.layers = list(patchcore_model.layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(inputs)
        pooled = {
            layer: self.feature_pooler(features[layer])
            for layer in self.layers
        }
        embedding = pooled[self.layers[0]]
        for layer in self.layers[1:]:
            resized = functional.interpolate(
                pooled[layer],
                size=embedding.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            embedding = torch.cat((embedding, resized), dim=1)
        return embedding


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def output_artifacts(model_path: Path) -> list[Path]:
    model = onnx.load(str(model_path), load_external_data=False)
    paths = [model_path]
    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        entries = {entry.key: entry.value for entry in initializer.external_data}
        location = entries.get("location")
        if not location:
            raise RuntimeError(f"External initializer has no location: {initializer.name}")
        external_path = (model_path.parent / location).resolve()
        if external_path not in paths:
            paths.append(external_path)
    return paths


def deterministic_input(height: int, width: int, device: torch.device) -> torch.Tensor:
    elements = 3 * height * width
    tensor = torch.linspace(-2.0, 2.0, elements, dtype=torch.float32, device=device)
    return tensor.reshape(1, 3, height, width)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--memory-bank", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-image", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--input-width", type=positive, default=256)
    parser.add_argument("--input-height", type=positive, default=256)
    parser.add_argument("--expected-channels", type=positive, default=1536)
    parser.add_argument("--expected-feature-height", type=positive, default=32)
    parser.add_argument("--expected-feature-width", type=positive, default=32)
    parser.add_argument("--opset", type=positive)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--external-data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.validation_image is not None and not args.validation_image.is_file():
        parser.error(f"validation image does not exist: {args.validation_image}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if args.atol < 0 or args.rtol < 0 or not np.isfinite((args.atol, args.rtol)).all():
        parser.error("atol and rtol must be finite and non-negative")
    if args.onnx.suffix.lower() != ".onnx":
        parser.error("--onnx must use the .onnx extension")
    if args.memory_bank.suffix.lower() != ".npy":
        parser.error("--memory-bank must use the .npy extension")
    if args.manifest.suffix.lower() != ".json":
        parser.error("--manifest must use the .json extension")
    requested_outputs = (args.onnx, args.memory_bank, args.manifest)
    resolved_outputs = [path.resolve() for path in requested_outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        parser.error("ONNX, memory-bank, and manifest outputs must be different paths")
    known_external_data = args.onnx.with_name(args.onnx.name + ".data")
    protected_outputs = (*requested_outputs, known_external_data)
    existing = [path for path in protected_outputs if path.exists()]
    if existing:
        parser.error(
            "refusing to overwrite outputs: " + ", ".join(str(path) for path in existing)
        )

    device = torch.device(args.device)
    trained_model = Patchcore.load_from_checkpoint(
        str(args.checkpoint),
        weights_only=False,
        map_location=device,
    ).to(device)
    trained_model.eval()
    core = trained_model.model
    feature_model = PatchcoreFeatureExtractor(core).to(device).eval()

    memory_bank = core.memory_bank.detach().cpu().numpy()
    if memory_bank.ndim != 2 or memory_bank.shape[0] == 0:
        raise RuntimeError(f"Expected a non-empty 2D memory bank, got {memory_bank.shape}")
    if memory_bank.shape[1] != args.expected_channels:
        raise RuntimeError(
            f"Memory-bank dimension {memory_bank.shape[1]} does not match "
            f"expected embedding channels {args.expected_channels}"
        )
    memory_bank = np.ascontiguousarray(memory_bank, dtype=np.float32)
    if not np.isfinite(memory_bank).all():
        raise RuntimeError("Memory bank contains NaN or infinity")

    if args.validation_image is None:
        validation_input = deterministic_input(args.input_height, args.input_width, device)
        validation_source = "deterministic_linspace"
        validation_image_sha256 = None
    else:
        validation_input = preprocess_image(
            args.validation_image,
            args.input_width,
            args.input_height,
            args.input_width,
            args.input_height,
        ).to(device)
        validation_source = str(args.validation_image.resolve())
        validation_image_sha256 = sha256(args.validation_image)

    with torch.inference_mode():
        pytorch_embedding = feature_model(validation_input)
        core_features = core.feature_extractor(validation_input)
        core_features = {
            layer: core.feature_pooler(core_features[layer])
            for layer in core.layers
        }
        core_embedding = core.generate_embedding(core_features)
    wrapper_error = torch.max(torch.abs(pytorch_embedding - core_embedding)).item()
    if not torch.equal(pytorch_embedding, core_embedding):
        raise RuntimeError(
            "Export wrapper differs from PatchcoreModel.generate_embedding: "
            f"max_abs_error={wrapper_error:.9g}"
        )
    expected_shape = (
        1,
        args.expected_channels,
        args.expected_feature_height,
        args.expected_feature_width,
    )
    if tuple(pytorch_embedding.shape) != expected_shape:
        raise RuntimeError(
            f"Feature output contract mismatch: expected {expected_shape}, "
            f"got {tuple(pytorch_embedding.shape)}"
        )

    for path in requested_outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    export_options: dict[str, object] = {
        "input_names": ["input"],
        "output_names": ["embedding"],
        "dynamo": True,
        "external_data": args.external_data,
    }
    if args.opset is not None:
        export_options["opset_version"] = args.opset
    torch.onnx.export(
        feature_model,
        (validation_input,),
        str(args.onnx),
        **export_options,
    )

    onnx_model = onnx.load(str(args.onnx), load_external_data=True)
    onnx.checker.check_model(onnx_model)
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required for export validation") from error
    session = ort.InferenceSession(
        str(args.onnx),
        providers=["CPUExecutionProvider"],
    )
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise RuntimeError("Exported ONNX model must have exactly one input and one output")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    if input_name != "input" or output_name != "embedding":
        raise RuntimeError(
            f"Unexpected ONNX tensor names: input={input_name}, output={output_name}"
        )
    input_numpy = validation_input.detach().cpu().numpy().astype(np.float32, copy=False)
    onnx_embedding = session.run(["embedding"], {"input": input_numpy})[0]
    pytorch_numpy = pytorch_embedding.detach().cpu().numpy().astype(np.float32, copy=False)
    if onnx_embedding.shape != expected_shape:
        raise RuntimeError(
            f"ONNX output contract mismatch: expected {expected_shape}, got {onnx_embedding.shape}"
        )
    absolute_error = np.abs(pytorch_numpy - onnx_embedding)
    if not np.allclose(pytorch_numpy, onnx_embedding, atol=args.atol, rtol=args.rtol):
        raise RuntimeError(
            f"PyTorch/ONNX mismatch: max_abs_error={absolute_error.max():.9g}, "
            f"mean_abs_error={absolute_error.mean():.9g}"
        )

    np.save(args.memory_bank, memory_bank, allow_pickle=False)
    artifacts = output_artifacts(args.onnx)
    artifacts.append(args.memory_bank)
    model_opsets = {item.domain or "ai.onnx": item.version for item in onnx_model.opset_import}
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256(args.checkpoint),
        },
        "versions": {
            "anomalib": str(anomalib.__version__),
            "torch": str(torch.__version__),
            "onnx": str(onnx.__version__),
            "onnxruntime": str(ort.__version__),
        },
        "device": str(device),
        "validation_source": validation_source,
        "validation_image_sha256": validation_image_sha256,
        "input_shape": list(validation_input.shape),
        "embedding_shape": list(pytorch_embedding.shape),
        "memory_bank_shape": list(memory_bank.shape),
        "memory_bank_dtype": str(memory_bank.dtype),
        "layers": list(core.layers),
        "num_neighbors": int(core.num_neighbors),
        "opsets": model_opsets,
        "external_data": args.external_data,
        "validation": {
            "wrapper_core_max_abs_error": wrapper_error,
            "atol": args.atol,
            "rtol": args.rtol,
            "max_abs_error": float(absolute_error.max()),
            "mean_abs_error": float(absolute_error.mean()),
            "passed": True,
        },
        "artifacts": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"onnx={args.onnx}")
    print(f"memory_bank={args.memory_bank}")
    print(f"manifest={args.manifest}")
    print(f"embedding_shape={tuple(pytorch_embedding.shape)}")
    print(f"memory_bank_shape={memory_bank.shape}")
    print(f"max_abs_error={absolute_error.max():.9g}")


if __name__ == "__main__":
    main()
