#!/usr/bin/env python3
"""Build a TensorRT INT8 engine with deterministic normal-image calibration."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorrt as trt


class CudaRuntime:
    MEMCPY_HOST_TO_DEVICE = 1

    def __init__(self) -> None:
        self.library = ctypes.CDLL("libcudart.so")
        self.library.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.library.cudaFree.argtypes = [ctypes.c_void_p]
        self.library.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

    def check(self, status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"{operation} failed with CUDA error {status}")

    def malloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self.check(self.library.cudaMalloc(ctypes.byref(pointer), size), "cudaMalloc")
        return pointer

    def free(self, pointer: ctypes.c_void_p) -> None:
        self.check(self.library.cudaFree(pointer), "cudaFree")

    def copy_to_device(self, destination: ctypes.c_void_p, array: np.ndarray) -> None:
        source = ctypes.c_void_p(array.ctypes.data)
        self.check(
            self.library.cudaMemcpy(destination, source, array.nbytes, self.MEMCPY_HOST_TO_DEVICE),
            "cudaMemcpy",
        )


class NormalTensorCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        tensors: list[Path],
        cache_path: Path,
        width: int,
        height: int,
        reuse_cache: bool,
    ) -> None:
        super().__init__()
        self.tensors = tensors
        self.cache_path = cache_path
        self.reuse_cache = reuse_cache
        self.width = width
        self.height = height
        self.index = 0
        self.cuda = CudaRuntime()
        self.host = np.empty((1, 3, height, width), dtype=np.float32)
        self.device = self.cuda.malloc(self.host.nbytes)

    def __del__(self) -> None:
        device = getattr(self, "device", None)
        if device:
            self.cuda.free(device)
            self.device = None

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names: list[str]) -> list[int] | None:
        del names
        if self.index >= len(self.tensors):
            return None
        tensor_path = self.tensors[self.index]
        self.index += 1
        tensor = np.load(tensor_path, allow_pickle=False)
        if tensor.shape != self.host.shape or tensor.dtype != np.float32:
            raise RuntimeError(
                f"Calibration tensor must be FP32 {self.host.shape}, got "
                f"{tensor.dtype} {tensor.shape}: {tensor_path}"
            )
        if not np.isfinite(tensor).all():
            raise RuntimeError(f"Calibration tensor contains NaN or infinity: {tensor_path}")
        np.copyto(self.host, tensor)
        self.cuda.copy_to_device(self.device, self.host)
        return [int(self.device.value)]

    def read_calibration_cache(self) -> bytes | None:
        return self.cache_path.read_bytes() if self.reuse_cache and self.cache_path.exists() else None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_paths(directory: Path, limit: int, width: int, height: int) -> tuple[list[Path], Path]:
    manifest = directory / "calibration_manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"Calibration manifest not found: {manifest}")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("source_policy") != "train_normal_only":
        raise RuntimeError("Calibration manifest is not marked train_normal_only")
    if metadata.get("tensor_shape") != [1, 3, height, width]:
        raise RuntimeError(
            f"Calibration manifest tensor shape does not match [1, 3, {height}, {width}]"
        )
    entries = metadata.get("entries")
    if not isinstance(entries, list) or len(entries) < limit:
        raise RuntimeError(f"Calibration manifest contains fewer than {limit} entries")
    tensors: list[Path] = []
    for entry in entries[:limit]:
        tensor = directory / Path(entry["tensor_path"]).name
        if not tensor.is_file():
            raise RuntimeError(f"Calibration tensor not found: {tensor}")
        if sha256(tensor) != entry.get("tensor_sha256"):
            raise RuntimeError(f"Calibration tensor hash mismatch: {tensor}")
        tensors.append(tensor)
    return tensors, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--calibration-input-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--expected-output-channels", type=int, default=1536)
    parser.add_argument("--expected-output-height", type=int, default=32)
    parser.add_argument("--expected-output-width", type=int, default=32)
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    if (
        args.images <= 0
        or args.width <= 0
        or args.height <= 0
        or args.expected_output_channels <= 0
        or args.expected_output_height <= 0
        or args.expected_output_width <= 0
        or args.workspace_gib <= 0
    ):
        parser.error("numeric arguments must be positive")
    if not args.onnx.is_file():
        parser.error(f"ONNX model does not exist: {args.onnx}")
    if not args.export_manifest.is_file():
        parser.error(f"export manifest does not exist: {args.export_manifest}")
    existing_outputs = [path for path in (args.engine, args.manifest) if path.exists()]
    if existing_outputs:
        parser.error(
            "refusing to overwrite outputs: " + ", ".join(str(path) for path in existing_outputs)
        )
    if args.cache.exists() and not args.reuse_cache:
        parser.error(
            f"calibration cache already exists; use a new path or explicitly pass --reuse-cache: {args.cache}"
        )
    if args.reuse_cache and not args.cache.is_file():
        parser.error(f"--reuse-cache requires an existing cache file: {args.cache}")

    tensors, calibration_manifest = tensor_paths(
        args.calibration_input_dir,
        args.images,
        args.width,
        args.height,
    )

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(args.onnx)):
        errors = "\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors))
        raise RuntimeError(f"ONNX parsing failed:\n{errors}")
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError("ONNX network must have exactly one input and one output")
    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    expected_input_shape = (1, 3, args.height, args.width)
    expected_output_shape = (
        1,
        args.expected_output_channels,
        args.expected_output_height,
        args.expected_output_width,
    )
    if tuple(input_tensor.shape) != expected_input_shape:
        raise RuntimeError(
            f"ONNX input shape must be {expected_input_shape}, got {tuple(input_tensor.shape)}"
        )
    if tuple(output_tensor.shape) != expected_output_shape:
        raise RuntimeError(
            f"ONNX output shape must be {expected_output_shape}, got {tuple(output_tensor.shape)}"
        )
    if input_tensor.dtype != trt.float32 or output_tensor.dtype != trt.float32:
        raise RuntimeError("Runtime contract requires FP32 TensorRT input and output tensors")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 1024**3))
    calibrator = NormalTensorCalibrator(
        tensors,
        args.cache,
        args.width,
        args.height,
        args.reuse_cache,
    )
    config.int8_calibrator = calibrator
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the INT8 engine")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(serialized)
    external_data = args.onnx.with_name(args.onnx.name + ".data")
    build_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "precision": "int8",
        "tensorrt_version": str(trt.__version__),
        "artifacts": {
            "onnx": {"path": str(args.onnx.resolve()), "sha256": sha256(args.onnx)},
            "onnx_external_data": {
                "path": str(external_data.resolve()),
                "sha256": sha256(external_data),
            } if external_data.is_file() else None,
            "engine": {"path": str(args.engine.resolve()), "sha256": sha256(args.engine)},
            "export_manifest": {
                "path": str(args.export_manifest.resolve()),
                "sha256": sha256(args.export_manifest),
            },
        },
        "calibration": {
            "source_policy": "train_normal_only",
            "manifest_path": str(calibration_manifest.resolve()),
            "manifest_sha256": sha256(calibration_manifest),
            "tensor_count": len(tensors),
            "tensor_sha256": [sha256(path) for path in tensors],
            "cache_path": str(args.cache.resolve()),
            "cache_sha256": sha256(args.cache),
            "cache_reused": args.reuse_cache,
        },
        "builder": {
            "workspace_gib": args.workspace_gib,
            "input_width": args.width,
            "input_height": args.height,
            "input_shape": list(expected_input_shape),
            "output_shape": list(expected_output_shape),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"INT8 engine created: {args.engine}")
    print(f"Build manifest: {args.manifest}")


if __name__ == "__main__":
    main()
