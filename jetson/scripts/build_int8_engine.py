#!/usr/bin/env python3
"""Build a TensorRT INT8 engine with deterministic normal-image calibration."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import cv2
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


class NormalImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, images: list[Path], cache_path: Path, width: int, height: int) -> None:
        super().__init__()
        self.images = images
        self.cache_path = cache_path
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
        if self.index >= len(self.images):
            return None
        image = cv2.imread(str(self.images[self.index]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read calibration image: {self.images[self.index]}")
        self.index += 1
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        self.host[0] = np.transpose(image, (2, 0, 1))
        self.cuda.copy_to_device(self.device, self.host)
        return [int(self.device.value)]

    def read_calibration_cache(self) -> bytes | None:
        return self.cache_path.read_bytes() if self.cache_path.exists() else None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)


def image_paths(directory: Path, limit: int) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(path for path in directory.rglob("*") if path.suffix.lower() in extensions)
    if len(images) < limit:
        raise RuntimeError(f"Need {limit} calibration images, found {len(images)} in {directory}")
    return images[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.images <= 0 or args.width <= 0 or args.height <= 0 or args.workspace_gib <= 0:
        parser.error("numeric arguments must be positive")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(args.onnx)):
        errors = "\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors))
        raise RuntimeError(f"ONNX parsing failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 1024**3))
    calibrator = NormalImageCalibrator(
        image_paths(args.calibration_dir, args.images), args.cache, args.width, args.height
    )
    config.int8_calibrator = calibrator
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the INT8 engine")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(serialized)
    print(f"INT8 engine created: {args.engine}")


if __name__ == "__main__":
    main()
