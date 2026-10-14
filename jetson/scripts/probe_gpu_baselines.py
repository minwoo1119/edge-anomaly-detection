#!/usr/bin/env python3
"""Record whether exact GPU retrieval baselines are usable on this Jetson."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command_output(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, *command[1:]], text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        return None
    return (completed.stdout or completed.stderr).strip()


def existing(paths: list[str]) -> list[str]:
    return [path for path in paths if Path(path).exists()]


def probe_python_faiss() -> dict[str, object]:
    spec = importlib.util.find_spec("faiss")
    result: dict[str, object] = {"importable": spec is not None, "gpu_count": None}
    if spec is None:
        return result
    try:
        import faiss  # type: ignore

        result["version"] = getattr(faiss, "__version__", "unknown")
        result["gpu_api"] = hasattr(faiss, "StandardGpuResources")
        if hasattr(faiss, "get_num_gpus"):
            result["gpu_count"] = int(faiss.get_num_gpus())
    except Exception as error:  # Preserve the failure as evidence.
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    faiss_headers = existing([
        "/usr/include/faiss/IndexFlat.h",
        "/usr/local/include/faiss/IndexFlat.h",
        "/usr/include/faiss/gpu/GpuIndexFlat.h",
        "/usr/local/include/faiss/gpu/GpuIndexFlat.h",
    ])
    faiss_libraries = existing([
        "/usr/lib/aarch64-linux-gnu/libfaiss.so",
        "/usr/local/lib/libfaiss.so",
        "/usr/lib/aarch64-linux-gnu/libfaiss_gpu.so",
        "/usr/local/lib/libfaiss_gpu.so",
    ])
    nvcc = command_output(["nvcc", "--version"])
    cublas_headers = existing([
        "/usr/local/cuda/include/cublas_v2.h",
        "/usr/include/cublas_v2.h",
    ])
    cublas_libraries = existing([
        "/usr/local/cuda/lib64/libcublas.so",
        "/usr/lib/aarch64-linux-gnu/libcublas.so",
    ])
    python_faiss = probe_python_faiss()
    native_faiss_gpu = any("gpu/GpuIndexFlat" in path for path in faiss_headers)
    native_faiss = any(path.endswith("IndexFlat.h") for path in faiss_headers)

    report = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cuda": {
            "nvcc_available": nvcc is not None,
            "nvcc_version": nvcc,
        },
        "cublas": {
            "headers": cublas_headers,
            "libraries": cublas_libraries,
            "build_ready": bool(cublas_headers and cublas_libraries),
            "runtime_backend": "cublas",
        },
        "faiss": {
            "headers": faiss_headers,
            "libraries": faiss_libraries,
            "native_cpu_api_detected": native_faiss,
            "native_gpu_api_detected": native_faiss_gpu,
            "python": python_faiss,
            "gpu_baseline_ready": bool(
                native_faiss_gpu
                or (python_faiss.get("gpu_api") and (python_faiss.get("gpu_count") or 0) > 0)
            ),
        },
        "interpretation": (
            "gpu_baseline_ready=false means FAISS GPU is unavailable in the current "
            "environment; it does not prove that an ARM64 source build is impossible"
        ),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            parser.error(f"refusing to overwrite output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"output={args.output}")
    print(text, end="")


if __name__ == "__main__":
    main()
