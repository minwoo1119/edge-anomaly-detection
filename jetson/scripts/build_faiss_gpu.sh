#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
version=${1:-v1.15.1}
prefix=${2:-"$root/artifacts/faiss-${version#v}"}
work_root=${FAISS_BUILD_ROOT:-"$root/artifacts/faiss-build/${version#v}"}
source_dir="$work_root/source"
build_dir="$work_root/build"
cmake_target=3.31.6
cmake_module="$work_root/cmake-$cmake_target"

mkdir -p "$work_root" "$root/results/metadata"

version_ok=$(cmake --version 2>/dev/null | python3 -c '
import re, sys
m = re.search(r"([0-9]+)\.([0-9]+)", sys.stdin.read())
print("yes" if m and tuple(map(int, m.groups())) >= (3, 24) else "no")
')
if [[ "$version_ok" == yes ]]; then
  cmake_command=$(command -v cmake)
  cmake_python_path=
else
  if [[ ! -x "$cmake_module/bin/cmake" ]]; then
    python3 -m pip install --target "$cmake_module" "cmake==$cmake_target"
  fi
  cmake_command="$cmake_module/bin/cmake"
  cmake_python_path="$cmake_module"
fi

run_cmake() {
  if [[ -n "$cmake_python_path" ]]; then
    PYTHONPATH="$cmake_python_path" "$cmake_command" "$@"
  else
    "$cmake_command" "$@"
  fi
}

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --depth 1 --branch "$version" https://github.com/facebookresearch/faiss.git "$source_dir"
fi

run_cmake -S "$source_dir" -B "$build_dir" \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_CUVS=OFF \
  -DFAISS_ENABLE_SVS=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DFAISS_OPT_LEVEL=generic \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87
run_cmake --build "$build_dir" --target faiss -j"${FAISS_BUILD_JOBS:-2}"
run_cmake --install "$build_dir" --prefix "$prefix"
cmake_version=$(run_cmake --version | head -1)

python3 - "$root" "$source_dir" "$prefix" "$version" "$cmake_version" <<'PY'
import hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

root, source, prefix = map(Path, sys.argv[1:4])
version, cmake_version = sys.argv[4:]
library = prefix / "lib/libfaiss.so"
manifest = {
    "schema_version": 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "version": version,
    "source_commit": subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip(),
    "machine": platform.machine(),
    "cmake": cmake_version,
    "nvcc": subprocess.check_output(["nvcc", "--version"], text=True).splitlines()[-1],
    "cuda_architectures": [87],
    "faiss_enable_gpu": True,
    "faiss_enable_python": False,
    "install_prefix": str(prefix.resolve()),
    "library": str(library.resolve()),
    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
}
output = root / "results/metadata" / f"faiss_gpu_build_{version.removeprefix('v')}.json"
output.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"faiss_prefix={prefix}")
print(f"manifest={output}")
PY
