#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 CONFIG IMAGE KERNEL_REGEX OUTPUT_DIR" >&2
  exit 2
fi

config=$1
image=$2
kernel_regex=$3
output_dir=$4
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
executable="$root/jetson/build/edge_anomaly"

for path in "$config" "$image" "$executable"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done
command -v nsys >/dev/null || { echo "ERROR: nsys is unavailable" >&2; exit 1; }
command -v ncu >/dev/null || { echo "ERROR: ncu is unavailable" >&2; exit 1; }
[[ ! -e "$output_dir" ]] || { echo "ERROR: refusing to overwrite $output_dir" >&2; exit 1; }
mkdir -p "$output_dir"

nsys profile --force-overwrite=false --trace=cuda,nvtx,osrt \
  --output "$output_dir/timeline" \
  "$executable" --config "$config" --image "$image"

ncu --kernel-name "regex:${kernel_regex}" --launch-count 1 --set full \
  --export "$output_dir/kernel" \
  "$executable" --config "$config" --image "$image"

python3 - "$root" "$config" "$image" "$kernel_regex" "$output_dir" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

root, config, image, kernel, output = map(Path, sys.argv[1:])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "schema_version": 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "config": str(config.resolve()), "config_sha256": sha(config),
    "image": str(image.resolve()), "image_sha256": sha(image),
    "kernel_regex": str(kernel),
    "nsys_report": str((output / "timeline.nsys-rep").resolve()),
    "ncu_report": str((output / "kernel.ncu-rep").resolve()),
}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

echo "profile_manifest=$output_dir/manifest.json"
