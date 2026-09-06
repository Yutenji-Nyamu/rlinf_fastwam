set -euo pipefail

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin

ROOT=/data/chenyiteng/projects/robotwin-native/RoboTwin/data/sz_collect_smoke_1ep_20260821
RUN=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1

test -d "$ROOT"
mapfile -t h5_files < <(find "$ROOT" -type f -name '*.hdf5' | sort)
mapfile -t mp4_files < <(find "$ROOT" -type f -name '*.mp4' | sort)
mapfile -t json_files < <(find "$ROOT" -type f -name '*.json' | sort)
test "${#h5_files[@]}" -ge 1
test "${#mp4_files[@]}" -ge 1
test "${#json_files[@]}" -ge 1

export SMOKE_H5="${h5_files[0]}"
export SMOKE_JSON="${json_files[0]}"
{
  printf 'root=%s\n' "$ROOT"
  du -sh "$ROOT"
  find "$ROOT" -type f -printf '%s\t%p\n' | sort
  find "$ROOT" -type f \( -name '*.hdf5' -o -name '*.mp4' -o -name '*.json' -o -name 'seed.txt' \) -exec sha256sum {} +
  python3 - <<'PY'
import json
import os

import h5py

h5_path = os.environ["SMOKE_H5"]
with h5py.File(h5_path, "r") as handle:
    rows = []
    handle.visititems(
        lambda name, obj: rows.append((name, tuple(obj.shape), str(obj.dtype)))
        if isinstance(obj, h5py.Dataset)
        else None
    )
print("hdf5=", h5_path)
for row in rows:
    print("dataset=", row)

json_path = os.environ["SMOKE_JSON"]
with open(json_path, encoding="utf-8") as handle:
    payload = json.load(handle)
print("json=", json_path, "type=", type(payload).__name__)
PY
  ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 "${mp4_files[0]}"
} | tee "$RUN/03_collect_manifest.txt"
