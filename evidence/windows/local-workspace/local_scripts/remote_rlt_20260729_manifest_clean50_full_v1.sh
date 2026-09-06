#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

REVISION=9dc9299c163db059931898a9f0852098a61155a1
DATASET_ROOT=/root/autodl-tmp/datasets/robotwin2
ZIP=${DATASET_ROOT}/source/${REVISION}/dataset/adjust_bottle/aloha-agilex_clean_50.zip
RAW=${DATASET_ROOT}/raw/${REVISION}/adjust_bottle/clean50-v1
INTERMEDIATE=${DATASET_ROOT}/intermediate/${REVISION}/adjust_bottle/pi0-aloha-clean50-v1
CANONICAL=${DATASET_ROOT}/canonical/pi0-aloha-clean50-v1
MANIFEST_PARENT=${DATASET_ROOT}/manifests
MANIFEST=${MANIFEST_PARENT}/pi0-aloha-clean50-v1.json
PROCESS_CONVERTER=/root/autodl-tmp/RoboTwin/policy/pi0/scripts/process_data.py
LEROBOT_CONVERTER=/root/autodl-tmp/RoboTwin/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
EXTRACT_WRAPPER=/root/autodl-tmp/tmp/rlt_extract_clean50_full_20260729_v1.sh
PROCESS_WRAPPER=/root/autodl-tmp/tmp/rlt_process_clean50_full_20260729_v1.sh
LEROBOT_WRAPPER=/root/autodl-tmp/tmp/rlt_lerobot_clean50_full_20260729_v1.sh
MANIFEST_WRAPPER=/root/autodl-tmp/tmp/rlt_manifest_clean50_full_20260729_v1.sh
LOCK=${DATASET_ROOT}/.pi0-aloha-clean50-v1.lock

mkdir -p "$(dirname "$LOCK")" "$MANIFEST_PARENT"
exec 9>"$LOCK"
flock -n 9

printf 'START\t%s\n' "$(date --iso-8601=seconds)"
test ! -e "$MANIFEST"
test ! -L "$MANIFEST"
for path in \
  "$ZIP" \
  "$RAW/rlt_raw_validation.json" \
  "$INTERMEDIATE/rlt_intermediate_validation.json" \
  "$CANONICAL/rlt_canonical_validation.json" \
  "$PROCESS_CONVERTER" \
  "$LEROBOT_CONVERTER" \
  "$EXTRACT_WRAPPER" \
  "$PROCESS_WRAPPER" \
  "$LEROBOT_WRAPPER" \
  "$MANIFEST_WRAPPER"; do
  test -f "$path"
done

STAGING=$(mktemp "${MANIFEST_PARENT}/.pi0-aloha-clean50-v1.manifest.XXXXXX")
ZIP_PATH="$ZIP" RAW_ROOT="$RAW" INTERMEDIATE_ROOT="$INTERMEDIATE" \
  CANONICAL_ROOT="$CANONICAL" MANIFEST_STAGING="$STAGING" REVISION="$REVISION" \
  PROCESS_CONVERTER="$PROCESS_CONVERTER" LEROBOT_CONVERTER="$LEROBOT_CONVERTER" \
  EXTRACT_WRAPPER="$EXTRACT_WRAPPER" PROCESS_WRAPPER="$PROCESS_WRAPPER" \
  LEROBOT_WRAPPER="$LEROBOT_WRAPPER" MANIFEST_WRAPPER="$MANIFEST_WRAPPER" \
  /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

zip_path = Path(os.environ["ZIP_PATH"])
raw = Path(os.environ["RAW_ROOT"])
intermediate = Path(os.environ["INTERMEDIATE_ROOT"])
canonical = Path(os.environ["CANONICAL_ROOT"])
staging = Path(os.environ["MANIFEST_STAGING"])
raw_validation = json.loads((raw / "rlt_raw_validation.json").read_text(encoding="utf-8"))
intermediate_validation = json.loads(
    (intermediate / "rlt_intermediate_validation.json").read_text(encoding="utf-8")
)
canonical_validation = json.loads(
    (canonical / "rlt_canonical_validation.json").read_text(encoding="utf-8")
)
expected_zip_size = 298_659_710
expected_zip_sha256 = "5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e"
actual_zip_sha256 = sha256(zip_path)
if zip_path.stat().st_size != expected_zip_size:
    raise ValueError(f"source ZIP size changed: {zip_path.stat().st_size}")
if actual_zip_sha256 != expected_zip_sha256:
    raise ValueError(f"source ZIP hash changed: {actual_zip_sha256}")
for payload in (raw_validation, intermediate_validation):
    if payload["episodes"] != 50 or payload["total_rows"] != canonical_validation["total_frames"]:
        raise ValueError("cross-stage episode/frame mismatch")
if intermediate_validation["qpos_raw_t_max_abs"] != 0.0:
    raise ValueError("qpos time contract mismatch")
if intermediate_validation["action_raw_t_plus_1_max_abs"] != 0.0:
    raise ValueError("action time contract mismatch")
if canonical_validation["episodes"] != 50 or canonical_validation["fps"] != 50:
    raise ValueError("canonical contract mismatch")

file_records = []
for path in sorted(item for item in canonical.rglob("*") if item.is_file()):
    file_records.append(
        {
            "path": path.relative_to(canonical).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )
file_list_payload = "".join(
    f"{item['sha256']}  {item['bytes']}  {item['path']}\n" for item in file_records
).encode()
file_list_digest = hashlib.sha256(file_list_payload).hexdigest()

robotwin_root = Path("/root/autodl-tmp/RoboTwin")
robotwin_head = subprocess.check_output(
    ["git", "-C", str(robotwin_root), "rev-parse", "HEAD"], text=True
).strip()
robotwin_status = subprocess.check_output(
    ["git", "-C", str(robotwin_root), "status", "--short"], text=True
).splitlines()

converter_paths = {
    "raw_to_aloha": Path(os.environ["PROCESS_CONVERTER"]),
    "aloha_to_lerobot": Path(os.environ["LEROBOT_CONVERTER"]),
}
wrapper_paths = {
    "extract": Path(os.environ["EXTRACT_WRAPPER"]),
    "process": Path(os.environ["PROCESS_WRAPPER"]),
    "lerobot": Path(os.environ["LEROBOT_WRAPPER"]),
    "manifest": Path(os.environ["MANIFEST_WRAPPER"]),
}
metadata_paths = {
    "info": canonical / "meta" / "info.json",
    "episodes": canonical / "meta" / "episodes.jsonl",
    "tasks": canonical / "meta" / "tasks.jsonl",
    "canonical_validation": canonical / "rlt_canonical_validation.json",
}

manifest = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_id": "pi0-aloha-clean50-v1",
    "task": "adjust_bottle",
    "budget_label": "single-task low-budget RLT port; not equal-scale ManiSkill reproduction",
    "source": {
        "repository": "TianxingChen/RoboTwin2.0",
        "revision": os.environ["REVISION"],
        "filename": "dataset/adjust_bottle/aloha-agilex_clean_50.zip",
        "archive_path": str(zip_path),
        "archive_bytes": zip_path.stat().st_size,
        "archive_sha256": actual_zip_sha256,
        "archive_entries": 207,
        "archive_file_entries": 202,
        "archive_uncompressed_bytes": 450_331_107,
        "episode_indices": list(range(50)),
    },
    "paths": {
        "raw": str(raw),
        "intermediate": str(intermediate),
        "canonical": str(canonical),
    },
    "conversion": {
        "seed": 0,
        "pythonhashseed": 0,
        "robotwin_git_head": robotwin_head,
        "robotwin_git_status": robotwin_status,
        "converters": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in converter_paths.items()
        },
        "wrappers": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in wrapper_paths.items()
        },
    },
    "contract": {
        "episodes": 50,
        "total_frames": canonical_validation["total_frames"],
        "per_episode": canonical_validation["per_episode"],
        "fps": 50,
        "state_dim": 14,
        "action_dim": 14,
        "camera_names": canonical_validation["camera_names"],
        "required_columns": canonical_validation["required_columns"],
        "qpos_raw_t_max_abs": intermediate_validation["qpos_raw_t_max_abs"],
        "action_raw_t_plus_1_max_abs": intermediate_validation[
            "action_raw_t_plus_1_max_abs"
        ],
        "converter_task_argument": "adjust the bottle",
        "default_prompt_fallback": "adjust the bottle",
        "canonical_task_records": canonical_validation["task_records"],
        "canonical_episode_metadata": canonical_validation["episode_metadata"],
    },
    "canonical_files": {
        "count": len(file_records),
        "total_bytes": sum(item["bytes"] for item in file_records),
        "sorted_sha256_size_path_digest": file_list_digest,
    },
    "metadata_sha256": {
        **{name: sha256(path) for name, path in metadata_paths.items()},
        "raw_validation": sha256(raw / "rlt_raw_validation.json"),
        "intermediate_validation": sha256(
            intermediate / "rlt_intermediate_validation.json"
        ),
    },
    "environment": {
        "python": platform.python_version(),
        "torch": version("torch"),
        "lerobot": version("lerobot"),
        "openpi_client": version("openpi-client"),
        "numpy": version("numpy"),
        "pyarrow": version("pyarrow"),
    },
}
staging.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            "episodes": manifest["contract"]["episodes"],
            "total_frames": manifest["contract"]["total_frames"],
            "canonical_files": manifest["canonical_files"],
            "converter_sha256": {
                key: value["sha256"]
                for key, value in manifest["conversion"]["converters"].items()
            },
        },
        indent=2,
        sort_keys=True,
    )
)
PY

test ! -e "$MANIFEST"
mv -T -- "$STAGING" "$MANIFEST"
printf 'MANIFEST_PROMOTED\t%s\n' "$MANIFEST"
sha256sum "$MANIFEST"
printf 'SUCCESS\t%s\n' "$(date --iso-8601=seconds)"
