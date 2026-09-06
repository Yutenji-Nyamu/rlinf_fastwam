from __future__ import annotations

from pathlib import Path
import shlex


INVENTORY = Path("tmp_grpo_ppo_checkpoint_target_inventory.tsv")
OUTPUT = Path("tmp_grpo_ppo_delete_top2_files_v2.sh")
EXCLUDED_RUN = (
    "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/"
    "dvac-global-z-formal100-4gpu32x8-g8-phys2367-v2/"
)


def q(value: str) -> str:
    return shlex.quote(value)


targets: list[tuple[int, str]] = []
keeps: list[tuple[str, int, int]] = []
for line in INVENTORY.read_text(encoding="utf-8").splitlines():
    fields = line.split("|")
    if line.startswith("TARGET|"):
        size = int(fields[1])
        path = fields[2]
        if EXCLUDED_RUN not in path:
            targets.append((size, path))
    elif line.startswith("CK|") and "|KEEP|" in line:
        ckpt_path = fields[-1]
        file_count = int(fields[-2].split("=", 1)[1])
        gpu_count = 4 if "4gpu" in fields[2] else 2
        keeps.append((ckpt_path, file_count, gpu_count))

if len(targets) != 128:
    raise SystemExit(f"expected 128 targets, got {len(targets)}")
if sum(size for size, _ in targets) != 903_430_988_085:
    raise SystemExit("unexpected target byte total")
if len(keeps) != 14:
    raise SystemExit(f"expected 14 keep checkpoints, got {len(keeps)}")
if len({path for _, path in targets}) != len(targets):
    raise SystemExit("duplicate delete target")

keep_files: list[str] = []
for ckpt, file_count, gpu_count in keeps:
    actor = f"{ckpt}/actor"
    local_shard = "localshard" in ckpt
    keep_files.append(f"{actor}/model_state_dict/full_weights.pt")
    if local_shard:
        keep_files.extend(
            [
                f"{actor}/local_shard_checkpoint/checkpoint_rank_0.pt",
                f"{actor}/local_shard_checkpoint/checkpoint_rank_1.pt",
            ]
        )
        base_count = 3
    else:
        keep_files.append(f"{actor}/dcp_checkpoint/.metadata")
        keep_files.extend(
            f"{actor}/dcp_checkpoint/__{rank}_0.distcp" for rank in range(gpu_count)
        )
        base_count = gpu_count + 2
    sidecar_count = file_count - base_count
    if sidecar_count:
        if sidecar_count != gpu_count:
            raise SystemExit(f"unexpected sidecar count for {ckpt}: {sidecar_count}")
        keep_files.extend(
            f"{actor}/dvac_state_rank{rank:04d}.json" for rank in range(gpu_count)
        )

if len(set(keep_files)) != len(keep_files):
    raise SystemExit("duplicate keep file")
target_paths = {path for _, path in targets}
if target_paths.intersection(keep_files):
    raise SystemExit("delete target overlaps protected keep file")

lines = [
    "set -eu",
    "GRPO_PREFIX=/data/chenyiteng/results/rlinf-shenzhen/grpo/",
    "PPO_PREFIX=/data/chenyiteng/results/rlinf-shenzhen/ppo/",
    "EXPECTED_TARGETS=128",
    "EXPECTED_BYTES=903430988085",
    "echo DF_BEFORE",
    "df -B1 /data",
    "precheck_target() {",
    "  expected_size=$1",
    "  path=$2",
    "  if [ \"${path#\"$GRPO_PREFIX\"}\" = \"$path\" ] && [ \"${path#\"$PPO_PREFIX\"}\" = \"$path\" ]; then",
    "    echo \"BAD_PREFIX|$path\" >&2",
    "    exit 1",
    "  fi",
    "  if [ ! -f \"$path\" ] || [ -L \"$path\" ]; then",
    "    echo \"NOT_REGULAR_FILE|$path\" >&2",
    "    exit 1",
    "  fi",
    "  actual_size=$(stat -c %s -- \"$path\")",
    "  if [ \"$actual_size\" -ne \"$expected_size\" ]; then",
    "    echo \"SIZE_MISMATCH|expected=$expected_size|actual=$actual_size|$path\" >&2",
    "    exit 1",
    "  fi",
    "}",
    "check_keep_files() {",
    "  phase=$1",
    "  shift",
    "  for path in \"$@\"; do",
    "    if [ ! -f \"$path\" ] || [ -L \"$path\" ]; then",
    "      echo \"KEEP_FILE_MISSING|phase=$phase|$path\" >&2",
    "      exit 1",
    "    fi",
    "  done",
    "  echo \"KEEP_CHECK_OK|phase=$phase|files=$#\"",
    "}",
    "",
    "# Phase 1: validate every delete target before any deletion.",
]
for size, path in targets:
    lines.append(f"precheck_target {size} {q(path)}")
lines.extend(
    [
        "",
        "# Protect every key payload in all 14 retained latest checkpoints.",
        "check_keep_files before \\",
    ]
)
for index, path in enumerate(keep_files):
    suffix = " \\" if index + 1 < len(keep_files) else ""
    lines.append(f"  {q(path)}{suffix}")
lines.extend(
    [
        "echo \"PRECHECK_OK|targets=$EXPECTED_TARGETS|bytes=$EXPECTED_BYTES\"",
        "",
        "# Phase 2: exact-file deletion only. No glob, recursion, or directory deletion.",
    ]
)
for _, path in targets:
    lines.append(f"rm -- {q(path)}")
lines.extend(
    [
        "echo \"DELETE_OK|files=$EXPECTED_TARGETS|expected_bytes=$EXPECTED_BYTES\"",
        "check_keep_files after \\",
    ]
)
for index, path in enumerate(keep_files):
    suffix = " \\" if index + 1 < len(keep_files) else ""
    lines.append(f"  {q(path)}{suffix}")
lines.extend(["echo DF_AFTER", "df -B1 /data"])

OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
phase2_index = lines.index(
    "# Phase 2: exact-file deletion only. No glob, recursion, or directory deletion."
)
Path("tmp_grpo_ppo_delete_top2_files_v2_preflight_only.sh").write_text(
    "\n".join(lines[:phase2_index] + ["echo PREFLIGHT_ONLY_OK"]) + "\n",
    encoding="utf-8",
)
Path("tmp_grpo_ppo_delete_top2_files_v2_bash_n.sh").write_text(
    "bash -n /dev/stdin <<'CHECKPOINT_CLEANUP_SCRIPT'\n"
    + "\n".join(lines)
    + "\nCHECKPOINT_CLEANUP_SCRIPT\n",
    encoding="utf-8",
)
print(f"targets={len(targets)} bytes={sum(size for size, _ in targets)} keeps={len(keeps)} keep_files={len(keep_files)}")
