from __future__ import annotations

from pathlib import Path
import shlex


source = Path("tmp_grpo_ppo_delete_top2_files_v2.sh").read_text(encoding="utf-8").splitlines()
targets = [shlex.split(line)[2] for line in source if line.startswith("rm -- ")]
after_index = source.index("check_keep_files after \\")
df_after_index = source.index("echo DF_AFTER", after_index)
keep_files = []
for line in source[after_index + 1 : df_after_index]:
    stripped = line.strip()
    if stripped.endswith("\\"):
        stripped = stripped[:-1].rstrip()
    if stripped:
        keep_files.append(shlex.split(stripped)[0])

if len(targets) != 128 or len(keep_files) != 82:
    raise SystemExit(f"unexpected counts targets={len(targets)} keep_files={len(keep_files)}")

lines = [
    "set -eu",
    "while IFS='|' read -r expected_size path; do",
    "  [ -n \"$path\" ] || continue",
    "  if [ -e \"$path\" ]; then echo \"TARGET_STILL_EXISTS|$path\" >&2; exit 1; fi",
    "done <<'DELETE_TARGETS'",
]
lines.extend(f"{size}|{path}" for size, path in zip([0] * len(targets), targets))
lines.extend(
    [
        "DELETE_TARGETS",
        "echo DELETE_TARGETS_ABSENT_OK=128",
        "while IFS= read -r path; do",
        "  [ -n \"$path\" ] || continue",
        "  if [ ! -f \"$path\" ] || [ -L \"$path\" ]; then echo \"KEEP_FILE_MISSING|$path\" >&2; exit 1; fi",
        "done <<'KEEP_FILES'",
    ]
)
lines.extend(keep_files)
lines.extend(
    [
        "KEEP_FILES",
        "echo KEEP_FILES_PRESENT_OK=82",
        "echo DF_VERIFY",
        "df -B1 /data",
        "echo PI05_GPU_PROCESSES",
        "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv,noheader",
        "echo PI05_PROCESS_MATCHES",
        r"ps -eo pid=,ppid=,user=,etimes=,args= --sort=pid | grep -E 'pi05|pi0\.5|sz-pi05-robotwin-rl|formal100-2gpu64x4-g8-b512-u5-m5' | grep -v grep || true",
        "echo PI05_RECENT_FILES",
        "for root in \\",
        "  /data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1 \\",
        "  /data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys67-localshard-v1 \\",
        "  /data/chenyiteng/results/rlinf-shenzhen/pi05/packets/pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1 \\",
        "  /data/chenyiteng/results/rlinf-shenzhen/pi05/packets/pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys67-localshard-v1; do",
        "  [ -d \"$root\" ] || continue",
        "  echo \"ROOT|$root\"",
        "  find \"$root\" -xdev -type f -mmin -180 -printf '%T@|%s|%p\\n' | sort -nr | head -n 12",
        "done",
        "echo PI05_FATAL_SCAN",
        "for root in \\",
        "  /data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1 \\",
        "  /data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys67-localshard-v1 \\",
        "  /data/chenyiteng/results/rlinf-shenzhen/pi05/packets/pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1 \\",
        "  /data/chenyiteng/results/rlinf-shenzhen/pi05/packets/pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys67-localshard-v1; do",
        "  [ -d \"$root\" ] || continue",
        "  find \"$root\" -xdev -type f -mmin -180 \\( -name '*.log' -o -name '*.out' -o -name '*.txt' \\) -print0 | xargs -0 -r grep -Ein 'Traceback|CUDA out of memory|OutOfMemory|FATAL|nonfinite|worker[^[:space:]]* died|actor died' || true",
        "done",
    ]
)

Path("tmp_grpo_ppo_cleanup_v2_verify_remote.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"targets={len(targets)} keep_files={len(keep_files)}")
