from pathlib import Path
import shlex

src = Path("tmp_grpo_ppo_checkpoint_target_inventory.tsv")
targets: list[tuple[int, str]] = []
for line in src.read_text(encoding="utf-8").splitlines():
    if not line.startswith("TARGET|"):
        continue
    _, size, path = line.split("|", 2)
    if not (
        path.startswith("/data/chenyiteng/results/rlinf-shenzhen/grpo/")
        or path.startswith("/data/chenyiteng/results/rlinf-shenzhen/ppo/")
    ):
        raise SystemExit(f"outside allowed roots: {path}")
    if "/checkpoints/global_step_" not in path:
        raise SystemExit(f"not a checkpoint payload: {path}")
    targets.append((int(size), path))

if len(targets) != 130:
    raise SystemExit(f"expected 130 targets, got {len(targets)}")

lines = [
    "set -eu",
    "# Generated from a live read-only inventory. Exact-file deletion only.",
    "# Scope excludes /data/chenyiteng/results/rlinf-shenzhen/pi05.",
]
for _, path in targets:
    q = shlex.quote(path)
    lines.append(f"test -f {q}")
    lines.append(f"rm -- {q}")
lines.append("df -h /data")
Path("tmp_grpo_ppo_delete_top2_files.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(targets), sum(size for size, _ in targets))
