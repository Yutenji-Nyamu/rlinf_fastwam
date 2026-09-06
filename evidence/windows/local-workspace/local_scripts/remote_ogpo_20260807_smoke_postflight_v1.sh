set -euo pipefail

WORKTREE=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
RUN_ROOT=/root/autodl-tmp/experiments/ogpo_robotwin_smoke_20260807_v1
RUNTIME_ROOT=/root/autodl-tmp/experiment_exports/ogpo_robotwin_smoke_20260807_v1/runtime
PYTHON=/root/autodl-tmp/RLinf/.venv/bin/python
ANALYZER="$RUNTIME_ROOT/analyze_ogpo_smoke_resources.py"
EXPECTED_ANALYZER_SHA=7feaca97e54c2ff5f4783a58f1153b13147b77d4eb1f3bcf75aa36cc50b4aafc

test "$(cat "$RUNTIME_ROOT/exit_code.txt")" = 0
test -s "$RUNTIME_ROOT/resources_1s.csv"
test -f "$RUNTIME_ROOT/resource_monitor.log"
test -s "$RUN_ROOT/metrics.log"
test -x "$PYTHON"
test -s "$ANALYZER"
test "$(sha256sum "$ANALYZER" | awk '{print $1}')" = "$EXPECTED_ANALYZER_SHA"

driver_pid=$(cat "$RUNTIME_ROOT/driver_pid.txt")
monitor_pid=$(cat "$RUNTIME_ROOT/monitor_pid.txt")
if kill -0 "$driver_pid" 2>/dev/null || kill -0 "$monitor_pid" 2>/dev/null; then
  echo 'postflight refused: smoke process is still alive' >&2
  exit 2
fi
remaining_processes=$(pgrep -af 'robotwin_adjust_bottle_ogpo_openpi|ray::EnvWorker|ray::EmbodiedOGPOFSDPPolicy|ray::MultiStepRolloutWorker' || true)
if test -n "$remaining_processes"; then
  printf '%s\n' "$remaining_processes" >&2
  echo 'postflight refused: relevant process remains' >&2
  exit 3
fi

cd "$WORKTREE"
test "$(git rev-parse HEAD)" = 5d5c84e3ac4efa1713a4139a05ac1b776e634ed3
test "$(git branch --show-current)" = codex/ogpo-pi0-robotwin
test -z "$(git status --short)"

mapfile -t manifests < <(find "$RUN_ROOT" -type f -path '*/global_step_1/actor/ogpo_components/complete.json' -print)
test "${#manifests[@]}" -eq 1
manifest=${manifests[0]}
actor_root=$(dirname "$(dirname "$manifest")")

"$PYTHON" "$ANALYZER" "$RUNTIME_ROOT/resources_1s.csv" > "$RUNTIME_ROOT/resource_summary.json"

"$PYTHON" - "$manifest" > "$RUNTIME_ROOT/checkpoint_summary.json" <<'PY'
import gc
import json
import sys
from pathlib import Path

import torch

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["complete"] is True
assert manifest["step"] == 1
assert manifest["global_online_rows"] == 80
assert manifest["policy_version"] == 1
assert manifest["sidecars"] == ["rank_0.pt", "rank_1.pt"]
contract = manifest["contract"]
assert contract["world_size"] == 2
assert contract["replay_capacity"] == 80
assert contract["replay_capacity_per_rank"] == 40
assert contract["model_horizon"] == 50
assert contract["execution_horizon"] == 10
assert contract["candidate_group_size"] == 8

rank_summaries = []
for rank, name in enumerate(manifest["sidecars"]):
    path = manifest_path.parent / name
    assert path.is_file() and path.stat().st_size > 0
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    assert state["contract"] == contract
    assert state["snapshot_id"] == manifest["snapshot_id"]
    assert state["rank"] == rank
    assert state["step"] == 1
    assert state["global_online_rows"] == 80
    assert state["actor_updates"] == 1
    assert state["critic_updates"] == 1
    assert state["policy_version"] == 1
    assert state["pending_actor_updates"] == 0
    assert state["pending_critic_updates"] == 0
    replay = state["replay"]
    assert replay["metadata"]["capacity"] == 40
    assert replay["size"] == 40
    assert len(replay["slots"]) == 40
    assert sum(slot is not None for slot in replay["slots"]) == 40
    rank_summaries.append(
        {
            "rank": rank,
            "bytes": path.stat().st_size,
            "replay_size": replay["size"],
            "actor_updates": state["actor_updates"],
            "critic_updates": state["critic_updates"],
            "policy_version": state["policy_version"],
            "critic_feature_dim": state["critic_feature_dim"],
            "ema_tensor_count": len(state["actor_ema_shadow"]),
        }
    )
    del state, replay
    gc.collect()

print(
    json.dumps(
        {
            "manifest": str(manifest_path),
            "snapshot_id": manifest["snapshot_id"],
            "step": manifest["step"],
            "global_online_rows": manifest["global_online_rows"],
            "policy_version": manifest["policy_version"],
            "contract": contract,
            "ranks": rank_summaries,
        },
        indent=2,
        sort_keys=True,
    )
)
PY

echo '=== markers ==='
for name in started_at finished_at exit_code driver_pid monitor_pid; do
  printf '%s\t' "$name"
  cat "$RUNTIME_ROOT/$name.txt"
done
echo '=== resource summary ==='
cat "$RUNTIME_ROOT/resource_summary.json"
echo '=== checkpoint summary ==='
cat "$RUNTIME_ROOT/checkpoint_summary.json"
echo '=== actor checkpoint tree ==='
find "$actor_root" -maxdepth 5 -type f -printf '%s\t%p\n' | sort -n
echo '=== run tree ==='
find "$RUN_ROOT" -maxdepth 4 -type f -printf '%s\t%p\n' | sort -n
echo '=== final metrics ==='
tail -n 120 "$RUN_ROOT/metrics.log"
echo '=== terminal resources ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.events
echo '=== source ==='
git status --short
git rev-parse HEAD
git rev-list --left-right --count '@{upstream}...HEAD'
echo 'OGPO_ROBOTWIN_SMOKE_POSTFLIGHT_OK'
