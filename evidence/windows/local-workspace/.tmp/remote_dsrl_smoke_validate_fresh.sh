set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
PY=/root/autodl-tmp/RLinf/.venv/bin/python
RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
CKPT1="$RUN_ROOT/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke/checkpoints/global_step_1"
FRESH_PID=$(cat "$RUN_ROOT/fresh.pid")

if kill -0 "$FRESH_PID" 2>/dev/null; then
  echo "FRESH_STILL_RUNNING=1"
  exit 41
fi

echo "== artifact contract =="
test -s "$RUN_ROOT/fresh_driver.log"
test -s "$RUN_ROOT/fresh_command.txt"
test -s "$RUN_ROOT/resource_monitor/fresh/resources.csv"
test -s "$RUN_ROOT/resource_monitor/fresh/peak.txt"
test -s "$RUN_ROOT/FRESH_SMOKE_VALIDATED_RESOLVED_20260728.yaml"
test -s "$RUN_ROOT/run_provenance.txt"

test -s "$CKPT1/actor/local_shard_checkpoint/checkpoint_rank_0.pt"
test -s "$CKPT1/actor/local_shard_checkpoint/checkpoint_rank_1.pt"
test -s "$CKPT1/actor/sac_components/alpha/dcp_checkpoint/.metadata"
test -n "$(find "$CKPT1/actor/sac_components/alpha/dcp_checkpoint" -maxdepth 1 -type f -name '*.distcp' -print -quit)"
test -s "$CKPT1/actor/sac_components/target_model/checkpoint_rank_0.pt"
test -s "$CKPT1/actor/sac_components/target_model/checkpoint_rank_1.pt"
test -s "$CKPT1/actor/sac_components/dsrl_trainer_state_rank_0.pt"
test -s "$CKPT1/actor/sac_components/dsrl_trainer_state_rank_1.pt"
test -s "$CKPT1/actor/sac_components/replay_buffer/rank_0/dsrl_transition_replay.pt"
test -s "$CKPT1/actor/sac_components/replay_buffer/rank_1/dsrl_transition_replay.pt"
test -z "$(find "$CKPT1" -type f -name '*.tmp' -print -quit)"

find "$CKPT1" -type f -printf '%s %p\n' | sort -n
du -sh "$CKPT1"

cat /sys/fs/cgroup/memory.events > "$RUN_ROOT/resource_monitor/fresh/memory.events.final.txt"
awk '$1 ~ /^(anon|file|shmem|file_mapped|inactive_file|active_file|kernel_stack|pagetables|slab)$/ {print}' \
  /sys/fs/cgroup/memory.stat > "$RUN_ROOT/resource_monitor/fresh/memory.stat.final.txt"

echo "== trainer/replay/tensorboard/resource validation =="
export CKPT1 RUN_ROOT
export PYTHONDONTWRITEBYTECODE=1
"$PY" -B - <<'PY'
from __future__ import annotations

import csv
import gc
import hashlib
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from tensorboard.backend.event_processing import event_accumulator

ckpt = Path(os.environ["CKPT1"]) / "actor"
run_root = Path(os.environ["RUN_ROOT"])


def shadow_summary(path: Path, rank: int) -> dict[str, object]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "rank",
        "world_size",
        "update_step",
        "policy_phase",
        "pending_local_new_transitions",
        "flat_replay",
        "target_shadow_f32",
    }
    assert set(state) == required, (set(state), required)
    assert state["schema_version"] == 1
    assert state["rank"] == rank
    assert state["world_size"] == 2
    assert state["policy_phase"] == 1
    assert state["pending_local_new_transitions"] == 0
    assert state["flat_replay"] is True
    shadows = state["target_shadow_f32"]
    assert shadows
    assert all(
        any(
            component in name.split(".")
            for component in ("critic_image_encoder", "critic_state_encoder", "q_head")
        )
        for name in shadows
    )
    assert all(
        all(
            component not in name.split(".")
            for component in ("dsrl_noise_actor", "paligemma", "gemma_expert")
        )
        for name in shadows
    )
    digest = hashlib.sha256()
    total_numel = 0
    total_bytes = 0
    l2_sq = 0.0
    absmax = 0.0
    for name in sorted(shadows):
        value = shadows[name]
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
        total_numel += value.numel()
        total_bytes += value.numel() * value.element_size()
        l2_sq += value.double().square().sum().item()
        absmax = max(absmax, value.abs().max().item())
    summary = {
        "rank": rank,
        "update_step": int(state["update_step"]),
        "shadow_tensors": len(shadows),
        "shadow_numel": total_numel,
        "shadow_bytes": total_bytes,
        "shadow_l2": math.sqrt(l2_sq),
        "shadow_absmax": absmax,
        "shadow_sha256": digest.hexdigest(),
    }
    del state, shadows
    gc.collect()
    return summary


def tree_leaves(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value):
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from tree_leaves(value[key], name)
    else:
        raise TypeError((prefix, type(value)))


def replay_summary(path: Path, rank: int) -> dict[str, object]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["schema_version"] == 1
    assert state["global_capacity"] == 25000
    assert state["local_capacity"] == 12500
    assert state["rank"] == rank
    assert state["world_size"] == 2
    resident = int(state["resident_size"])
    inserted = int(state["total_inserted"])
    cursor = int(state["write_cursor"])
    assert 0 < resident <= 12500
    assert inserted >= resident
    assert cursor == inserted % 12500
    assert isinstance(state["rng_state"], torch.Tensor)
    storage = state["storage"]
    assert set(storage) == {
        "curr_obs",
        "next_obs",
        "actions",
        "rewards",
        "continuations",
        "terminations",
        "truncations",
        "discounts",
    }
    allocated = 0
    resident_bytes = 0
    leaves = []
    scalar_uniques = {}
    for name, value in tree_leaves(storage):
        assert value.device.type == "cpu"
        assert value.shape[0] == 12500
        allocated += value.numel() * value.element_size()
        resident_bytes += value[:resident].numel() * value.element_size()
        if value.is_floating_point():
            assert torch.isfinite(value[:resident]).all(), name
        if name in {
            "rewards",
            "continuations",
            "terminations",
            "truncations",
            "discounts",
        }:
            scalar_uniques[name] = [
                float(x) for x in torch.unique(value[:resident]).tolist()
            ]
        leaves.append((name, tuple(value.shape), str(value.dtype)))
    action = storage["actions"]
    assert tuple(action.shape[1:]) == (32,)
    expected_layout = {
        "curr_obs.main_images": ((12500, 3, 64, 64), torch.bfloat16),
        "curr_obs.states": ((12500, 14), torch.float32),
        "next_obs.main_images": ((12500, 3, 64, 64), torch.bfloat16),
        "next_obs.states": ((12500, 14), torch.float32),
        "actions": ((12500, 32), torch.bfloat16),
        "rewards": ((12500, 1), torch.float32),
        "continuations": ((12500, 1), torch.bool),
        "terminations": ((12500, 1), torch.bool),
        "truncations": ((12500, 1), torch.bool),
        "discounts": ((12500, 1), torch.float32),
    }
    actual_layout = {
        name: (tuple(value.shape), value.dtype)
        for name, value in tree_leaves(storage)
    }
    assert actual_layout == expected_layout, actual_layout
    rewards = storage["rewards"][:resident]
    continuations = storage["continuations"][:resident]
    terminations = storage["terminations"][:resident]
    discounts = storage["discounts"][:resident]
    assert torch.equal(continuations, ~terminations)
    assert set(float(x) for x in torch.unique(rewards).tolist()) <= {-1.0, 0.0}
    assert torch.equal(rewards[terminations], torch.zeros_like(rewards[terminations]))
    assert torch.equal(
        rewards[~terminations], -torch.ones_like(rewards[~terminations])
    )
    expected_discount = torch.tensor(0.999**20, dtype=torch.float32)
    assert torch.equal(
        discounts, expected_discount.expand_as(discounts)
    ), torch.unique(discounts)
    summary = {
        "rank": rank,
        "resident": resident,
        "total_inserted": inserted,
        "write_cursor": cursor,
        "allocated_bytes": allocated,
        "resident_bytes": resident_bytes,
        "leaves": leaves,
        "scalar_uniques": scalar_uniques,
    }
    del state, storage
    gc.collect()
    return summary


trainer = [
    shadow_summary(
        ckpt / "sac_components" / f"dsrl_trainer_state_rank_{rank}.pt", rank
    )
    for rank in range(2)
]
replay = [
    replay_summary(
        ckpt
        / "sac_components"
        / "replay_buffer"
        / f"rank_{rank}"
        / "dsrl_transition_replay.pt",
        rank,
    )
    for rank in range(2)
]
global_inserted = sum(int(item["total_inserted"]) for item in replay)
expected_updates = 20 * global_inserted
assert 4 <= global_inserted <= 40
assert all(item["update_step"] == expected_updates for item in trainer)
assert trainer[0]["shadow_sha256"] == trainer[1]["shadow_sha256"]
print("TRAINER_SUMMARY", trainer)
print("REPLAY_SUMMARY", replay)
print("GLOBAL_INSERTED", global_inserted)
print("EXPECTED_UPDATES", expected_updates)

event_files = sorted((run_root / "tensorboard").glob("events.out.tfevents.*"))
assert event_files
ea = event_accumulator.EventAccumulator(str(run_root / "tensorboard"))
ea.Reload()
scalar_tags = sorted(ea.Tags().get("scalars", []))
print("SCALAR_TAGS", scalar_tags)
scalar_last = {}
for tag in scalar_tags:
    events = ea.Scalars(tag)
    if not events:
        continue
    for event in events:
        assert math.isfinite(event.value), (tag, event.value)
    scalar_last[tag] = {
        "count": len(events),
        "step": events[-1].step,
        "value": events[-1].value,
    }
print("SCALAR_LAST", scalar_last)
for tag, expected in (
    ("train/sac/global_new_transitions", global_inserted),
    ("train/sac/global_resident_transitions", global_inserted),
    ("train/sac/planned_optimizer_updates", expected_updates),
):
    assert tag in scalar_last, tag
    assert math.isclose(scalar_last[tag]["value"], expected, rel_tol=0, abs_tol=1e-6)
for tag in (
    "train/sac/critic_loss",
    "train/critic/q_data",
    "train/critic/lr",
    "train/critic/grad_norm",
    "train/sac/actor_loss",
    "train/sac/alpha_loss",
    "train/sac/alpha",
    "train/actor/lr",
    "train/actor/grad_norm",
    "train/actor/entropy",
    "train/alpha/grad_norm",
    "train/actor/q_pi",
    "time/sync_weights",
    "time/generate_rollouts",
    "time/actor/run_training",
    "time/eval",
    "eval/num_trajectories",
    "eval/success_once",
    "eval/return",
    "eval/episode_len",
    "eval/reward",
):
    assert tag in scalar_last, tag
for index in range(10):
    assert f"train/actor/q_value_{index}" in scalar_last
for tag in (
    "train/critic/grad_norm",
    "train/actor/grad_norm",
    "train/alpha/grad_norm",
):
    assert scalar_last[tag]["value"] > 0, (tag, scalar_last[tag])
assert scalar_last["train/sac/alpha"]["value"] > 0
assert math.isclose(
    scalar_last["train/critic/lr"]["value"], 3e-4, rel_tol=0, abs_tol=1e-9
)
assert math.isclose(
    scalar_last["train/actor/lr"]["value"], 1e-4, rel_tol=0, abs_tol=1e-9
)
assert math.isclose(
    scalar_last["eval/num_trajectories"]["value"], 4, rel_tol=0, abs_tol=1e-6
)

csv_path = run_root / "resource_monitor" / "fresh" / "resources.csv"
with csv_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert rows
for row in rows:
    assert int(row["cgroup_oom"]) == 0
    assert int(row["cgroup_oom_kill"]) == 0
start = datetime.strptime(rows[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
end = datetime.strptime(rows[-1]["timestamp"], "%Y-%m-%d %H:%M:%S")
train_rows = [row for row in rows if "run_training" in row["top_command"]]
top_commands = Counter(row["top_command"] for row in rows)


def fmax(key: str) -> float:
    return max(float(row[key]) for row in rows)


def favg(items, key: str) -> float:
    return sum(float(row[key]) for row in items) / len(items) if items else 0.0


resource_summary = {
    "samples": len(rows),
    "duration_s": int((end - start).total_seconds()),
    "peak_cgroup_ram_mb": fmax("cgroup_ram_mb"),
    "peak_gpu0_mb": fmax("gpu0_memory_mb"),
    "peak_gpu1_mb": fmax("gpu1_memory_mb"),
    "peak_gpu_total_mb": fmax("gpu_total_memory_mb"),
    "mean_gpu0_util_all": favg(rows, "gpu0_util_pct"),
    "mean_gpu1_util_all": favg(rows, "gpu1_util_pct"),
    "training_samples": len(train_rows),
    "mean_gpu0_util_training": favg(train_rows, "gpu0_util_pct"),
    "mean_gpu1_util_training": favg(train_rows, "gpu1_util_pct"),
    "peak_env_rss_mb": fmax("env_rss_mb"),
    "peak_actor_rss_mb": fmax("actor_rss_mb"),
    "peak_rollout_rss_mb": fmax("rollout_rss_mb"),
    "top_command_counts": top_commands.most_common(12),
}
print("RESOURCE_SUMMARY", resource_summary)
print("FRESH_VALIDATION_OK=1")
PY

echo "== terminal log markers =="
grep -En 'Saving checkpoint|global_step|sac/|replay_buffer/|success|eval|validation|shutdown|finished|complete' \
  "$RUN_ROOT/fresh_driver.log" | tail -n 160 || true
echo "== suspicious markers =="
grep -Ein 'raytaskerror|workercrashederror|actor died|out of memory|cuda error|nan|inf|killed|segmentation|fatal' \
  "$RUN_ROOT/fresh_driver.log" | tail -n 80 || true

echo "FRESH_ARTIFACT_VALIDATION_COMPLETE=1"
