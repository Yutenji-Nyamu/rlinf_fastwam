set -euo pipefail

PY=/root/autodl-tmp/RLinf/.venv/bin/python
RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
EXP="$RUN_ROOT/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke"
CKPT1="$EXP/checkpoints/global_step_1"
CKPT2="$EXP/checkpoints/global_step_2"
RESUME_PID=$(cat "$RUN_ROOT/resume.pid")

if kill -0 "$RESUME_PID" 2>/dev/null; then
  echo "RESUME_STILL_RUNNING=1"
  exit 41
fi

echo "== resume log contract =="
grep -F "Resuming training from checkpoint directory $CKPT1." "$RUN_ROOT/resume_driver.log"
grep -F "[Checkpoint] loading local shard checkpoint from" "$RUN_ROOT/resume_driver.log" || true
grep -F "[Checkpoint] loading DCP checkpoint from" "$RUN_ROOT/resume_driver.log" || true
if grep -Eiq \
  'legacy DSRL|bitwise continuous|failed to load checkpoint|layout mismatch|phase mismatch|shadow.*mismatch' \
  "$RUN_ROOT/resume_driver.log"; then
  echo "RESUME_FALLBACK_OR_MISMATCH_FOUND=1"
  exit 61
fi

echo "== DCP2 artifact contract =="
test -s "$CKPT2/actor/local_shard_checkpoint/checkpoint_rank_0.pt"
test -s "$CKPT2/actor/local_shard_checkpoint/checkpoint_rank_1.pt"
test -s "$CKPT2/actor/sac_components/alpha/dcp_checkpoint/.metadata"
test -n "$(find "$CKPT2/actor/sac_components/alpha/dcp_checkpoint" -maxdepth 1 -type f -name '*.distcp' -print -quit)"
test -s "$CKPT2/actor/sac_components/target_model/checkpoint_rank_0.pt"
test -s "$CKPT2/actor/sac_components/target_model/checkpoint_rank_1.pt"
test -s "$CKPT2/actor/sac_components/dsrl_trainer_state_rank_0.pt"
test -s "$CKPT2/actor/sac_components/dsrl_trainer_state_rank_1.pt"
test -s "$CKPT2/actor/sac_components/replay_buffer/rank_0/dsrl_transition_replay.pt"
test -s "$CKPT2/actor/sac_components/replay_buffer/rank_1/dsrl_transition_replay.pt"
test -z "$(find "$CKPT2" -type f -name '*.tmp' -print -quit)"
du -sh "$CKPT1" "$CKPT2"

cat /sys/fs/cgroup/memory.events > "$RUN_ROOT/resource_monitor/resume/memory.events.final.txt"
awk '$1 ~ /^(anon|file|shmem|file_mapped|inactive_file|active_file|kernel_stack|pagetables|slab)$/ {print}' \
  /sys/fs/cgroup/memory.stat > "$RUN_ROOT/resource_monitor/resume/memory.stat.final.txt"

echo "== resume state/replay/tensorboard/resource validation =="
export CKPT1 CKPT2 RUN_ROOT
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

ckpt1 = Path(os.environ["CKPT1"]) / "actor"
ckpt2 = Path(os.environ["CKPT2"]) / "actor"
run_root = Path(os.environ["RUN_ROOT"])


def load_trainer(base: Path, rank: int) -> tuple[dict, str]:
    state = torch.load(
        base / "sac_components" / f"dsrl_trainer_state_rank_{rank}.pt",
        map_location="cpu",
        weights_only=True,
    )
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
    digest = hashlib.sha256()
    for name in sorted(shadows):
        value = shadows[name]
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return state, digest.hexdigest()


def load_replay(base: Path, rank: int) -> dict:
    state = torch.load(
        base
        / "sac_components"
        / "replay_buffer"
        / f"rank_{rank}"
        / "dsrl_transition_replay.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert state["schema_version"] == 1
    assert state["global_capacity"] == 25000
    assert state["local_capacity"] == 12500
    assert state["rank"] == rank
    assert state["world_size"] == 2
    assert state["seed"] == 1234
    assert state["storage"] is not None
    return state


def tree_pairs(left, right, prefix=""):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        yield prefix, left, right
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in sorted(left):
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from tree_pairs(left[key], right[key], name)
    else:
        raise TypeError((prefix, type(left)))


t1 = []
t2 = []
h1 = []
h2 = []
for rank in range(2):
    state1, digest1 = load_trainer(ckpt1, rank)
    state2, digest2 = load_trainer(ckpt2, rank)
    t1.append(state1)
    t2.append(state2)
    h1.append(digest1)
    h2.append(digest2)
assert h1[0] == h1[1]
assert h2[0] == h2[1]
assert h1[0] != h2[0]

replay_summaries = []
global_old = 0
global_new = 0
for rank in range(2):
    before = load_replay(ckpt1, rank)
    after = load_replay(ckpt2, rank)
    old_resident = int(before["resident_size"])
    new_resident = int(after["resident_size"])
    inserted_delta = int(after["total_inserted"]) - int(before["total_inserted"])
    assert 0 < inserted_delta <= 20
    assert new_resident == old_resident + inserted_delta
    assert int(after["write_cursor"]) == int(before["write_cursor"]) + inserted_delta

    for name, old_value, new_value in tree_pairs(
        before["storage"], after["storage"]
    ):
        assert old_value.shape == new_value.shape
        assert old_value.dtype == new_value.dtype
        assert torch.equal(
            old_value[:old_resident], new_value[:old_resident]
        ), f"restored replay prefix changed: {name}"
        appended = new_value[old_resident:new_resident]
        if appended.is_floating_point():
            assert torch.isfinite(appended).all(), name

    appended_actions = after["storage"]["actions"][old_resident:new_resident]
    assert tuple(appended_actions.shape[1:]) == (32,)
    assert appended_actions.abs().max().item() <= 1.0
    assert torch.count_nonzero(appended_actions).item() > 0
    appended_rewards = after["storage"]["rewards"][old_resident:new_resident]
    appended_terminations = after["storage"]["terminations"][
        old_resident:new_resident
    ]
    appended_continuations = after["storage"]["continuations"][
        old_resident:new_resident
    ]
    appended_discounts = after["storage"]["discounts"][old_resident:new_resident]
    assert torch.equal(appended_continuations, ~appended_terminations)
    assert set(float(x) for x in torch.unique(appended_rewards).tolist()) <= {
        -1.0,
        0.0,
    }
    assert torch.equal(
        appended_discounts,
        torch.tensor(0.999**20, dtype=torch.float32).expand_as(
            appended_discounts
        ),
    )

    # The flat replay uses one local batch of 128 indices per optimizer update.
    update_delta = int(t2[rank]["update_step"]) - int(t1[rank]["update_step"])
    generator = torch.Generator(device="cpu")
    generator.set_state(before["rng_state"])
    for _ in range(update_delta):
        torch.randint(
            0,
            new_resident,
            (128,),
            generator=generator,
            device="cpu",
        )
    assert torch.equal(generator.get_state(), after["rng_state"])

    replay_summaries.append(
        {
            "rank": rank,
            "old_resident": old_resident,
            "new_resident": new_resident,
            "inserted_delta": inserted_delta,
            "new_action_absmax": appended_actions.abs().max().item(),
            "new_action_absmean": appended_actions.abs().float().mean().item(),
            "new_reward_unique": [
                float(x) for x in torch.unique(appended_rewards).tolist()
            ],
            "new_termination_unique": [
                bool(x) for x in torch.unique(appended_terminations).tolist()
            ],
            "new_truncation_unique": [
                bool(x)
                for x in torch.unique(
                    after["storage"]["truncations"][old_resident:new_resident]
                ).tolist()
            ],
            "rng_continuity": True,
        }
    )
    global_old += old_resident
    global_new += inserted_delta
    del before, after
    gc.collect()

planned2 = 20 * global_new
assert global_old == 40
assert 4 <= global_new <= 40
assert all(int(item["update_step"]) == 800 for item in t1)
assert all(int(item["update_step"]) == 800 + planned2 for item in t2)
print(
    "TRAINER_CONTINUITY",
    {
        "ckpt1_update_step": 800,
        "ckpt2_update_step": 800 + planned2,
        "ckpt1_shadow_sha256": h1[0],
        "ckpt2_shadow_sha256": h2[0],
        "shadow_changed": h1[0] != h2[0],
    },
)
print("REPLAY_CONTINUITY", replay_summaries)
print("RESUME_GLOBAL_NEW", global_new)
print("RESUME_PLANNED_UPDATES", planned2)

ea = event_accumulator.EventAccumulator(str(run_root / "tensorboard"))
ea.Reload()
tags = sorted(ea.Tags().get("scalars", []))
last = {}
for tag in tags:
    events = ea.Scalars(tag)
    if not events:
        continue
    assert all(math.isfinite(event.value) for event in events), tag
    last[tag] = {
        "count": len(events),
        "step": events[-1].step,
        "value": events[-1].value,
    }
for tag, expected in (
    ("train/sac/global_new_transitions", global_new),
    ("train/sac/global_resident_transitions", global_old + global_new),
    ("train/sac/planned_optimizer_updates", planned2),
):
    assert last[tag]["step"] == 1
    assert math.isclose(last[tag]["value"], expected, rel_tol=0, abs_tol=1e-6)
for tag in (
    "train/sac/critic_loss",
    "train/critic/grad_norm",
    "train/sac/actor_loss",
    "train/sac/alpha_loss",
    "train/sac/alpha",
    "train/actor/grad_norm",
    "train/alpha/grad_norm",
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
    assert tag in last and last[tag]["step"] == 1, tag
for tag in (
    "train/critic/grad_norm",
    "train/actor/grad_norm",
    "train/alpha/grad_norm",
):
    assert last[tag]["value"] > 0, (tag, last[tag])
assert last["train/sac/alpha"]["value"] > 0
assert math.isclose(last["eval/num_trajectories"]["value"], 4, abs_tol=1e-6)
print("RESUME_SCALAR_LAST", last)

csv_path = run_root / "resource_monitor" / "resume" / "resources.csv"
with csv_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert rows
assert all(int(row["cgroup_oom"]) == 0 for row in rows)
assert all(int(row["cgroup_oom_kill"]) == 0 for row in rows)
start = datetime.strptime(rows[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
end = datetime.strptime(rows[-1]["timestamp"], "%Y-%m-%d %H:%M:%S")
train_rows = [row for row in rows if "run_training" in row["top_command"]]


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
    "top_command_counts": Counter(row["top_command"] for row in rows).most_common(
        12
    ),
}
print("RESUME_RESOURCE_SUMMARY", resource_summary)
print("RESUME_VALIDATION_OK=1")
PY

echo "== terminal metrics =="
tail -n 90 "$RUN_ROOT/resume_driver.log"
echo "RESUME_ARTIFACT_VALIDATION_COMPLETE=1"
