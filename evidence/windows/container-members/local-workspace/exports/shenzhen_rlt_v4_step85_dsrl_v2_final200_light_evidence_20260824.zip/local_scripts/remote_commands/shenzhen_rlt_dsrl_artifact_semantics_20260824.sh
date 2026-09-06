#!/usr/bin/env bash
set -u

python=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
rlt_base=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix/robotwin_adjust_bottle_rlt_stage2_current_ar_8env250_optimizer_warmup_v3/checkpoints
rlt_event=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix/tensorboard/events.out.tfevents.1787542941.admin.1389995.0
dsrl_base=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run/dsrl-current-formal-200c-v2/checkpoints
dsrl_event=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run/tensorboard/events.out.tfevents.1787503673.admin.370987.0
stage1_manifest=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1/artifacts/stage1_artifact_manifest.json

"$python" - "$rlt_base" "$rlt_event" "$dsrl_base" "$dsrl_event" "$stage1_manifest" <<'PY'
import json
import sys
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

rlt_base, rlt_event, dsrl_base, dsrl_event, manifest_path = map(Path, sys.argv[1:])


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def tb_summary(path: Path):
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in sorted(acc.Tags().get("scalars", [])):
        values = acc.Scalars(tag)
        if values:
            out[tag] = {
                "count": len(values),
                "first_step": values[0].step,
                "first_value": values[0].value,
                "last_step": values[-1].step,
                "last_value": values[-1].value,
            }
    return out


out = {"stage1_manifest": {}, "rlt_checkpoints": {}, "dsrl_checkpoints": {}}
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    out["stage1_manifest"] = {
        "path": str(manifest_path),
        "id": manifest.get("id"),
        "model_identity": manifest.get("model_identity"),
        "reconstruction": manifest.get("reconstruction"),
        "train_vla": manifest.get("train_vla"),
        "dataset": manifest.get("dataset"),
        "full_weights": manifest.get("full_weights"),
    }

for checkpoint in sorted(rlt_base.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1])):
    state_dir = checkpoint / "actor/sac_components/rlt_trainer_state"
    complete_path = state_dir / "complete.json"
    complete = json.loads(complete_path.read_text()) if complete_path.exists() else None
    rank_states = []
    for state_path in sorted(state_dir.glob("checkpoint_rank_*.pt")):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        rank_states.append({
            key: state.get(key)
            for key in (
                "rank",
                "actor_world_size",
                "saved_runner_step",
                "update_step",
                "local_total_transitions_added",
                "global_warmup_ready_total_transitions",
            )
        })
    replay_files = list((checkpoint / "actor/replay_buffer").rglob("*")) if (checkpoint / "actor/replay_buffer").exists() else []
    dcp = checkpoint / "actor/dcp_checkpoint"
    out["rlt_checkpoints"][checkpoint.name] = {
        "bytes": tree_bytes(checkpoint),
        "files": sum(1 for item in checkpoint.rglob("*") if item.is_file()),
        "complete": complete,
        "rank_states": rank_states,
        "replay_files": sum(1 for item in replay_files if item.is_file()),
        "dcp_metadata": (dcp / ".metadata").exists(),
        "dcp_shards": [item.stat().st_size for item in sorted(dcp.glob("*.distcp"))],
        "full_weights_bytes": (checkpoint / "actor/model_state_dict/full_weights.pt").stat().st_size if (checkpoint / "actor/model_state_dict/full_weights.pt").exists() else None,
    }

for checkpoint in sorted(dsrl_base.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1])):
    actor = checkpoint / "actor"
    state_paths = sorted((actor / "sac_components").glob("dsrl_trainer_state_rank_*.pt"))
    states = []
    for state_path in state_paths:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        shadow = state.get("target_shadow_f32", {})
        states.append({
            "rank": state.get("rank"),
            "saved_runner_step": state.get("saved_runner_step"),
            "update_step": state.get("update_step"),
            "learned_policy_phase": state.get("learned_policy_phase"),
            "pending_optimizer_updates": state.get("pending_optimizer_updates"),
            "shadow_tensors": len(shadow),
            "shadow_numel": sum(value.numel() for value in shadow.values() if isinstance(value, torch.Tensor)),
        })
    replay_paths = sorted((actor / "replay_buffer").rglob("*.pt")) if (actor / "replay_buffer").exists() else []
    dcp = actor / "dcp_checkpoint"
    out["dsrl_checkpoints"][checkpoint.name] = {
        "bytes": tree_bytes(checkpoint),
        "files": sum(1 for item in checkpoint.rglob("*") if item.is_file()),
        "rank_states": states,
        "replay_files": len(replay_paths),
        "replay_bytes": sum(item.stat().st_size for item in replay_paths),
        "dcp_metadata": (dcp / ".metadata").exists(),
        "dcp_shards": [item.stat().st_size for item in sorted(dcp.glob("*.distcp"))],
    }

out["rlt_tensorboard"] = tb_summary(rlt_event)
out["dsrl_tensorboard"] = tb_summary(dsrl_event)
print(json.dumps(out, indent=2, sort_keys=True))
PY
