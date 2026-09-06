set -euo pipefail

PY=/root/autodl-tmp/RLinf/.venv/bin/python
RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
EXP="$RUN_ROOT/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke"
export CKPT1="$EXP/checkpoints/global_step_1/actor"
export CKPT2="$EXP/checkpoints/global_step_2/actor"
export PYTHONDONTWRITEBYTECODE=1

"$PY" -B - <<'PY'
from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import torch

ckpt1 = Path(os.environ["CKPT1"])
ckpt2 = Path(os.environ["CKPT2"])

actor_components = {
    "dsrl_action_noise_net",
    "actor_image_encoder",
    "actor_state_encoder",
}
critic_components = {
    "critic_image_encoder",
    "critic_state_encoder",
    "q_head",
}


def normalize_name(name: str) -> str:
    parts = [part for part in name.split(".") if part != "_fsdp_wrapped_module"]
    return ".".join(parts)


def classify(name: str) -> str:
    parts = set(normalize_name(name).split("."))
    if "dsrl_policy_phase" in parts:
        return "phase"
    if parts & actor_components:
        return "actor"
    if parts & critic_components:
        return "critic"
    return "frozen_base"


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def summarize_model(model: dict[str, torch.Tensor]) -> dict[str, dict]:
    result = {
        "frozen_base": {},
        "actor": {},
        "critic": {},
        "phase": {},
    }
    for raw_name, value in model.items():
        assert isinstance(value, torch.Tensor), (raw_name, type(value))
        name = normalize_name(raw_name)
        group = classify(name)
        result[group][name] = {
            "hash": tensor_hash(value),
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "value": int(value.item()) if group == "phase" else None,
        }
    assert len(result["phase"]) == 1, result["phase"].keys()
    assert next(iter(result["phase"].values()))["value"] == 1
    assert result["frozen_base"] and result["actor"] and result["critic"]
    return result


def optimizer_summary(optimizers: list[dict]) -> list[dict]:
    output = []
    for index, optimizer in enumerate(optimizers):
        states = optimizer["state"]
        exp_avg_nonzero = 0
        exp_avg_sq_nonzero = 0
        tensor_states = 0
        for item in states.values():
            for key, value in item.items():
                if not isinstance(value, torch.Tensor):
                    continue
                assert torch.isfinite(value).all(), (index, key)
                tensor_states += 1
                if key == "exp_avg" and torch.count_nonzero(value).item() > 0:
                    exp_avg_nonzero += 1
                if key == "exp_avg_sq" and torch.count_nonzero(value).item() > 0:
                    exp_avg_sq_nonzero += 1
        assert states
        assert exp_avg_nonzero > 0
        assert exp_avg_sq_nonzero > 0
        output.append(
            {
                "optimizer_index": index,
                "parameter_states": len(states),
                "tensor_states": tensor_states,
                "exp_avg_nonzero": exp_avg_nonzero,
                "exp_avg_sq_nonzero": exp_avg_sq_nonzero,
            }
        )
    assert len(output) == 2
    return output


def load_online(path: Path) -> tuple[dict[str, dict], list[dict]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert set(state) == {
        "model",
        "optimizers",
        "lr_schedulers",
        "fsdp_version",
        "rng",
    }
    summary = summarize_model(state["model"])
    optimizers = optimizer_summary(state["optimizers"])
    del state
    gc.collect()
    return summary, optimizers


def load_target(path: Path) -> dict[str, dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    summary = summarize_model(state)
    del state
    gc.collect()
    return summary


def changed_count(left: dict, right: dict) -> int:
    assert set(left) == set(right)
    for key in left:
        assert left[key]["shape"] == right[key]["shape"]
        assert left[key]["dtype"] == right[key]["dtype"]
    return sum(left[key]["hash"] != right[key]["hash"] for key in left)


print("LOAD_ONLINE_CKPT1")
online1, optimizer1 = load_online(
    ckpt1 / "local_shard_checkpoint" / "checkpoint_rank_0.pt"
)
print("LOAD_TARGET_CKPT1")
target1 = load_target(
    ckpt1 / "sac_components" / "target_model" / "checkpoint_rank_0.pt"
)
print("LOAD_ONLINE_CKPT2")
online2, optimizer2 = load_online(
    ckpt2 / "local_shard_checkpoint" / "checkpoint_rank_0.pt"
)

fresh_online_vs_target_base = changed_count(
    online1["frozen_base"], target1["frozen_base"]
)
resume_frozen_base_changed = changed_count(
    online1["frozen_base"], online2["frozen_base"]
)
resume_actor_changed = changed_count(online1["actor"], online2["actor"])
resume_critic_changed = changed_count(online1["critic"], online2["critic"])
resume_phase_changed = changed_count(online1["phase"], online2["phase"])

assert fresh_online_vs_target_base == 0
assert resume_frozen_base_changed == 0
assert resume_actor_changed > 0
assert resume_critic_changed > 0
assert resume_phase_changed == 0

summary = {
    "frozen_base_tensors": len(online1["frozen_base"]),
    "frozen_base_numel": sum(
        item["numel"] for item in online1["frozen_base"].values()
    ),
    "fresh_online_vs_target_base_changed": fresh_online_vs_target_base,
    "resume_frozen_base_changed": resume_frozen_base_changed,
    "actor_tensors": len(online1["actor"]),
    "resume_actor_changed": resume_actor_changed,
    "critic_tensors": len(online1["critic"]),
    "resume_critic_changed": resume_critic_changed,
    "phase_changed": resume_phase_changed,
    "optimizer_ckpt1": optimizer1,
    "optimizer_ckpt2": optimizer2,
}
print("BASE_FREEZE_AND_SMALL_MODULE_DELTA", summary)
print("BASE_FREEZE_VALIDATION_OK=1")
PY
