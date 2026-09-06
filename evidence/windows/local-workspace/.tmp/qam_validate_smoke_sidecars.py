from __future__ import annotations

import gc
import hashlib
import json
import math
import pathlib
import sys
from typing import Any

import torch


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return value.numpy().tobytes()


def digest(value: Any) -> str:
    hasher = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            hasher.update(b"T")
            hasher.update(str(item.dtype).encode())
            hasher.update(str(tuple(item.shape)).encode())
            hasher.update(tensor_bytes(item))
        elif isinstance(item, dict):
            hasher.update(b"D")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            hasher.update(b"L" if isinstance(item, list) else b"U")
            for child in item:
                update(child)
        else:
            hasher.update(type(item).__name__.encode())
            hasher.update(repr(item).encode())

    update(value)
    return hasher.hexdigest()


def finite_tensor_summary(value: Any) -> tuple[int, int]:
    tensor_count = 0
    nonfinite_count = 0

    def visit(item: Any) -> None:
        nonlocal tensor_count, nonfinite_count
        if torch.is_tensor(item):
            tensor_count += 1
            if item.is_floating_point():
                nonfinite_count += int((~torch.isfinite(item)).sum().item())
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return tensor_count, nonfinite_count


root = pathlib.Path(sys.argv[1])
sidecar_summaries: list[dict[str, Any]] = []
for rank in range(2):
    path = root / f"rank_{rank}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor_count, nonfinite = finite_tensor_summary(payload)
    critic = payload["critic"]
    target = payload["target_critic"]
    max_abs = 0.0
    squared_sum = 0.0
    for key, online_value in critic.items():
        target_value = target[key]
        delta = online_value.double() - target_value.double()
        max_abs = max(max_abs, float(delta.abs().max().item()))
        squared_sum += float(delta.square().sum().item())
    head_digests = [
        digest(critic[f"q_functions.{index}.network.0.weight"])
        for index in range(10)
    ]
    sidecar_summaries.append(
        {
            "rank": payload["rank"],
            "complete": payload["complete"],
            "world_size": payload["world_size"],
            "phase": payload["saved_phase"],
            "contract": payload["contract_fingerprint"],
            "runner_global_step": payload["runner_global_step"],
            "fine_policy_version": payload["fine_policy_version"],
            "critic_updates": payload["critic_updates"],
            "fine_updates": payload["fine_updates"],
            "local_total_inserts": payload["local_total_inserts"],
            "global_total_inserts": payload["global_total_inserts"],
            "pending_update_credit": payload["pending_update_credit"],
            "q_only_anchor": payload["q_only_anchor_global_inserts"],
            "critic_digest": digest(critic),
            "target_digest": digest(target),
            "optimizer_digest": digest(payload["critic_optimizer"]),
            "head_first_layer_unique": len(set(head_digests)),
            "critic_target_max_abs": max_abs,
            "critic_target_l2": math.sqrt(squared_sum),
            "tensor_count": tensor_count,
            "nonfinite_tensor_values": nonfinite,
            "bytes": path.stat().st_size,
        }
    )
    del payload, critic, target
    gc.collect()

replay_summaries: list[dict[str, Any]] = []
for rank in range(2):
    path = root / f"replay_rank_{rank}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    active = payload["slots"][: payload["size"]]
    tensor_count, nonfinite = finite_tensor_summary(active)
    normalized = torch.stack(
        [slot["planned_actions_normalized"] for slot in active]
    )
    env_actions = torch.stack([slot["planned_actions_env"] for slot in active])
    rewards = torch.stack([slot["chunk_rewards_native"] for slot in active])
    chain_links = sum(
        left["next_obs_id"] == right["obs_id"]
        for left, right in zip(active[:-1], active[1:])
    )
    replay_summaries.append(
        {
            "rank": payload["metadata"]["rank"],
            "complete": payload["complete"],
            "world_size": payload["metadata"]["world_size"],
            "capacity": payload["metadata"]["capacity"],
            "cursor": payload["cursor"],
            "size": payload["size"],
            "total_inserted": payload["total_inserted"],
            "contract": payload["metadata"]["contract_fingerprint"],
            "observations": len(payload["observations"]),
            "chain_links": chain_links,
            "chain_links_expected": max(len(active) - 1, 0),
            "next_state_valid": sum(slot["next_state_valid"] for slot in active),
            "success_terminated": sum(
                slot["success_terminated"] for slot in active
            ),
            "time_limit_truncated": sum(
                slot["time_limit_truncated"] for slot in active
            ),
            "other_truncated": sum(slot["other_truncated"] for slot in active),
            "bootstrap_mask_sum": sum(slot["bootstrap_mask"] for slot in active),
            "policy_versions": sorted(
                {slot["policy_version"] for slot in active}
            ),
            "normalized_min": float(normalized.min().item()),
            "normalized_max": float(normalized.max().item()),
            "env_min": float(env_actions.min().item()),
            "env_max": float(env_actions.max().item()),
            "reward_sum": float(rewards.sum().item()),
            "distinct_plans": len(
                {digest(slot["planned_actions_normalized"]) for slot in active}
            ),
            "tensor_count": tensor_count,
            "nonfinite_tensor_values": nonfinite,
            "bytes": path.stat().st_size,
        }
    )
    del payload, active, normalized, env_actions, rewards
    gc.collect()

checks = {
    "sidecars_complete": all(item["complete"] for item in sidecar_summaries),
    "sidecar_contract_equal": len(
        {item["contract"] for item in sidecar_summaries}
    )
    == 1,
    "critic_cross_rank_equal": len(
        {item["critic_digest"] for item in sidecar_summaries}
    )
    == 1,
    "target_cross_rank_equal": len(
        {item["target_digest"] for item in sidecar_summaries}
    )
    == 1,
    "optimizer_cross_rank_equal": len(
        {item["optimizer_digest"] for item in sidecar_summaries}
    )
    == 1,
    "critic_target_different": all(
        item["critic_target_max_abs"] > 0.0 for item in sidecar_summaries
    ),
    "ten_heads_independent": all(
        item["head_first_layer_unique"] == 10 for item in sidecar_summaries
    ),
    "sidecars_finite": all(
        item["nonfinite_tensor_values"] == 0 for item in sidecar_summaries
    ),
    "replays_complete": all(item["complete"] for item in replay_summaries),
    "replays_have_10": all(item["size"] == 10 for item in replay_summaries),
    "replay_chains_complete": all(
        item["chain_links"] == item["chain_links_expected"]
        for item in replay_summaries
    ),
    "replays_finite": all(
        item["nonfinite_tensor_values"] == 0 for item in replay_summaries
    ),
}

print(
    json.dumps(
        {
            "checks": checks,
            "sidecars": sidecar_summaries,
            "replays": replay_summaries,
        },
        indent=2,
        sort_keys=True,
    )
)
if not all(checks.values()):
    raise SystemExit(1)

