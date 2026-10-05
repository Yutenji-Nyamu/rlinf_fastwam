"""CPU-only Rynn video sampling and sparse-success transport contracts."""

import numpy as np


INVALID_REWARD_SENTINEL = -1.0


def sample_history(history, frame_count=8):
    if not history or frame_count != 8:
        raise ValueError("Expected a nonempty history and the reviewed K8 protocol")
    indices = np.linspace(0, len(history) - 1, frame_count, dtype=np.int64)
    frames = np.stack([history[index] for index in indices])
    if frames.shape != (8, 256, 320, 3) or frames.dtype != np.uint8:
        raise ValueError("Rynn history must contain uint8 head RGB[256,320,3]")
    return frames, indices * 4


def validate_success_reply(reply, episode_uids, end_action_indices):
    items = reply.get("items")
    if reply.get("ok") is not True or not isinstance(items, list) or len(items) != len(episode_uids):
        raise ValueError("Rynn returned an incomplete result batch")
    success = []
    for item, uid, end in zip(items, episode_uids, end_action_indices):
        if item.get("episode_uid") != str(uid) or item.get("end_action_idx") != int(end):
            raise ValueError("Rynn result identity/order mismatch")
        if "success" not in item:
            raise ValueError("Missing Rynn success result")
        value = item["success"]
        if value is not None and not isinstance(value, bool):
            raise ValueError("Rynn success must be bool or null")
        if not isinstance(item.get("parse_status"), str):
            raise ValueError("Missing Rynn parse status")
        success.append(-1 if value is None else int(value))
    return np.asarray(success, dtype=np.int8)


def invalid_group_members(unknown_indices, num_envs, group_size):
    if num_envs % group_size:
        raise ValueError("Partial GRPO group")
    mask = np.zeros(num_envs, dtype=bool)
    for index in unknown_indices:
        start = int(index) // group_size * group_size
        mask[start:start + group_size] = True
    return mask
