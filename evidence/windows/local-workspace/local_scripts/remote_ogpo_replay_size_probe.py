"""Measure actual in-memory and serialized bytes for RoboTwin OGPO replay rows."""

from __future__ import annotations

import gc
import io
import os

import psutil
import torch

from rlinf.data.ogpo_replay import OGPOPrimitiveRow, OGPOReplayBuffer


def gib(value: float) -> float:
    return value / 1024**3


def observation(fill: int) -> dict[str, torch.Tensor]:
    return {
        "main_images": torch.full((240, 320, 3), fill % 256, dtype=torch.uint8),
        "wrist_images": torch.full(
            (2, 240, 320, 3), (fill + 1) % 256, dtype=torch.uint8
        ),
        "states": torch.zeros(14, dtype=torch.float32),
        "prompt_utf8": torch.zeros(256, dtype=torch.uint8),
        "prompt_length": torch.tensor(17, dtype=torch.long),
    }


count = 128
process = psutil.Process(os.getpid())
replay = OGPOReplayBuffer(
    capacity=count,
    max_sequence_length=10,
    action_dim=14,
    model_action_dim=32,
    seed=1234,
)
gc.collect()
rss_before = process.memory_info().rss
for index in range(count):
    replay.add(
        OGPOPrimitiveRow(
            observation=observation(index),
            next_observation=observation(index + 1),
            action_model=torch.zeros(32),
            action=torch.zeros(14),
            reward=0.0,
            terminated=index == count - 1,
            truncated=False,
            episode_id=0,
            step_id=index,
        )
    )
gc.collect()
rss_after = process.memory_info().rss
rss_per_row = (rss_after - rss_before) / count

payload = replay.state_dict()
buffer = io.BytesIO()
torch.save(payload, buffer)
serialized_per_row = buffer.tell() / count

raw_tensor_bytes = 0
row = replay.get_row(0)
for view in (row.observation, row.next_observation):
    raw_tensor_bytes += sum(
        tensor.numel() * tensor.element_size() for tensor in view.values()
    )
raw_tensor_bytes += row.action.numel() * row.action.element_size()
raw_tensor_bytes += row.action_model.numel() * row.action_model.element_size()

print(
    "REPLAY_ROW_SIZE "
    f"sample_rows={count} raw_tensor_bytes={raw_tensor_bytes} "
    f"rss_bytes_per_row={rss_per_row:.1f} "
    f"serialized_bytes_per_row={serialized_per_row:.1f}",
    flush=True,
)
for global_capacity in (20_000, 50_000, 250_000):
    per_rank = (global_capacity + 1) // 2
    print(
        "REPLAY_CAPACITY_ESTIMATE "
        f"global_rows={global_capacity} per_rank_rows={per_rank} "
        f"global_rss_gib={gib(rss_per_row * global_capacity):.2f} "
        f"global_sidecar_gib={gib(serialized_per_row * global_capacity):.2f}",
        flush=True,
    )
