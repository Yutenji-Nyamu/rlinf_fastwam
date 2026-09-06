from __future__ import annotations

import torch

from rlinf.data.ogpo_replay import OGPOPrimitiveRow, OGPOReplayBuffer


def _observation(episode_id: int, step_id: int) -> dict[str, torch.Tensor]:
    return {
        "image": torch.full(
            (2, 2, 3),
            (episode_id + step_id) % 255,
            dtype=torch.uint8,
        ),
        "proprio": torch.tensor([episode_id, step_id], dtype=torch.float32),
        "prompt_bytes": torch.tensor([1, 2, 3, 0], dtype=torch.uint8),
        "prompt_length": torch.tensor(3, dtype=torch.long),
    }


def _row(
    episode_id: int,
    step_id: int,
    *,
    terminated: bool = False,
    truncated: bool = False,
) -> OGPOPrimitiveRow:
    return OGPOPrimitiveRow(
        observation=_observation(episode_id, step_id),
        next_observation=_observation(episode_id, step_id + 1),
        action_model=torch.full((32,), float(step_id + 1)),
        action=torch.full((14,), float(step_id + 1)),
        reward=float(step_id + 1),
        terminated=terminated,
        truncated=truncated,
        episode_id=episode_id,
        step_id=step_id,
    )


def _assert_sequence_batches_equal(left, right) -> None:
    for name in (
        "action_model",
        "action",
        "rewards",
        "terminated",
        "truncated",
        "valid",
        "h",
        "bootstrap_mask",
        "row_ids",
        "episode_ids",
        "start_step_ids",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name))
    for name in left.observation:
        assert torch.equal(left.observation[name], right.observation[name])
        assert torch.equal(
            left.next_observation[name],
            right.next_observation[name],
        )


def test_random_sequences_do_not_cross_episodes_and_handle_done_masks() -> None:
    replay = OGPOReplayBuffer(capacity=20, max_sequence_length=3, seed=7)
    assert replay.sample_success_sequences(2) is None

    # Interleave two vector-environment episodes in physical insertion order.
    first_ids = []
    second_ids = []
    for step_id in range(3):
        first_ids.append(
            replay.add(
                _row(10, step_id, terminated=step_id == 2)
            )
        )
        second_ids.append(
            replay.add(
                _row(20, step_id, truncated=step_id == 2)
            )
        )
    replay.mark_episode_success(20)

    batch = replay.sample_sequences(512)
    saw_termination = False
    saw_truncation = False
    for batch_index in range(batch.h.numel()):
        length = int(batch.h[batch_index])
        sampled_ids = batch.row_ids[batch_index, :length].tolist()
        rows = [replay.get_row(row_id) for row_id in sampled_ids]
        assert len({row.episode_id for row in rows}) == 1
        assert [row.step_id for row in rows] == list(
            range(rows[0].step_id, rows[0].step_id + length)
        )
        # The replay requests the full C-step sequence.  Shorter horizons arise
        # only because this episode naturally ends, not from random h sampling.
        assert length == 3 - rows[0].step_id
        assert batch.valid[batch_index, :length].all()
        assert not batch.valid[batch_index, length:].any()
        assert not batch.action[batch_index, length:].any()
        assert torch.equal(
            batch.next_observation["proprio"][batch_index],
            rows[-1].next_observation["proprio"],
        )
        if sampled_ids[-1] == first_ids[-1]:
            saw_termination = True
            assert batch.bootstrap_mask[batch_index].item() == 0.0
        if sampled_ids[-1] == second_ids[-1]:
            saw_truncation = True
            assert batch.bootstrap_mask[batch_index].item() == 1.0
    assert saw_termination
    assert saw_truncation

    success = replay.sample_success_sequences(8)
    assert success is not None
    assert replay.success_row_ids == tuple(second_ids)
    assert torch.equal(
        success.row_ids,
        torch.tensor([second_ids] * 8, dtype=torch.long),
    )
    assert torch.all(success.episode_ids == 20)


def test_state_dict_and_snapshot_restore_ring_success_view_and_rng(tmp_path) -> None:
    replay = OGPOReplayBuffer(capacity=8, max_sequence_length=3, seed=19)
    replay.add_episode(
        [
            _row(30, 0),
            _row(30, 1),
            _row(30, 2),
            _row(30, 3, truncated=True),
        ],
        success=True,
    )

    restored = OGPOReplayBuffer(capacity=8, max_sequence_length=3, seed=19)
    restored.load_state_dict(replay.state_dict())

    snapshot_path = tmp_path / "ogpo_replay.pt"
    replay.save_snapshot(snapshot_path)
    loaded = OGPOReplayBuffer(capacity=8, max_sequence_length=3, seed=19)
    loaded.load_snapshot(snapshot_path)

    assert len(restored) == len(loaded) == len(replay)
    assert restored.cursor == loaded.cursor == replay.cursor
    assert restored.total_inserted == loaded.total_inserted == replay.total_inserted
    assert restored.success_row_ids == loaded.success_row_ids == replay.success_row_ids

    original_batch = replay.sample_sequences(32)
    restored_batch = restored.sample_sequences(32)
    loaded_batch = loaded.sample_sequences(32)
    _assert_sequence_batches_equal(original_batch, restored_batch)
    _assert_sequence_batches_equal(original_batch, loaded_batch)
