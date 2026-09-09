"""CPU checks for real query boundaries, clean replay and global-batch weights."""
import torch
import pytest

from rlinf.data.online_bc import (
    SuccessEpisodeCollector,
    SuccessReplay,
    masked_fm_loss,
    snapshot_rabc_observation,
)


def inputs(n=2):
    return {"observation/state": torch.zeros(n, 14),
            "tokenized_prompt": torch.arange(4).repeat(n, 1)}


def obs(values, task="Move the pill bottle onto the pad."):
    return {"main_images": torch.tensor(values, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 4, 3).clone(),
            "task_descriptions": [task] * len(values)}


def append(collector, before, after, success, dones, final=None):
    collector.append(inputs(), torch.zeros(2, 50, 14), torch.tensor(success),
                     torch.tensor(dones)[:, None],
                     pre_observation=before, post_observation=after, final_observation=final)


def test_query_boundaries_per_env_terminal_and_stable_episode_key():
    collector = SuccessEpisodeCollector(2, rabc_enabled=True)
    source = obs([10, 20])
    frozen = snapshot_rabc_observation(source, 2)
    source["main_images"].fill_(200)  # Simulate mutable renderer/IPC storage.
    append(collector, frozen, obs([11, 21]), [False, False], [False, False])
    append(collector, obs([11, 21]), obs([201, 22]), [True, True], [True, False], obs([12, 202]))
    episodes = collector.drain()
    assert len(episodes) == 2 and [len(ep) for ep in episodes] == [2, 2]
    assert episodes[0][0]["rabc_pre_image"].unique().item() == 10
    assert episodes[0][-1]["rabc_post_image"].unique().item() == 12  # Terminal, not reset 201.
    assert episodes[1][-1]["rabc_post_image"].unique().item() == 22  # Neighbour keeps normal post.
    keys = []
    for episode in episodes:
        key = episode[0]["rabc_episode_key"]
        keys.append(key.clone())
        assert key.dtype == torch.uint8 and key.shape == (16,)
        assert all(torch.equal(row["rabc_episode_key"], key) for row in episode)
        assert all(isinstance(value, torch.Tensor) for row in episode for value in row.values())
        first = episode[0]
        task = bytes(first["rabc_task_utf8"][:first["rabc_task_length"].item()].tolist()).decode("utf-8")
        assert task == "Move the pill bottle onto the pad."
    assert not torch.equal(keys[0], keys[1])
    # A finished collector ignores further queries until the real environment reset.
    append(collector, obs([201, 22]), obs([202, 23]), [True, True], [False, False])
    assert collector.drain() == []
    collector.reset()
    append(collector, obs([30, 40]), obs([31, 41]), [True, True], [False, False])
    fresh = collector.drain()
    assert not torch.equal(fresh[0][0]["rabc_episode_key"], keys[0])


def test_bad_rgb_task_or_time_alignment_fails_explicitly():
    with pytest.raises(ValueError, match="uint8"):
        bad = obs([1, 2])
        bad["main_images"] = bad["main_images"].float() / 255
        snapshot_rabc_observation(bad, 2)
    with pytest.raises(ValueError, match="4096"):
        snapshot_rabc_observation(obs([1, 2], task="长" * 1366), 2)
    collector = SuccessEpisodeCollector(2, rabc_enabled=True)
    append(collector, obs([1, 2]), obs([3, 4]), [False, False], [False, False])
    with pytest.raises(ValueError, match="instruction changed"):
        append(collector, obs([3, 4], task="A different task"), obs([5, 6]), [False, False], [False, False])
    with pytest.raises(ValueError, match="not contiguous"):
        append(collector, obs([9, 4]), obs([5, 6]), [False, False], [False, False])
    with pytest.raises(ValueError, match="final_observation"):
        append(collector, obs([3, 4]), obs([5, 6]), [True, False], [True, False])


def test_disabled_collector_and_success_replay_keep_clean_contract(tmp_path):
    collector = SuccessEpisodeCollector(2)
    forward = inputs()
    actions = torch.arange(2 * 50 * 14).reshape(2, 50, 14).float()
    versions = torch.tensor([7, 8])
    collector.append(forward, actions, torch.tensor([True, False]),
                     torch.tensor([[False], [True]]), versions)
    episodes = collector.drain()
    assert len(episodes) == 1 and len(episodes[0]) == 1
    record = episodes[0][0]
    assert set(record) == set(forward) | {"action", "action_valid_mask", "query_idx", "episode_id", "policy_version"}
    assert torch.equal(record["action"], actions[0].flatten())
    assert torch.equal(record["episode_id"], torch.tensor([0, 0]))
    assert record["policy_version"].item() == 7
    actions.zero_()
    assert record["action"].sum() > 0  # Label remains a clone of submitted command.
    replay = SuccessReplay(seed=5, archive_path=str(tmp_path / "archive"), max_success_chunks=3)
    replay.add_episodes(episodes)
    replay.save_checkpoint(tmp_path / "checkpoint")
    restored = SuccessReplay(seed=999, archive_path=str(tmp_path / "unused"), max_success_chunks=3)
    restored.load_checkpoint(tmp_path / "checkpoint")
    expected, actual = replay.sample(4)["forward_inputs"], restored.sample(4)["forward_inputs"]
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)


def test_weighted_masked_loss_keeps_query_mean_and_global_micro_scaling():
    # Masks deliberately give different valid action counts per query.
    mask = torch.tensor([[[1, 1], [0, 0]], [[1, 1], [1, 1]],
                         [[1, 1], [0, 0]], [[1, 1], [1, 1]]], dtype=torch.bool)
    target = torch.tensor([[[1., 3.], [900., 900.]], [[2., 4.], [6., 8.]],
                           [[3., 5.], [900., 900.]], [[4., 6.], [8., 10.]]])
    raw = torch.tensor([0., 0., 1., 3.], dtype=torch.float64)
    global_g = (4 * raw / raw.sum()).requires_grad_()
    parameter = torch.tensor(2., requires_grad=True)
    full = masked_fm_loss(parameter * target, mask, global_g)
    # Query means are [2,5,4,7], g=[0,0,1,3]; no microbatch renormalization.
    assert full.item() == pytest.approx(12.5)
    full.backward()
    assert parameter.grad.item() == pytest.approx(6.25)
    assert global_g.grad is None
    micro_parameter = torch.tensor(2., requires_grad=True)
    combined = sum(masked_fm_loss(micro_parameter * target[start:start+2], mask[start:start+2],
                                  global_g[start:start+2]) / 2 for start in (0, 2))
    combined.backward()
    torch.testing.assert_close(combined, full.detach(), rtol=0, atol=0)
    torch.testing.assert_close(micro_parameter.grad, parameter.grad, rtol=0, atol=0)
    clean = ((target * mask).sum((1, 2)) / mask.sum((1, 2))).mean()
    torch.testing.assert_close(masked_fm_loss(target, mask), clean, rtol=0, atol=0)


def test_sample_weights_reject_shape_nonfinite_and_negative():
    loss = torch.ones(2, 3, 4)
    mask = torch.ones_like(loss, dtype=torch.bool)
    for weights in (torch.ones(2, 1), torch.tensor([1., -1.]),
                    torch.tensor([1., float("nan")]), torch.tensor([1., float("inf")])):
        with pytest.raises(ValueError, match="Sample weights"):
            masked_fm_loss(loss, mask, weights)
