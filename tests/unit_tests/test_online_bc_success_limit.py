# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rlinf.data.online_bc import SuccessReplay


def episode(length):
    return [{"query_idx": torch.tensor(i), "action": torch.tensor([length, i])}
            for i in range(length)]


def test_disabled_exact_replay_and_rng(tmp_path):
    a = SuccessReplay(42, str(tmp_path / "a"))
    b = SuccessReplay(42, str(tmp_path / "b"), max_success_chunks=None)
    data = [episode(3), episode(4)]
    a.add_episodes(data)
    b.add_episodes(data)
    assert len(a) == len(b) == 7
    assert a.get_stats() == b.get_stats()
    assert torch.equal(a.sample(32)["forward_inputs"]["action"],
                       b.sample(32)["forward_inputs"]["action"])


def test_inclusive_whole_episode_limit_archive_and_restore(tmp_path):
    pool = SuccessReplay(42, str(tmp_path / "data"), max_success_chunks=3)
    data = [episode(2), episode(3), episode(4)]
    pool.add_episodes(data)
    saved = torch.load(tmp_path / "data/batch_000000.pt", weights_only=True)
    assert [len(ep) for ep in saved] == [2, 3]
    assert len(data[-1]) == 4  # Never mutate/truncate the rejected trajectory.
    assert len(pool) == 5 and pool.episodes == 2
    assert pool.get_stats()["filtered_success_episodes"] == 1
    pool.save_checkpoint(tmp_path / "ckpt")
    expected = pool.sample(16)
    restored = SuccessReplay(9, str(tmp_path / "restored"), max_success_chunks=3)
    restored.load_checkpoint(tmp_path / "ckpt")
    assert restored.get_stats() == pool.get_stats()
    assert torch.equal(expected["forward_inputs"]["action"],
                       restored.sample(16)["forward_inputs"]["action"])


def test_all_rejected_keeps_empty_pool_and_does_not_write_archive(tmp_path):
    path = tmp_path / "empty"
    pool = SuccessReplay(42, str(path), max_success_chunks=3)
    pool.add_episodes([episode(4)])
    pool.add_episodes([])
    assert not pool.is_ready() and pool.archive_id == 0 and not path.exists()
    assert pool.filtered_success_episodes == 1


@pytest.mark.parametrize("limit", [0, -1, 3.5, True, "3"])
def test_invalid_limit(limit, tmp_path):
    with pytest.raises(ValueError, match="max_success_chunks"):
        SuccessReplay(1, str(tmp_path), max_success_chunks=limit)
