"""Server-only tests of real observation alignment, scoring, and replay state."""
import copy

import pytest
import torch

from rlinf.data.online_bc import SuccessEpisodeCollector
from rlinf.data.online_iql import (
    IQLRynnValueClient, TransitionEpisodeCollector, TransitionReplay,
    snapshot_iql_observation,
)


def observation(t, n=2):
    return {
        "main_images": torch.stack([torch.full((4, 5, 3), t + i * 20, dtype=torch.uint8) for i in range(n)]),
        "wrist_images": torch.stack([torch.full((2, 4, 5, 3), t + i * 20 + 1, dtype=torch.uint8) for i in range(n)]),
        "states": torch.full((n, 14), float(t)),
        "task_descriptions": ["Move the pill bottle onto the blue pad."] * n,
    }


def inputs(obs):
    n = obs["states"].shape[0]
    return {"observation/image": obs["main_images"].clone(),
            "observation/wrist_image": obs["wrist_images"].clone(),
            "observation/state": obs["states"].clone(),
            "tokenized_prompt": torch.ones((n, 8), dtype=torch.int64),
            "tokenized_prompt_mask": torch.ones((n, 8), dtype=torch.bool)}


def append(collector, before, after, success, term, trunc, *, final=None, clean=None):
    n = collector.num_envs
    command = torch.arange(n * 700, dtype=torch.float64).reshape(n, 50, 14) / 1000
    versions = torch.zeros((n, 50, 14))
    term, trunc = torch.tensor(term), torch.tensor(trunc)
    done = term | trunc
    collector.append(inputs(before), command, torch.tensor(success), done, versions,
                     pre_observation=before, post_observation=after, final_observation=final,
                     terminations=term, truncations=trunc)
    if clean is not None:
        clean.append(inputs(before), command, torch.tensor(success), done, versions)
    return command


def episodes(success=True):
    collector = TransitionEpisodeCollector(1, run_id="test", max_commands=100)
    append(collector, observation(0, 1), observation(1, 1), [False], [False], [False])
    append(collector, observation(1, 1), observation(2, 1), [success], [success], [True])
    return collector.drain()


def scorer(tmp_path, monkeypatch):
    calls = []
    fingerprint = {"revision": "test-revision", "model": "fixed8b"}

    def request(self, route, body=None):
        calls.append(route)
        if route == "/health":
            return {"ok": True, "fingerprint": fingerprint}
        return {"ok": True, "fingerprint": fingerprint, "episode_id": body["episode_id"],
                "remaining_seconds": [5., 3., 2.], "delta_seconds": [2., 1.]}

    monkeypatch.setattr(IQLRynnValueClient, "request", request)
    client = IQLRynnValueClient("http://127.0.0.1:18798", tmp_path, "test-revision", "ALOHA", "head camera")
    return client, calls


def test_success_pool_fields_order_unchanged_and_failures_retained():
    collector = TransitionEpisodeCollector(2, run_id="test", max_commands=100)
    clean = SuccessEpisodeCollector(2)
    append(collector, observation(0), observation(1), [False, False], [False, False], [False, False], clean=clean)
    raw = append(collector, observation(1), observation(2), [True, False], [True, False], [True, True], clean=clean)
    full, successes = collector.drain(), clean.drain()
    assert len(full) == 2 and len(successes) == 1
    for plain, transition in zip(successes[0], full[0]):
        for key, value in plain.items():
            assert torch.equal(value, transition[key])
    assert torch.equal(full[1][-1]["action"], raw[1].flatten())
    assert full[1][-1]["action_valid_mask"].all()
    assert int(full[1][-1]["iql_actual_execution_length"]) == -1
    assert bool(full[1][-1]["iql_truncated"])
    assert full[1][-1]["iql_bootstrap_mask"] == 0
    assert full[1][-1]["iql_r_task"] == 0
    assert full[0][-1]["iql_r_task"] == 1  # success + timeout: success wins


def test_terminal_observation_selected_only_for_finished_environment():
    collector = TransitionEpisodeCollector(2, run_id="test", max_commands=100, auto_reset=True)
    before, reset_post, true_terminal = observation(0), observation(1), observation(9)
    append(collector, before, reset_post, [True, False], [True, False], [False, False], final=true_terminal)
    assert torch.equal(collector.completed[0][0]["iql_next_image"], true_terminal["main_images"][0])
    assert torch.equal(collector.pending[1][0]["iql_next_image"], reset_post["main_images"][1])
    assert torch.equal(collector.completed[0][0]["iql_next_wrist_image"], true_terminal["wrist_images"][0])
    append(collector, reset_post, observation(2), [False, False], [False, False], [False, True], final=observation(2))
    full = collector.drain()
    assert [len(ep) for ep in full] == [1, 2]  # no reset episode appended for env0


def test_missing_final_observation_and_interrupted_episode_are_errors():
    collector = TransitionEpisodeCollector(1, run_id="test", auto_reset=True)
    with pytest.raises(ValueError, match="true pre-reset"):
        append(collector, observation(0, 1), observation(1, 1), [True], [True], [False])
    collector = TransitionEpisodeCollector(1, run_id="test")
    append(collector, observation(0, 1), observation(1, 1), [False], [False], [False])
    with pytest.raises(ValueError, match="incomplete"):
        collector.drain()
    with pytest.raises(ValueError, match="unfinished"):
        collector.reset()
    with pytest.raises(ValueError, match="contiguous"):
        append(collector, observation(3, 1), observation(4, 1), [True], [True], [False])


def test_snapshot_freezes_all_cameras_and_rejects_missing_camera():
    raw = observation(0)
    frozen = snapshot_iql_observation(raw, 2)
    raw["main_images"].fill_(99); raw["wrist_images"].fill_(99); raw["states"].fill_(99)
    assert frozen["main_images"][0, 0, 0, 0] == 0
    assert frozen["wrist_images"][0, 0, 0, 0, 0] == 1
    assert frozen["states"][0, 0] == 0
    del raw["wrist_images"]
    with pytest.raises(ValueError, match="Missing"):
        snapshot_iql_observation(raw, 2)


@pytest.mark.parametrize("terminated,truncated,budget", [(True, False, 200), (False, True, 200), (False, False, 50)])
def test_all_task_failure_boundaries_are_absorbing_without_changing_raw_flags(terminated, truncated, budget):
    collector = TransitionEpisodeCollector(1, run_id="test", max_commands=budget)
    append(collector, observation(0, 1), observation(1, 1), [False], [terminated], [truncated])
    row = collector.drain()[0][0]
    assert row["iql_bootstrap_mask"] == 0 and row["iql_r_task"] == 0
    assert bool(row["iql_terminated"]) == terminated
    assert bool(row["iql_truncated"]) == truncated
    assert row["action"].numel() == 700 and row["action_valid_mask"].all()
    assert row["iql_actual_execution_length"] == -1


@pytest.mark.parametrize("success", [True, False])
def test_frozen_scoring_cache_terminal_seconds_and_rewards(tmp_path, monkeypatch, success):
    client, calls = scorer(tmp_path, monkeypatch)
    ep = episodes(success)[0]
    scored, values, key = client.score_episode(ep)
    assert values.tolist() == [5., 3., 2.]
    assert scored[0]["iql_reward"].item() == pytest.approx(.203)
    assert scored[1]["iql_reward"].item() == pytest.approx((1 if success else 0) + .3)
    assert scored[-1]["iql_remaining_after"] == 2  # never overwrite true terminal score
    assert torch.equal(scored[-1]["iql_next_wrist_image"], ep[-1]["iql_next_wrist_image"])
    assert "iql_reward" not in ep[0]
    client.score_episode(ep)
    assert calls.count("/score") == 1
    changed = copy.deepcopy(ep); changed[-1]["iql_next_image"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="cache identity/content"):
        client.score_episode(changed)
    assert (tmp_path / (key + ".json")).exists()


def test_replay_duplicate_rng_isolation_resume_and_attempt_archive(tmp_path, monkeypatch):
    client, _ = scorer(tmp_path / "scores", monkeypatch)
    success = client.score_episode(episodes(True)[0])[0]
    failure = client.score_episode(episodes(False)[0])[0]
    replay = TransitionReplay(7, tmp_path / "attempt1", contract=client.identity)
    global_rng = torch.get_rng_state().clone()
    replay.add_episodes([success, failure])
    replay.add_episodes([success])
    assert len(replay) == 4 and replay.get_stats()["episodes"] == 2
    actor_rng = replay.actor_rng.get_state().clone()
    critic_batch = replay.sample_critic(64)["forward_inputs"]
    assert len(critic_batch["action"]) == 4
    ids = [(bytes(k.tolist()).hex(), int(q)) for k, q in zip(critic_batch["iql_episode_key"], critic_batch["query_idx"])]
    assert len(set(ids)) == 4
    assert torch.equal(actor_rng, replay.actor_rng.get_state())
    assert torch.equal(global_rng, torch.get_rng_state())
    replay.save_checkpoint(tmp_path / "checkpoint")
    expected = replay.sample_actor(10)["forward_inputs"]
    restored = TransitionReplay(999, tmp_path / "attempt2", contract=client.identity)
    restored.load_checkpoint(tmp_path / "checkpoint")
    assert restored.archive_path == tmp_path / "attempt2"
    actual = restored.sample_actor(10)["forward_inputs"]
    assert all(torch.equal(expected[k], actual[k]) for k in expected)
    assert torch.equal(replay.critic_rng.get_state(), restored.critic_rng.get_state())
    altered = copy.deepcopy(success); altered[0]["iql_reward"] += .01
    with pytest.raises(ValueError, match="different content"):
        restored.add_episodes([altered])
    wrong = TransitionReplay(7, tmp_path / "attempt3", contract={"wrong": True})
    with pytest.raises(ValueError, match="contract"):
        wrong.load_checkpoint(tmp_path / "checkpoint")
