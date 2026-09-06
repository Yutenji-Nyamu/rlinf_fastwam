import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.utils.dvac_telemetry import DVACEpisodeWriter, DVACTelemetryWriter


class _SamplerConfig:
    num_steps = 4
    action_horizon = 3
    action_dim = 5
    action_chunk = 3
    action_env_dim = 2
    noise_method = "flow_ode"
    joint_logprob = False
    ignore_last = False
    is_nft = False


class _FakeSampler:
    config = _SamplerConfig()
    use_vlm_value = False

    def __init__(self):
        self.velocities = []
        self.action_in_proj = SimpleNamespace(
            weight=torch.empty((), dtype=torch.float32)
        )

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        sample_method,
        num_steps,
        compute_values,
    ):
        del prefix_pad_masks, past_key_values, sample_method, compute_values
        velocity = torch.full_like(x_t, float(idx + 1))
        self.velocities.append(velocity.clone())
        return (
            x_t - velocity / num_steps,
            torch.zeros_like(x_t),
            torch.zeros(x_t.shape[:2], device=x_t.device),
            velocity,
        )

    def _get_timesteps(self, denoise_steps, device):
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        return torch.cat([timesteps, torch.zeros(1, device=device)])

    def _init_nft_state(self, *args):
        return None

    def _update_nft_state(self, *args):
        return None

    def get_logprob_norm(self, sample, mu, sigma):
        del mu, sigma
        return torch.zeros_like(sample)


def _sample(fake, noise, telemetry):
    return OpenPi0ForRLActionPrediction._sample_actions_with_prefix_cache(
        fake,
        state=torch.zeros(noise.shape[0], 14),
        prefix_output=None,
        prefix_pad_masks=None,
        past_key_values=None,
        noise=noise.clone(),
        mode="eval",
        compute_values=False,
        return_dvac_telemetry=telemetry,
    )


def test_sampler_telemetry_preserves_actions_and_rng_sequence():
    noise = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5) / 10

    torch.manual_seed(7)
    without = _sample(_FakeSampler(), noise, telemetry=False)
    rng_without = torch.random.get_rng_state()

    fake_with = _FakeSampler()
    torch.manual_seed(7)
    with_telemetry = _sample(fake_with, noise, telemetry=True)
    rng_with = torch.random.get_rng_state()

    torch.testing.assert_close(without["actions"], with_telemetry["actions"])
    torch.testing.assert_close(without["chains"], with_telemetry["chains"])
    assert torch.equal(rng_without, rng_with)

    telemetry = with_telemetry["dvac_telemetry"]
    assert telemetry["x_chain"].shape == (2, 5, 3, 2)
    assert telemetry["z_endpoint"].shape == (2, 4, 3, 2)
    assert telemetry["timesteps"].shape == (4,)
    assert telemetry["final_model_action"].shape == (2, 3, 2)
    for idx, velocity in enumerate(fake_with.velocities):
        expected = (
            with_telemetry["chains"][:, idx]
            - telemetry["timesteps"][idx] * velocity
        )[..., :2]
        torch.testing.assert_close(telemetry["z_endpoint"][:, idx], expected)


def test_rank_local_writers_preserve_query_and_episode_joins(tmp_path):
    writer = DVACTelemetryWriter(
        str(tmp_path),
        rank=0,
        save_query_inputs=True,
        run_metadata={
            "run_id": "test",
            "source_commit": "test",
            "robotwin_commit": "robotwin-test",
            "seed_file_sha256": "seed-test",
            "launch_command": "test launch",
            "run_started_at_utc": "2026-08-20T00:00:00+00:00",
            "hostname": "test-host",
            "model_action_horizon": 3,
            "model_action_dim": 32,
            "active_action_dim": 2,
            "execution_chunk_length": 50,
            "denoising_steps": 4,
            "model_parameter_dtype": "bfloat16",
            "rollout_world_size": 2,
            "env_world_size": 2,
        },
        resolved_config="runner: test\n",
    )
    batch_size, denoise_steps, horizon, action_dim = 2, 4, 3, 2
    metadata = {
        "episode_idx": torch.tensor([8, 9]),
        "eval_epoch": torch.tensor([0, 0]),
        "query_idx": torch.tensor([1, 1]),
        "action_slot_start": torch.tensor([50, 50]),
        "action_chunk": torch.tensor([50, 50]),
        "source_env_rank": torch.tensor([0, 0]),
        "stage_id": torch.tensor([0, 0]),
        "local_env_slot": torch.tensor([0, 1]),
        "reset_id": torch.tensor([101, 202]),
        "video_worker_seed": torch.tensor([0, 0]),
        "video_index": torch.tensor([0, 0]),
        "video_tile_index": torch.tensor([0, 1]),
        "video_pre_frame": torch.tensor([1, 1]),
        "video_post_frame": torch.tensor([2, 2]),
        "video_relpath": ["seed_0/0.mp4", "seed_0/0.mp4"],
        "success_before": torch.tensor([False, True]),
    }
    writer.append(
        telemetry={
            "x_chain": torch.zeros(batch_size, denoise_steps + 1, horizon, action_dim),
            "z_endpoint": torch.ones(batch_size, denoise_steps, horizon, action_dim),
            "timesteps": torch.tensor([1.0, 0.75, 0.5, 0.25]),
            "final_model_action": torch.full(
                (batch_size, horizon, action_dim), 2.0
            ),
        },
        env_action=torch.zeros(batch_size, 50, action_dim),
        env_obs={
            "states": torch.zeros(batch_size, 14),
            "main_images": torch.zeros(batch_size, 4, 5, 3, dtype=torch.uint8),
            "wrist_images": torch.zeros(
                batch_size, 2, 4, 5, 3, dtype=torch.uint8
            ),
        },
        query_metadata=metadata,
    )
    paths = writer.finalize()

    traces = np.load(paths["trace"])
    assert traces["x_chain"].shape == (2, 5, 3, 2)
    assert traces["z_endpoint"].shape == (2, 4, 3, 2)
    assert traces["timesteps"].shape == (4,)
    with open(paths["query_index"], encoding="utf-8", newline="") as handle:
        query_rows = list(csv.DictReader(handle))
    assert [row["reset_id"] for row in query_rows] == ["101", "202"]
    assert all(row["head_image_relpath"] for row in query_rows)
    assert len(list((tmp_path / "query_images").rglob("*.png"))) == 6
    run_manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert run_manifest["run_id"] == "test"
    assert run_manifest["robotwin_commit"] == "robotwin-test"
    assert run_manifest["model_action_horizon"] == 3
    assert run_manifest["model_action_dim"] == 32
    assert run_manifest["active_action_dim"] == 2
    assert run_manifest["model_parameter_dtype"] == "bfloat16"
    assert len(run_manifest["expected_rollout_shards"]) == 2
    assert len(run_manifest["expected_episode_shards"]) == 2

    duplicate_writer = DVACTelemetryWriter(
        str(tmp_path),
        rank=0,
        save_query_inputs=False,
        run_metadata={
            "run_id": "duplicate",
            "source_commit": "test",
            "rollout_world_size": 2,
            "env_world_size": 2,
        },
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        duplicate_writer.finalize()

    episode_writer = DVACEpisodeWriter(str(tmp_path), rank=0)
    episode_writer.append(
        query_metadata=metadata,
        env_info={
            "success_once": torch.tensor([True]),
            "success_at_end": torch.tensor([True]),
            "return": torch.tensor([1.0]),
            "reward": torch.tensor([0.005]),
            "episode_len": torch.tensor([200]),
        },
        newly_done=torch.tensor([True, False]),
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([True, False]),
    )
    episode_path = episode_writer.finalize()
    with open(episode_path, encoding="utf-8", newline="") as handle:
        episode_rows = list(csv.DictReader(handle))
    assert len(episode_rows) == 1
    assert episode_rows[0]["reset_id"] == "101"
    assert episode_rows[0]["termination_reason"] == "truncation"
