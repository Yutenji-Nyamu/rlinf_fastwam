"""Mock-service CPU contract checks for execution in the server RLinf checkout."""

import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rlinf.envs.world_model.opendw_robotwin_env import OpenDWRobotwinEnv


class Config(dict):
    __getattr__ = dict.__getitem__


class FakeServiceEnv(OpenDWRobotwinEnv):
    def _post(self, endpoint, payload, content_type):
        self.requests.append(endpoint)
        if endpoint != "/infer":
            return json.dumps({"ok": True, "is_offloaded": endpoint == "/offload"}).encode()
        with np.load(io.BytesIO(payload), allow_pickle=False) as request:
            self.last_request = {key: request[key].copy() for key in request.files}
        batch = len(self.last_request["states"])
        scores = np.tile(np.linspace(0.1, 0.4, 8, dtype=np.float32), (batch, 1))
        scores[0, 1] = 0.95
        result = io.BytesIO()
        np.savez(result, next_images=self.last_request["images"], scores=scores)
        return result.getvalue()


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "reset.npz"
        head = np.full((2, 256, 256, 3), 20, dtype=np.uint8)
        wrists = np.zeros((2, 2, 256, 256, 3), dtype=np.uint8)
        wrists[:, 0, ::2] = 255
        wrists[:, 1] = 80
        self.native_wrists = wrists
        np.savez(path, main_images=head, wrist_images=wrists,
                 states=np.zeros((2, 14), dtype=np.float32),
                 instructions=np.asarray(["adjust the bottle", "adjust the bottle"]))
        self.cfg = Config(seed=42, group_size=8, chunk=32, frame_stride=4,
                          auto_reset=False, max_episode_steps=64,
                          service_url="http://127.0.0.1:8080",
                          initial_state_path=str(path))
        self.env = FakeServiceEnv(self.cfg, 16, 0, 1)
        self.env.requests = []

    def test_native_reset_is_not_downsampled_via_world_model_canvas(self):
        obs, _ = self.env.reset()
        self.assertEqual(obs["wrist_images"].shape, (16, 2, 256, 256, 3))
        np.testing.assert_equal(obs["wrist_images"].numpy(), self.native_wrists[self.env.reset_state_ids])
        ids = self.env.reset_state_ids.reshape(2, 8)
        self.assertTrue(np.all(ids == ids[:, :1]))

    def test_step_contract_done_padding_and_independent_seeds(self):
        self.env.reset()
        actions = np.zeros((16, 32, 14), dtype=np.float32)
        actions[:, -1, 0] = np.arange(16) / 100
        actions[..., 6] = 1.5
        obs, reward, term, trunc, infos = self.env.chunk_step(actions)
        self.assertEqual(reward.shape, (16, 32))
        self.assertTrue(term[0, -1])
        self.assertFalse(trunc.any())
        self.assertEqual(infos[0]["first_success_action"][0], 8)
        np.testing.assert_equal(obs[0]["states"][:, 0].numpy(), actions[:, -1, 0])
        self.assertTrue(np.all(obs[0]["states"][:, 6].numpy() == 1))
        self.assertEqual(len(set(self.env.last_request["seeds"].tolist())), 16)
        self.assertTrue(np.all(self.env.last_request["actions"][..., 6] == 1))
        _, reward2, _, trunc2, _ = self.env.chunk_step(actions)
        self.assertTrue(np.all(reward2[0].numpy() == 0))
        self.assertEqual(len(self.env.last_request["states"]), 15)
        self.assertTrue(trunc2[1:, -1].all())
        self.env.offload()
        self.assertEqual(self.env.requests[-1], "/offload")
        self.assertTrue(self.env._is_offloaded)

    def test_snapshot_restores_images_and_random_streams(self):
        first, _ = self.env.reset()
        snapshot = self.env.get_state()
        expected = [generator.integers(2**31) for generator in self.env._env_generators]
        self.env.states[:] = 99
        self.env.set_state(snapshot)
        actual = [generator.integers(2**31) for generator in self.env._env_generators]
        self.assertEqual(actual, expected)
        np.testing.assert_equal(self.env._wrap_obs()["states"].numpy(), first["states"].numpy())
        np.testing.assert_equal(self.env._wrap_obs()["wrist_images"].numpy(), first["wrist_images"].numpy())

    def test_service_routing_uses_worker_seed_offset(self):
        cfg = Config(self.cfg)
        cfg["service_urls"] = ["http://127.0.0.1:8080", "http://127.0.0.1:8081"]
        env = FakeServiceEnv(cfg, 8, 1, 2)
        self.assertEqual(env.service_url, "http://127.0.0.1:8081")


if __name__ == "__main__":
    unittest.main()
