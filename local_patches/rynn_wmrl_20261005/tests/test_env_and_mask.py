"""Server CPU fixtures: no Ray launch, CUDA or HTTP connections."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rlinf.envs.world_model.opendw_robotwin_env import OpenDWRobotwinEnv
from rlinf.envs.world_model.rynn_success import sample_history, validate_success_reply
from rlinf.envs.world_model.rynn_actor_mask import mask_invalid_rynn_groups, rynn_effective_group_metrics


class Config(dict):
    __getattr__ = dict.__getitem__


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reset_path = Path(self.tmp.name) / "reset.npz"
        np.savez(self.reset_path, main_images=np.zeros((2, 24, 32, 3), dtype=np.uint8),
                 wrist_images=np.zeros((2, 2, 24, 32, 3), dtype=np.uint8),
                 states=np.zeros((2, 14), dtype=np.float32), instructions=np.asarray(["adjust bottle"] * 2))

    def tearDown(self):
        self.tmp.cleanup()

    def make_env(self, decisions, **overrides):
        cfg = Config(seed=42, group_size=8, chunk=32, frame_stride=4,
                     max_episode_steps=384, initial_state_path=str(self.reset_path),
                     service_urls=["http://127.0.0.1:18946", "http://127.0.0.1:18947"],
                     rynn_service_urls=["http://127.0.0.1:18944", "http://127.0.0.1:18945"],
                     reward_source="rynn_success", rynn_invalid_reward_sentinel=-1.0,
                     rynn_run_id="cpu-fixture", rynn_batch_size=4)
        cfg.update(overrides)
        env = OpenDWRobotwinEnv(cfg, 16, 0, 2)
        env.mock_calls = []
        def post(endpoint, payload, content_type, service_url=None):
            env.mock_calls.append((endpoint, service_url))
            if endpoint == "/onload":
                return b'{"ok":true,"is_offloaded":false}'
            if endpoint == "/offload":
                if service_url == env.rynn_url:
                    self.assertEqual(payload, b"")
                return b'{"ok":true,"is_offloaded":true}'
            with np.load(io.BytesIO(payload), allow_pickle=False) as request:
                if service_url == env.rynn_url:
                    self.assertEqual(request["frames"].shape[1:], (8, 256, 320, 3))
                    self.assertEqual(int(request["rm_batch_size"]), env.rynn_batch_size)
                    ends = request["end_action_indices"]
                    self.assertTrue(np.all(request["frame_action_indices"][:, -1] == ends))
                    items = []
                    for uid, end in zip(request["episode_uids"], ends):
                        index = int(str(uid).split(":e")[1].split(":")[0])
                        result = decisions(index, int(end))
                        items.append(dict(episode_uid=str(uid), end_action_idx=int(end),
                                          success=result, parse_status="missing_success" if result is None else "ok"))
                    return json.dumps(dict(ok=True, items=items)).encode()
                images = request["images"]
                batch = len(images)
                stream = io.BytesIO()
                heads = np.stack([np.full((8, 256, 320, 3), i + 1, dtype=np.uint8) for i in range(batch)])
                np.savez(stream, next_images=images, scores=np.full((batch, 8), 0.999, dtype=np.float32),
                         future_head_frames=heads)
                return stream.getvalue()
        env._post = post
        env.reset()
        return env

    def test_sparse_success_unknown_and_legacy_scores_are_diagnostic(self):
        env = self.make_env(lambda index, end: None if index == 8 else index == 0)
        _, rewards, terminated, truncated, infos = env.chunk_step(np.zeros((16, 32, 14), dtype=np.float32))
        self.assertEqual(float(rewards[0, -1]), 1)
        self.assertEqual(float(rewards[1:8].sum()), 0)  # old RM .999 is ignored
        self.assertTrue(bool(terminated[0, -1]))
        self.assertEqual(float(rewards[8, -1]), -1)
        self.assertTrue(bool(truncated[8:, -1].all()))
        self.assertTrue(bool(infos[0]["episode"]["rynn_invalid"][8:].all()))
        self.assertFalse(bool(terminated[8:, -1].any()))
        _, later_rewards, _, _, _ = env.chunk_step(np.zeros((16, 32, 14), dtype=np.float32))
        self.assertEqual(float(later_rewards.sum()), 0)  # no repeated terminal success
        self.assertEqual(len(env.head_histories[0]), 9)
        self.assertEqual(len(env.head_histories[1]), 17)

    def test_state_roundtrip_reset_and_stage_offload(self):
        env = self.make_env(lambda index, end: False)
        env.chunk_step(np.zeros((16, 32, 14), dtype=np.float32))
        saved = env.get_state()
        uid = env.episode_uids.copy()
        restored = self.make_env(lambda index, end: False)
        restored.set_state(saved)
        self.assertEqual(restored.episode_uids.tolist(), uid.tolist())
        self.assertTrue(np.array_equal(np.stack(restored.head_histories[3]), np.stack(env.head_histories[3])))
        restored.chunk_step(np.zeros((16, 32, 14), dtype=np.float32))
        self.assertEqual(len(restored.head_histories[3]), 17)
        restored.reset()
        self.assertNotEqual(str(restored.episode_uids[0]), str(uid[0]))
        self.assertEqual(len(restored.head_histories[0]), 1)
        restored.offload()
        self.assertEqual(restored.mock_calls[-2:], [("/offload", None), ("/offload", restored.rynn_url)])

    def test_timeout_and_gate_fail_closed(self):
        env = self.make_env(lambda index, end: False, max_episode_steps=32)
        _, rewards, terminated, truncated, _ = env.chunk_step(np.zeros((16, 32, 14), dtype=np.float32))
        self.assertEqual(float(rewards.sum()), 0)
        self.assertFalse(bool(terminated.any()))
        self.assertTrue(bool(truncated[:, -1].all()))
        gate = Path(self.tmp.name) / "gate.json"
        gate.write_text('{"passed":false,"selected_rm_batch":16}')
        with self.assertRaises(ValueError):
            self.make_env(lambda i, t: False, rynn_batch_size_file=str(gate))
        gate.write_text('{"passed":true,"selected_rm_batch":8}')
        env = self.make_env(lambda i, t: False, rynn_batch_size_file=str(gate))
        self.assertEqual(env.rynn_batch_size, 8)

    def test_sampling_and_reply_identity(self):
        frames, times = sample_history([np.full((256, 320, 3), i, dtype=np.uint8) for i in range(17)])
        self.assertEqual(times.tolist(), [0, 8, 16, 24, 36, 44, 52, 64])
        self.assertEqual(int(frames[-1, 0, 0, 0]), 16)
        with self.assertRaises(ValueError):
            validate_success_reply(dict(ok=True, items=[dict(episode_uid="wrong", end_action_idx=32,
                                                            success=True, parse_status="ok")]), ["expected"], [32])


class ActorMaskTests(unittest.TestCase):
    def batch(self):
        return {"rewards": torch.zeros((3, 16, 32)), "loss_mask": torch.ones((3, 16, 1), dtype=torch.bool)}

    def test_unknown_masks_whole_group_all_times_only(self):
        batch = self.batch()
        batch["rewards"][0, 0, -1] = 1
        batch["rewards"][0, 8, -1] = 1
        batch["rewards"][2, 14, -1] = -1
        metrics = mask_invalid_rynn_groups(batch, 8)
        self.assertEqual(metrics["rynn_invalid_group_count"], 1)
        self.assertFalse(bool(batch["loss_mask"][:, 8:].any()))
        self.assertTrue(bool(batch["loss_mask"][:, :8].all()))
        self.assertEqual(float(batch["rewards"][:, 8:].sum()), 0)
        self.assertEqual(float(batch["rewards"][0, 0, -1]), 1)

    def test_regular_binary_unchanged_and_all_invalid_has_zero_effective_groups(self):
        batch = self.batch()
        batch["rewards"][1, [2, 9], -1] = 1
        before = batch["rewards"].clone()
        mask_invalid_rynn_groups(batch, 8)
        self.assertTrue(torch.equal(batch["rewards"], before))
        self.assertTrue(bool(batch["loss_mask"].all()))
        batch["rewards"][2, [1, 15], -1] = -1
        mask_invalid_rynn_groups(batch, 8)
        self.assertEqual(rynn_effective_group_metrics(batch, 8)["rynn_effective_group_count"], 0)
        self.assertEqual(float(batch["rewards"].sum()), 0)
        self.assertFalse(bool(batch["loss_mask"].any()))


if __name__ == "__main__":
    unittest.main()
