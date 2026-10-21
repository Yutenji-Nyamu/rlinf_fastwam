# Copyright 2026 The RLinf Authors and task-reward contributors.
# Licensed under the Apache License, Version 2.0.
"""Exercise the real OpenDW env across active-slot shrink, reset and rewind."""

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.world_model.opendw_robotwin_env import OpenDWRobotwinEnv


class HistoryTests(unittest.TestCase):
    def test_active_slots_reset_and_state_restore(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reset.npz"
            np.savez(
                p,
                main_images=np.zeros((1, 256, 320, 3), np.uint8),
                wrist_images=np.zeros((1, 2, 128, 160, 3), np.uint8),
                states=np.zeros((1, 14), np.float32),
                instructions=np.asarray(["Lift the pot."]),
            )
            cfg = OmegaConf.create(
                {
                    "seed": 0,
                    "group_size": 1,
                    "reward_backend": "topreward",
                    "reward_mode": "first_success_binary",
                    "ignore_terminations": False,
                    "use_rel_reward": False,
                    "reward_coef": 1.0,
                    "max_episode_steps": 384,
                    "initial_state_path": str(p),
                    "service_url": "http://127.0.0.1:18985",
                    "success_reward_threshold": 0.5,
                }
            )
            env = OpenDWRobotwinEnv(cfg, 3, 0, 1)
            requests = []

            def post(endpoint, payload, content_type):
                if endpoint == "/onload":
                    return b'{"ok":true,"is_offloaded":false}'
                with np.load(io.BytesIO(payload), allow_pickle=False) as z:
                    d = {k: z[k].copy() for k in z.files}
                requests.append(d)
                idx = d["env_indices"]
                history = np.stack(
                    [np.full((8, 256, 320, 3), 10 + int(i), np.uint8) for i in idx]
                )
                scores = np.repeat(
                    np.where(idx == 1, 0.9, 0.1)[:, None], 8, axis=1
                ).astype(np.float32)
                out = io.BytesIO()
                np.savez(
                    out, next_images=d["images"], scores=scores, reward_history=history
                )
                return out.getvalue()

            env._post = post
            actions = np.zeros((3, 32, 14), np.float32)
            env.reset()
            _, rew, done, _, info = env.chunk_step(actions)
            self.assertEqual(rew.sum().item(), 1.0)
            self.assertEqual(info[0]["first_success_action"].tolist(), [-1, 32, -1])
            self.assertEqual(requests[0]["reward_history_lengths"].tolist(), [0, 0, 0])
            saved = env.get_state()
            env.chunk_step(actions)
            self.assertEqual(requests[-1]["env_indices"].tolist(), [0, 2])
            self.assertEqual(
                requests[-1]["reward_history"][:, 0, 0, 0, 0].tolist(), [10, 12]
            )
            env._reward_history.fill(99)
            env.set_state(saved)
            env.chunk_step(actions)
            self.assertEqual(
                requests[-1]["reward_history"][:, 0, 0, 0, 0].tolist(), [10, 12]
            )
            env.reset()
            env.chunk_step(actions)
            self.assertEqual(requests[-1]["env_indices"].tolist(), [0, 1, 2])
            self.assertEqual(requests[-1]["reward_history_lengths"].tolist(), [0, 0, 0])
            self.assertFalse(requests[-1]["reward_history"].any())


if __name__ == "__main__":
    unittest.main()
