"""Focused CPU checks. Run on the server with the existing OpenDW dependencies."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

from patch_click_bell import adapter_source, env_source, base_owner_source, formal_owner_source


ROOT = Path(os.environ.get('CLICK_BELL_DONOR_ROOT', Path(__file__).resolve().parents[1] / 'opendw_smoke_20261003'))
DONORS = {
    'adapter': Path(os.environ.get('CLICK_BELL_ADAPTER', ROOT / 'multigpu/rlinf/envs/world_model/opendw_adapter.py')),
    'env': Path(os.environ.get('CLICK_BELL_ENV', ROOT / 'multigpu/rlinf/envs/world_model/opendw_robotwin_env.py')),
    'base_owner': Path(os.environ.get('CLICK_BELL_BASE_OWNER', ROOT / 'multigpu/opendw_multigpu_owner.py')),
    'formal_owner': Path(os.environ.get('CLICK_BELL_FORMAL_OWNER', ROOT / 'formal/opendw_formal_owner.py')),
}


def load_adapter():
    path = DONORS['adapter']
    module = types.ModuleType('click_bell_adapter_test')
    sys.modules[module.__name__] = module
    exec(compile(adapter_source(path.read_text(encoding='utf-8')), str(path), 'exec'), module.__dict__)
    return module


class SparseRewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def feedback(self, scores, **kwargs):
        return self.adapter.map_frame_scores(np.asarray(scores, dtype=np.float32), [0] * len(scores), [0] * len(scores),
            max_episode_steps=384, reward_mode='first_success_binary', relative_reward=False, **kwargs)

    def test_many_high_frames_give_one_reward(self):
        result = self.feedback([[1] * 8])
        self.assertEqual(float(result.rewards.sum()), 1.0)
        self.assertEqual(result.rewards[0, -1], 1.0)
        self.assertTrue(result.terminated[0, -1])
        self.assertEqual(result.first_success_action[0], 4)

    def test_transient_hit_is_latched_at_chunk_boundary(self):
        result = self.feedback([[.1, .2, .95, .2, .1, .1, .1, .1]])
        self.assertEqual(float(result.rewards.sum()), 1.0)
        self.assertEqual(result.first_success_action[0], 12)
        self.assertAlmostEqual(float(result.next_score[0]), .1)

    def test_no_hit_is_zero_even_if_progress_increases(self):
        result = self.feedback([[.1, .2, .3, .4, .5, .6, .7, .89]])
        self.assertEqual(float(result.rewards.sum()), 0.0)
        self.assertFalse(result.terminated.any())

    def test_timeout_can_coincide_with_success(self):
        result = self.adapter.map_frame_scores([[1] * 8], [0], [352], max_episode_steps=384,
            reward_mode='first_success_binary', relative_reward=False)
        self.assertEqual(float(result.rewards.sum()), 1.0)
        self.assertTrue(result.truncated[0, -1])

    def test_nonfinite_and_relative_are_rejected(self):
        with self.assertRaises(ValueError):
            self.feedback([[float('nan')] * 8])
        with self.assertRaises(ValueError):
            self.adapter.map_frame_scores([[1] * 8], [0], [0], max_episode_steps=384,
                reward_mode='first_success_binary', relative_reward=True)

    def test_old_difference_reward_stays_identical(self):
        result = self.adapter.map_frame_scores([[.1] * 8], [0], [0], max_episode_steps=384)
        self.assertAlmostEqual(float(result.rewards.sum()), .1)

    def test_all_source_hooks_are_exact_and_compile(self):
        rows = [
            (DONORS['env'], env_source),
            (DONORS['base_owner'], base_owner_source),
            (DONORS['formal_owner'], formal_owner_source),
        ]
        for donor, transform in rows:
            result = transform(donor.read_text(encoding='utf-8'))
            compile(result, str(donor), 'exec')
            with self.assertRaises(ValueError):
                transform(result)

    def test_environment_never_rewards_a_completed_row_twice(self):
        adapter_name = 'rlinf.envs.world_model.opendw_adapter'
        previous = sys.modules.get(adapter_name)
        sys.modules[adapter_name] = self.adapter
        try:
            module = types.ModuleType('click_bell_env_test')
            path = DONORS['env']
            exec(compile(env_source(path.read_text(encoding='utf-8')), str(path), 'exec'), module.__dict__)
        finally:
            if previous is None:
                del sys.modules[adapter_name]
            else:
                sys.modules[adapter_name] = previous

        class Config(dict):
            __getattr__ = dict.__getitem__

        class FakeEnv(module.OpenDWRobotwinEnv):
            def _post(self, endpoint, payload, content_type):
                if endpoint != '/infer':
                    return json.dumps({'ok': True, 'is_offloaded': endpoint == '/offload'}).encode()
                with np.load(io.BytesIO(payload), allow_pickle=False) as request:
                    images = request['images'].copy()
                self.last_batch = len(images)
                scores = np.full((len(images), 8), .1, dtype=np.float32)
                scores[0] = .99
                response = io.BytesIO()
                np.savez(response, next_images=images, scores=scores)
                return response.getvalue()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reset.npz'
            np.savez(path, main_images=np.zeros((1, 8, 8, 3), dtype=np.uint8),
                     wrist_images=np.zeros((1, 2, 8, 8, 3), dtype=np.uint8),
                     states=np.zeros((1, 14), dtype=np.float32), instructions=np.asarray(['click the bell']))
            cfg = Config(seed=0, group_size=8, chunk=32, frame_stride=4, max_episode_steps=64,
                         initial_state_path=str(path), service_url='http://127.0.0.1:18999',
                         reward_mode='first_success_binary', use_rel_reward=False, reward_coef=1.0)
            env = FakeEnv(cfg, 8, 0, 1)
            env.reset()
            actions = np.zeros((8, 32, 14), dtype=np.float32)
            _, first, done, _, _ = env.chunk_step(actions)
            self.assertEqual(first[0].sum().item(), 1.0)
            self.assertTrue(done[0, -1])
            _, second, _, _, info = env.chunk_step(actions)
            self.assertEqual(second[0].sum().item(), 0.0)
            self.assertEqual(env.last_batch, 7)
            self.assertEqual(info[0]['episode']['return'][0].item(), 1.0)


if __name__ == '__main__':
    unittest.main()
