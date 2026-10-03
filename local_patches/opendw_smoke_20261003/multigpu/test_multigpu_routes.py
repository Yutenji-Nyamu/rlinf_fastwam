"""CPU checks for the isolated multi-GPU overlay. No HTTP/GPU/SSH calls."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
import opendw_service as service

spec = importlib.util.spec_from_file_location('multigpu_env_test_target', ROOT / 'rlinf/envs/world_model/opendw_robotwin_env.py')
env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env_module)


class Config(dict):
    __getattr__ = dict.__getitem__


class RouteTests(unittest.TestCase):
    def test_exact_rank_assignment_and_legacy_single_url(self):
        cfg = Config(service_urls=['http://127.0.0.1:18946', 'http://127.0.0.1:18947/'])
        self.assertEqual(env_module.resolve_service_route(cfg, 0, 2), ('http://127.0.0.1:18946', 0, 2))
        self.assertEqual(env_module.resolve_service_route(cfg, 1, 2), ('http://127.0.0.1:18947', 1, 2))
        old = Config(service_url='http://127.0.0.1:18941')
        self.assertEqual(env_module.resolve_service_route(old, 0, 1)[0], old.service_url)

    def test_wrong_process_count_and_alias_duplicates_rejected(self):
        cfg = Config(service_urls=['http://127.0.0.1:18946', 'http://127.0.0.1:18947'])
        with self.assertRaises(ValueError):
            env_module.resolve_service_route(cfg, 1, 4)
        with self.assertRaises(ValueError):
            env_module.resolve_service_route(cfg, 2, 2)
        cfg['service_urls'][1] = 'http://localhost:18946/'
        with self.assertRaises(ValueError):
            env_module.resolve_service_route(cfg, 0, 2)

    def test_independent_per_process_state_and_random_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'resets.npz'
            np.savez(path, main_images=np.zeros((4, 8, 8, 3), dtype=np.uint8),
                     wrist_images=np.zeros((4, 2, 8, 8, 3), dtype=np.uint8),
                     states=np.zeros((4, 14), dtype=np.float32), instructions=np.asarray(['adjust bottle'] * 4))
            cfg = Config(seed=0, group_size=8, chunk=32, frame_stride=4, auto_reset=False,
                         max_episode_steps=32, use_fixed_reset_state_ids=False,
                         service_urls=['http://127.0.0.1:18946', 'http://127.0.0.1:18947'],
                         initial_state_path=str(path))
            left = env_module.OpenDWRobotwinEnv(cfg, 32, 0, 2)
            right = env_module.OpenDWRobotwinEnv(cfg, 32, 1, 2)
            left.reset()
            right.reset()
            left.states[:] = 123
            self.assertTrue((right.states == 0).all())
            left.prev_step_reward[:] = 0.6
            self.assertTrue((right.prev_step_reward == 0).all())
            self.assertNotEqual(left._env_generators[0].bit_generator.state,
                                right._env_generators[0].bit_generator.state)
            for env in (left, right):
                ids = env.reset_state_ids.reshape(4, 8)
                self.assertTrue((ids == ids[:, :1]).all())

    def test_physical_inventory_and_row_identity(self):
        inventory = '6, GPU-six, 00000000:CA:00.0\n7, GPU-seven, 00000000:CD:00.0\n'
        self.assertEqual(service.parse_gpu_identity(inventory, 7)['gpu_uuid'], 'GPU-seven')
        with self.assertRaises(RuntimeError):
            service.parse_gpu_identity(inventory, 4)
        with self.assertRaises(RuntimeError):
            service.parse_gpu_identity(inventory + inventory, 6)
        data = {'images': np.empty((2,)), 'env_indices': np.array([0, 1]),
                'global_env_indices': np.array([32, 33]), 'reset_ids': np.array([9, 9]),
                'env_process_index': np.array(1), 'env_process_count': np.array(2)}
        identity = service.row_identity(data, 1)
        self.assertEqual(identity, dict(env_index=1, reset_id=9, global_env_index=33,
                                        env_process_index=1, env_process_count=2))


if __name__ == '__main__':
    unittest.main()
