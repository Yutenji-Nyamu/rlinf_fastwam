"""RLinf adapter over one native multi-environment Isaac process per GPU.

Interface follows RLinf RoboTwin and PR1754. Transport/layout batching replace
PR1754's process-per-environment runtime. No simulator or model weights import
in this process, and train/eval share the same persistent native server.
"""

import time
from multiprocessing.connection import Client

import gymnasium as gym
import numpy as np
import torch

from .protocol import grouped_indices


class RoboDojoEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        self.cfg, self.num_envs, self.seed_offset = cfg, num_envs, seed_offset
        self.group_size = int(cfg.group_size)
        self.mode = "eval" if cfg.is_eval else "train"
        self.is_start = True
        self._rng = np.random.RandomState(cfg.seed + seed_offset)
        # Logical EnvWorker rank maps to the explicit physical pair, never to GPU0.
        self.gpu = int(cfg.physical_gpus[seed_offset % len(cfg.physical_gpus)])
        self.socket = str(cfg.rpc_dir) + f"/gpu{self.gpu}.sock"
        self._key = open(str(cfg.rpc_dir) + "/authkey", "rb").read()
        self.reset_state_ids = torch.zeros(num_envs, dtype=torch.long)
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.long)
        self._returns = torch.zeros(num_envs)
        self._success = torch.zeros(num_envs, dtype=torch.bool)
        self._fixed_offset = seed_offset * num_envs
        self.pool_size = self._rpc("describe")["pools"][self.mode]
        self.update_reset_state_ids()

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def _rpc(self, op, **args):
        timeout_key = "reset_timeout" if op == "reset" else "rpc_timeout"
        deadline = time.monotonic() + float(self.cfg.get(timeout_key, 900))
        restarts = 0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Dojo {op} timed out on GPU {self.gpu}")
            try:
                connection = Client(self.socket, family="AF_UNIX", authkey=self._key)
            except (FileNotFoundError, ConnectionRefusedError):
                if not restarts:
                    raise
                time.sleep(0.5)
                continue
            with connection:
                connection.send(dict(op=op, mode=self.mode, **args))
                if not connection.poll(max(0, deadline - time.monotonic())):
                    raise TimeoutError(f"Dojo {op} timed out on GPU {self.gpu}")
                result = connection.recv()
            if result.get("restart_native"):
                restarts += 1
                if op != "reset" or restarts > 3:
                    raise RuntimeError("Dojo reset native restart limit exceeded")
                time.sleep(0.5)
                continue
            if not result["ok"]:
                raise RuntimeError("Dojo native error: " + result["error"])
            return result["value"]

    @staticmethod
    def _obs(value):
        return {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in value.items()
        }

    def update_reset_state_ids(self, env_idx=None):
        assert env_idx is None, "Native GRPO resets complete waves only"
        ids = grouped_indices(
            self.pool_size,
            self.num_envs,
            self.group_size,
            self._rng,
            self._fixed_offset,
            fixed=bool(self.cfg.use_fixed_reset_state_ids),
        )
        self.reset_state_ids = torch.tensor(ids, dtype=torch.long)

    def reset(self, env_idx=None, env_seeds=None, **kwargs):
        assert env_idx is None, "Partial resets would change GRPO group membership"
        ids = self.reset_state_ids.tolist() if env_seeds is None else list(env_seeds)
        value = self._rpc("reset", layout_ids=ids, group_size=self.group_size)
        self.reset_state_ids = torch.tensor(
            value["audit"]["layout_ids"], dtype=torch.long
        )
        self._elapsed_steps.zero_()
        self._returns.zero_()
        self._success.zero_()
        self.is_start = False
        return self._obs(value["obs"]), {"reset_audit": value["audit"]}

    def chunk_step(self, actions):
        if isinstance(actions, dict):
            actions = actions["actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().float().numpy()
        value = self._rpc("step", actions=np.asarray(actions, dtype=np.float32))
        rewards = torch.from_numpy(value["rewards"])
        terms = torch.from_numpy(value["terminations"])
        truncs = torch.from_numpy(value["truncations"])
        self._elapsed_steps = torch.tensor(value["counts"], dtype=torch.long)
        self._returns += rewards.sum(1)
        self._success |= torch.tensor(value["success"], dtype=torch.bool)
        obs = self._obs(value["obs"])
        info = {
            "success": self._success.clone(),
            "episode": {
                "success_once": self._success.clone(),
                "success_at_end": self._success.clone(),
                "return": self._returns.clone(),
                "episode_len": self._elapsed_steps.clone(),
                "reward": self._returns / self._elapsed_steps.clamp_min(1),
            },
        }
        # No auto-reset inside a wave: success latches, later action slots mask out.
        return [obs], rewards, terms, truncs, [info]

    def offload(self, **kwargs):
        pass

    def close(self):
        pass
