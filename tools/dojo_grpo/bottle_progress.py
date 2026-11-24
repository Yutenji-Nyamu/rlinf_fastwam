"""One quarter of a training point per first attained native bottle milestone."""
import numpy as np


class BottleProgress:
    def __init__(self, num_envs):
        self.count = np.zeros(num_envs, dtype=np.int32)
        self.levels = np.arange(1, 5, dtype=np.int32)
        self.first_steps = np.full((num_envs, 4), -1, dtype=np.int32)
        self.episode_return = np.zeros(num_envs, dtype=np.float32)

    def advance(self, native_stages, success, active, action_steps):
        # Official success already verifies all four bottles and both arms home.
        # Its detector can fire before the consecutive stage counter catches up.
        reached = np.where(success, 4, np.asarray(native_stages, dtype=np.int32))
        reached = np.where(active, np.maximum(self.count, reached), self.count)
        crossed = (self.count[:, None] < self.levels) & (reached[:, None] >= self.levels)
        np.copyto(self.first_steps, np.asarray(action_steps, dtype=np.int32)[:, None], where=crossed)
        increments = (reached - self.count).astype(np.float32) * 0.25
        self.count = reached
        return increments

    def episode(self, env_index):
        return {
            "bottles": int(self.count[env_index]),
            "first_bottle_steps": self.first_steps[env_index].tolist(),
            "episode_return": float(self.episode_return[env_index]),
        }
