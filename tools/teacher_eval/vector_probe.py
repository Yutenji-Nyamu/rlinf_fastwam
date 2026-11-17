"""Observe native seed/count metadata without changing simulation behavior."""
from robotwin.envs.vector_env import VectorEnv

class SnapshotVectorEnv(VectorEnv):
    def episode_state(self):
        return [dict(accepted_seed=int(e.accepted_seed),
                     instruction=e.task.get_instruction(),
                     native_action_count=int(getattr(e.task,'take_action_cnt',0)),
                     success=bool(getattr(e.task,'eval_success',False)))
                for e in self.envs]
