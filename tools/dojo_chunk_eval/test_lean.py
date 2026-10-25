# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Protect native action order and active-slot pruning across observation modes."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "lean_deploy", Path(__file__).with_name("deploy.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("chunk", [10, 50])
def test_decision_observations_do_not_change_native_actions(monkeypatch, chunk):
    traces = []
    calls = []
    for boundary in (False, True):
        monkeypatch.setenv("DOJO_U_CONFIG", "unused")
        monkeypatch.setattr(
            module,
            "read",
            lambda _: {"execute_chunk": chunk, "decision_observations_only": boundary},
        )

        class Env:
            physx_monitor_enabled = False

            def __init__(self):
                self.count = [0, 0]
                self.trace = []
                self.obs_calls = 0

            def is_episode_end(self):
                return self.count == [13, 57]

            def get_running_env_idx_list(self):
                return [i for i, limit in enumerate([13, 57]) if self.count[i] < limit]

            def get_obs_batch(self, ids):
                self.obs_calls += 1
                return [{"env_idx": i, "state": self.count[i]} for i in ids]

            def take_action_batch(self, actions, ids):
                for a, i in zip(actions, ids):
                    self.trace.append((i, a))
                    self.count[i] += 1

        class Client:
            def call(self, func_name, obs=None):
                if func_name == "update_obs_batch":
                    self.obs = obs
                if func_name == "get_action_batch":
                    return [[o["state"] + i for i in range(50)] for o in self.obs]

        env = Env()
        module.eval_one_episode_batch(env, Client())
        traces.append(env.trace)
        calls.append(env.obs_calls)
    assert traces[0] == traces[1]
    assert calls[1] < calls[0]
    assert [a for i, a in traces[1] if i == 0] == list(range(13))
    assert [a for i, a in traces[1] if i == 1] == list(range(57))
