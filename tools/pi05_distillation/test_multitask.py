# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for task balance, resume order, instruction isolation, and finite waits."""

import importlib.util
import json
from concurrent.futures import Future, TimeoutError
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from rlinf.algorithms.pi05_distillation.data import BalancedWindows, balanced_indices

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "step_timeout", ROOT / "rlinf/envs/robotwin/step_timeout.py"
)
wait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wait)


def test_balanced_window_cycles_and_resume():
    sizes = [7, 31, 100]
    offsets = np.r_[0, np.cumsum(sizes)]
    cache = {}
    whole = np.concatenate(
        [balanced_indices(sizes, s, 128, 42, cache) for s in range(12)]
    )
    task = np.searchsorted(offsets, whole, side="right") - 1
    assert max(np.bincount(task)) - min(np.bincount(task)) <= 1
    for t, n in enumerate(sizes):
        ids = whole[task == t] - offsets[t]
        for j in range(len(ids) // n):
            assert set(ids[j * n : (j + 1) * n]) == set(range(n))
    resumed = np.concatenate(
        [balanced_indices(sizes, s, 128, 42) for s in range(7, 12)]
    )
    np.testing.assert_array_equal(whole[7 * 128 :], resumed)
    assert set(whole[::2]) | set(whole[1::2]) == set(whole)


def test_task_prompt_namespaces_do_not_collide():
    class Task:
        def __init__(self, prompt):
            self.prompts = {0: prompt}

        def record(self, index):
            return index, self.prompts[0]

    data = BalancedWindows.__new__(BalancedWindows)
    data.datasets = [Task("lift bottle"), Task("press bell")]
    data.offsets = np.array([0, 5, 10])
    assert data.record(2) == (2, "lift bottle")
    assert data.record(7) == (2, "press bell")


def test_evaluation_task_native_budget_and_template_isolation():
    spec = importlib.util.spec_from_file_location(
        "multitask_eval", ROOT / "tools/pi05_distillation/multitask_eval.py"
    )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    template = {"model": {"openpi": {}}, "env": {"task_config": {}, "video_cfg": {}}}
    c = evaluator.make_config(
        template,
        {"task": "open_microwave", "step_limit": 1500},
        5,
        Path("/tmp/evaluation"),
        Path("/tmp/seeds.json"),
        "cp-test",
    )
    assert (
        c["num_envs"] == 16
        and c["step_limit"] == c["env"]["task_config"]["step_lim"] == 1500
    )
    assert c["env"]["task_config"]["task_name"] == "open_microwave"
    assert template["env"]["task_config"] == {}


def test_lean_evaluation_keeps_native_budget_and_disables_recording():
    spec = importlib.util.spec_from_file_location(
        "multitask_eval", ROOT / "tools/pi05_distillation/multitask_eval.py"
    )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    template = {
        "model": {"openpi": {}},
        "env": {"task_config": {}, "video_cfg": {}},
        "record_full_video": True,
        "environment_step_timeout_seconds": 1200,
    }
    c = evaluator.make_config(
        template,
        {"task": "open_microwave", "step_limit": 1500},
        3,
        Path("/tmp/lean-eval"),
        Path("/tmp/seeds.json"),
        "endpoint-test",
        policy="/tmp/student.pt",
        method="endpoint",
        options={
            "episodes_per_task": 8,
            "recording_mode": "none",
            "record_decisions": False,
        },
    )
    assert c["num_envs"] == c["env"]["total_num_envs"] == 8
    assert c["step_limit"] == c["official_step_limit"] == 1500
    assert c["model"]["openpi"]["distill_target_time"] is False
    assert c["environment_step_timeout_seconds"] == 1200
    spec = importlib.util.spec_from_file_location(
        "eval_protocol", ROOT / "tools/pi05_policy_eval/protocol.py"
    )
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    assert protocol.recording_mode(c) == "none"
    assert protocol.recording_mode(template) == "full"
    assert protocol.recording_mode({"record_full_video": False}) == "sparse"


def test_continuation_requires_complete_old_evaluation(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "multitask_eval", ROOT / "tools/pi05_distillation/multitask_eval.py"
    )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    (tmp_path / "onedp").mkdir()
    checkpoint = tmp_path / "onedp/checkpoint-001113.pt"
    checkpoint.touch()
    receipt = {"step": 1113, "path": str(checkpoint)}
    (tmp_path / "onedp/checkpoint.json").write_text(json.dumps(receipt))
    for step in [556, 1113]:
        (tmp_path / f"summary-onedp-s{step:06d}.json").write_text(
            json.dumps({"attempted_tasks": 50, "tasks": [None] * 50})
        )
    assert (
        evaluator.completed_checkpoint(tmp_path, "onedp", [556, 1113, 2226]) == receipt
    )
    (tmp_path / "summary-onedp-s001113.json").write_text(
        json.dumps({"attempted_tasks": 50, "tasks": [None] * 49})
    )
    with pytest.raises(AssertionError):
        evaluator.completed_checkpoint(tmp_path, "onedp", [556, 1113, 2226])


class Pending:
    def __init__(self, finish):
        self.finish, self.calls = finish, []

    def done(self):
        return False

    def result(self, timeout=None):
        self.calls.append(timeout)
        if len(self.calls) == 1 or not self.finish:
            raise TimeoutError()
        return "same-action-result"


def test_slow_step_waits_same_future_once_more():
    f = Pending(True)
    with patch.object(wait.faulthandler, "dump_traceback"):
        assert wait.wait_for_step(f, 0) == "same-action-result"
    assert f.calls == [120, 480]


def test_hang_is_bounded_and_exception_preserved():
    f = Pending(False)
    with (
        patch.object(wait.faulthandler, "dump_traceback"),
        pytest.raises(TimeoutError, match="not repeated"),
    ):
        wait.wait_for_step(f, 0)
    ready = Future()
    error = TimeoutError("inside task")
    ready.set_exception(error)
    with pytest.raises(TimeoutError) as caught:
        wait.wait_for_step(ready, 0)
    assert caught.value is error


class Native:
    def step(self, actions):
        results = []
        for i, future in enumerate(actions):
            results.append(future.result(timeout=120))
        return results


def test_dispatcher_keeps_results_and_order():
    ready = []
    for n in [4, 9]:
        f = Future()
        f.set_result(n)
        ready.append(f)
    assert wait.bounded_step(Native.step)(Native(), ready) == [4, 9]
