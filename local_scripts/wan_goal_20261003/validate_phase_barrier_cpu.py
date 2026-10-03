"""Server CPU-only regression checks; executes extracted real runner.run with fakes.

Checks actual phase ordering, fail-closed behavior when env completion raises, and
telemetry's no-CUDA-init contract. Does not import RLinf, load a model or use a GPU.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time
from types import SimpleNamespace


def runner_method(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EmbodiedRunner")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"time": time, "Handle": object}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["run"]


def exercise(method, *, fail_env=False):
    events = []
    state = {"env_done": False}

    class Handle:
        def __init__(self, name): self.name = name
        def wait(self):
            events.append(self.name + ".wait")
            if self.name == "env":
                if fail_env: raise RuntimeError("owned env offload failed")
                state["env_done"] = True
            return []

    class Worker:
        def __init__(self, name): self.name = name
        def set_global_step(self, step): return Handle(self.name + ".set_global_step")
        def interact(self, **kwargs): events.append("env.start"); return Handle("env")
        def generate(self, **kwargs): return Handle("rollout")
        def recv_rollout_trajectories(self, **kwargs): return Handle("actor.recv")
        def compute_advantages_and_returns(self): return Handle("actor.advantages")
        def run_training(self):
            assert state["env_done"], "actor started while env still owns memory"
            events.append("actor.start")
            return Handle("actor.train")

    r = SimpleNamespace(
        cfg=SimpleNamespace(runner={}), global_step=0, max_steps=1,
        actor=Worker("actor"), rollout=Worker("rollout"), env=Worker("env"),
        reward=None, weight_sync_interval=1, overlap_env_bootstrap=False,
        env_channel=None, rollout_channel=None, reward_channel=None, actor_channel=None,
        _should_profile_step=lambda step: False,
        timer=lambda *args, **kwargs: nullcontext(),
        update_rollout_weights=lambda: None,
        _maybe_eval_and_checkpoint=lambda step: {},
        _log_step_metrics=lambda **kwargs: kwargs["env_handle"].wait(),
        _finish_run=lambda: events.append("finished"),
    )
    try:
        method(r)
    except RuntimeError as error:
        assert fail_env and str(error) == "owned env offload failed"
        assert "actor.start" not in events
        return events
    assert not fail_env
    assert events.index("env.wait") < events.index("actor.start")
    assert r.global_step == 1 and events[-1] == "finished"
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--baseline-repo", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    checks = {}
    method = runner_method(args.repo / "rlinf/runners/embodied_runner.py")
    checks["normal_env_offload_before_actor"] = exercise(method)
    checks["env_error_prevents_actor"] = exercise(method, fail_env=True)
    if args.baseline_repo:
        baseline = runner_method(args.baseline_repo / "rlinf/runners/embodied_runner.py")
        try:
            exercise(baseline)
        except AssertionError as error:
            assert "actor started while env still owns memory" in str(error)
            checks["baseline_reproduces_missing_barrier"] = True
        else:
            raise AssertionError("Baseline unexpectedly has an environment barrier")

    class NoCUDA:
        @staticmethod
        def is_initialized(): return False
        def __getattr__(self, name): raise AssertionError("CUDA operation forbidden: " + name)
    old_torch = sys.modules.get("torch")
    previous = os.environ.get("WAN_GOAL_RESOURCE_DIR")
    with tempfile.TemporaryDirectory(prefix="wan-resource-cpu-") as temporary:
        try:
            sys.modules["torch"] = SimpleNamespace(cuda=NoCUDA())
            os.environ["WAN_GOAL_RESOURCE_DIR"] = temporary
            helper = runpy.run_path(str(args.repo / "rlinf/utils/resource_telemetry.py"))
            helper["record_resource_boundary"](SimpleNamespace(_rank=2, version=120), "test", reset_peak=True)
            rows = list(Path(temporary).glob("boundary-*.jsonl"))
            assert len(rows) == 1
            row = json.loads(rows[0].read_text())
            assert row["cuda_initialized"] is False and "cuda_allocated" not in row
            assert row["step"] == 120 and row["rank"] == 2 and row["VmRSS_bytes"] > 0
            checks["telemetry_no_cuda_init"] = True
        finally:
            if old_torch is None: sys.modules.pop("torch", None)
            else: sys.modules["torch"] = old_torch
            if previous is None: os.environ.pop("WAN_GOAL_RESOURCE_DIR", None)
            else: os.environ["WAN_GOAL_RESOURCE_DIR"] = previous
    result = {"ok": True, "time": time.time(), "cuda_used": False, "checks": checks}
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
