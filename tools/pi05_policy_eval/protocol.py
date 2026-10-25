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

"""Small CPU helpers for the frozen paired policy evaluation."""

import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def validate_config(c):
    if c.get("suite") == "pi05_rand_chunk_sweep":
        assert c["num_envs"] in (4, 8, 32) and c["num_steps"] in (1, 10)
        assert c["execute_chunk"] in (10, 50)
        assert c["model"]["openpi"]["action_chunk"] == 50
        assert c["step_limit"] == c["official_step_limit"] > 0
        assert c["env"]["task_config"]["task_name"] == c["task"]
        assert c["policy"] == "weak" and not c.get("collect_weak_u", False)
        assert not any(
            c.get(k) for k in ("weak_weights", "strong_weights", "distill_weights")
        )
        assert c["recording_mode"] == "none" and not c["record_decisions"]
        dr = c["env"]["task_config"]["domain_randomization"]
        assert dr["random_background"] and dr["cluttered_table"] and dr["random_light"]
        return
    if c.get("suite") == "pi05_chunk_sweep":
        assert c["num_envs"] == 8 and c["num_steps"] in (1, 10)
        assert c["execute_chunk"] in (10, 20, 30, 40)
        assert c["model"]["openpi"]["action_chunk"] == 50
        assert c["model"]["num_action_chunks"] == 50
        assert c["step_limit"] == c["official_step_limit"] > 0
        assert c["env"]["task_config"]["task_name"] == c["task"]
        assert c["policy"] == "weak" and not c.get("collect_weak_u", False)
        assert not any(
            c.get(k) for k in ("weak_weights", "strong_weights", "distill_weights")
        )
        assert c["recording_mode"] == "none" and not c["record_decisions"]
        assert c["batch"] in range(4)
        return
    if c.get("suite") in ("pi05_distillation", "pi05_distillation_multitask"):
        if c["suite"] == "pi05_distillation":
            assert c["task"] == "adjust_bottle" and c["num_envs"] == 32
            assert c["step_limit"] == 400
        else:
            assert c["num_envs"] in (8, 16) and c["step_limit"] > 0
            assert c["official_step_limit"] == c["step_limit"]
            assert c["env"]["task_config"]["task_name"] == c["task"]
        assert c["num_steps"] in (1, 10)
        assert c["model"]["openpi"]["action_chunk"] == 50
        allowed_chunks = (
            (10, 50) if c["suite"] == "pi05_distillation_multitask" else (50,)
        )
        assert c["execute_chunk"] in allowed_chunks
        assert c["policy"] == "weak" and not c.get("collect_weak_u", False)
        assert not c.get("weak_weights") and not c.get("strong_weights")
        return
    assert c["num_envs"] == 32 and c["step_limit"] == 200
    assert c["num_steps"] in (5, 10, 15, 20, 30)
    assert c["execute_chunk"] in (10, 20, 30, 50)
    assert c["model"]["openpi"]["action_chunk"] == 50
    assert c["policy"] in ("weak", "strong", "calibration", "high", "low", "random")
    assert c.get("takeover_mode", "weights") in ("weights", "c20")
    if c.get("takeover_mode") == "c20":
        assert c["policy"] in ("high", "low", "random")
        assert c["execute_chunk"] == 50
    if c["policy"] in ("calibration", "high", "low", "random"):
        assert c["num_steps"] == 10 and c["execute_chunk"] in (20, 50)
    assert c.get("max_takeover_queries") in (None, 2, 3)
    if c.get("max_takeover_queries") is not None:
        assert c["policy"] in ("high", "low", "random")


def recording_mode(config):
    """An explicit none mode disables both full and legacy sparse video."""
    mode = config.get(
        "recording_mode", "full" if config.get("record_full_video", False) else "sparse"
    )
    assert mode in {"none", "sparse", "full"}
    return mode


def execution_lengths(selected, active, counts, chunk=50, limit=200, mode="weights"):
    """Per-slot prefixes, clipping the last query to the native action budget."""
    counts = np.asarray(counts)
    assert np.all((counts >= 0) & (counts <= limit))
    lengths = (
        np.where(selected, 20, chunk) if mode == "c20" else np.full_like(counts, chunk)
    )
    return np.where(active, np.minimum(lengths, limit - counts), 0)


def variable_chunk_step(env, actions, lengths):
    """Keep each whole prefix intact for native TOPP; do not split C50 into substeps.

    Native VectorEnv already accepts a list of differently sized prefixes. Its
    completed-task guard runs before reading actions, so inactive slots get an
    ordinary C50 payload and do no additional controller work. This eval path
    reads success/counts directly and needs no rectangular reward tensor.
    """
    payload = [actions[i, : int(n)] if n else actions[i] for i, n in enumerate(lengths)]
    raw_obs, *_ = env.venv.step(payload)
    return env._extract_obs_image(raw_obs)


def configure_recording_wait(env, seconds=300):
    """Allow measured video overhead without changing native actions or physics.

    B32 complete video needed144s for a native block, above its120s default.
    Keep the installed dispatcher byte-for-byte except this wall-clock wait.
    The caller may supply the measured host-specific limit. The lane keeps
    a separate finite stall and whole-batch limit.
    """
    import inspect
    import textwrap
    import types

    assert 300 <= seconds <= 1200
    original = env.venv.step.__func__
    source = textwrap.dedent(inspect.getsource(original))
    anchor = "future.result(timeout=120)"
    assert source.count(anchor) == 1, "Native dispatcher changed; review first"
    namespace = {}
    exec(
        compile(
            source.replace(anchor, f"future.result(timeout={seconds})"),
            "<native-video-wait>",
            "exec",
        ),
        original.__globals__,
        namespace,
    )
    env.venv.step = types.MethodType(namespace[original.__name__], env.venv)


def choose(
    policy, u, active, threshold, random_values, *, query=0, max_takeover_queries=None
):
    """Choose complete strong chunks only for active environments."""
    if policy == "strong":
        selected = np.ones_like(active)
    elif policy == "high":
        selected = u > threshold["high"]
    elif policy == "low":
        selected = u < threshold["low"]
    elif policy == "random":
        selected = random_values < 1 / 3
    else:
        selected = np.zeros_like(active)
    eligible = max_takeover_queries is None or query < max_takeover_queries
    return selected & active & eligible


def thresholds(query_files):
    values = [row["u"] for p in query_files for row in read(p)]
    assert values and np.isfinite(values).all()
    return {
        "low": float(np.quantile(values, 1 / 3)),
        "high": float(np.quantile(values, 2 / 3)),
        "calibration_decisions": len(values),
        "target_fraction": 1 / 3,
    }
