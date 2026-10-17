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
    assert c["num_envs"] == 32 and c["step_limit"] == 200
    assert c["num_steps"] in (5, 10, 15, 20, 30)
    assert c["execute_chunk"] in (10, 20, 30, 50)
    assert c["model"]["openpi"]["action_chunk"] == 50
    assert c["policy"] in ("weak", "strong", "calibration", "high", "low", "random")
    if c["policy"] in ("calibration", "high", "low", "random"):
        assert c["num_steps"] == 10 and c["execute_chunk"] == 50


def choose(policy, u, active, threshold, random_values):
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
    return selected & active


def thresholds(query_files):
    values = [row["u"] for p in query_files for row in read(p)]
    assert values and np.isfinite(values).all()
    return {
        "low": float(np.quantile(values, 1 / 3)),
        "high": float(np.quantile(values, 2 / 3)),
        "calibration_decisions": len(values),
        "target_fraction": 1 / 3,
    }
