# Copyright 2025 The RLinf Authors.
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


"""Bounded wait for native resets; keep submitted work and seed selection intact."""

import faulthandler
import logging
import sys
from concurrent.futures import Future, TimeoutError
from typing import Any


def wait_for_reset(
    future: Future, env_id: int, warn_seconds: float = 120, hard_seconds: float = 600
) -> Any:
    """Diagnose a slow reset, then wait on the original future with a finite cap."""
    # Same finite-wait pattern as the validated October3 RoboTwin step recovery.
    try:
        return future.result(timeout=warn_seconds)
    except TimeoutError:
        if future.done():
            return future.result()
        logging.warning(
            "RoboTwin SubEnv %s reset pending after %ss (including lock queue); "
            "waiting on the same reset up to %ss total",
            env_id,
            warn_seconds,
            hard_seconds,
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        try:
            return future.result(timeout=hard_seconds - warn_seconds)
        except TimeoutError as error:
            if future.done():
                return future.result()
            raise TimeoutError(
                f"SubEnv {env_id} reset exceeded {hard_seconds}s; reset was not repeated"
            ) from error


def reset_with_bounded_wait(
    self: Any, env_idx: Any = None, env_seeds: Any = None
) -> None:
    """Keep native reset submission and seed mapping; replace only result waits."""
    # Native VectorEnv.reset submission order and seed mapping are unchanged.
    import torch

    if len(self.envs) == 0:
        self._init_envs()
    if env_idx is None:
        env_idx = list(range(self.n_envs))
    elif isinstance(env_idx, (list, tuple)):
        env_idx = list(env_idx)
    elif isinstance(env_idx, torch.Tensor):
        env_idx = env_idx.tolist()
    else:
        env_idx = [env_idx]
    futures = {}
    for idx in env_idx:
        if 0 <= idx < self.n_envs:
            seed = None
            if env_seeds is not None and len(env_seeds) == len(env_idx):
                seed = env_seeds[env_idx.index(idx)]
            futures[idx] = self.env_thread_pool.submit(
                self.envs[idx].reset, env_seed=seed
            )
    for idx in env_idx:
        if 0 <= idx < self.n_envs:
            try:
                wait_for_reset(futures[idx], idx)
            except Exception as error:
                raise RuntimeError(
                    f"SubEnv {idx} reset error: {type(error).__name__}: {error!r}"
                ) from error
