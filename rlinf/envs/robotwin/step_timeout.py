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

"""Bound a slow native step without resubmitting an action or changing physics."""

import faulthandler
import inspect
import logging
import sys
import textwrap
from concurrent.futures import TimeoutError


def wait_for_step(future, env_id, warn_seconds=120, hard_seconds=600):
    # Same-future finite wait already used by the accepted reset recovery.
    try:
        return future.result(timeout=warn_seconds)
    except TimeoutError:
        if future.done():
            return future.result()
        logging.warning(
            "RoboTwin SubEnv %s step pending after %ss; waiting on the same action up to %ss",
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
                f"SubEnv {env_id} step exceeded {hard_seconds}s; action was not repeated"
            ) from error


def bounded_step(original):
    """Reuse the installed native dispatcher, changing its one result wait only."""
    source = textwrap.dedent(inspect.getsource(original))
    anchor = "future.result(timeout=120)"
    if source.count(anchor) != 1:
        raise ValueError("Native step changed; review its result wait before patching")
    namespace = dict(original.__globals__, wait_for_step=wait_for_step)
    exec(
        compile(
            source.replace(anchor, "wait_for_step(future, i)"),
            "<robotwin-step-bounded-wait>",
            "exec",
        ),
        namespace,
    )
    return namespace[original.__name__]
