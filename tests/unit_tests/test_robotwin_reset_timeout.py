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


import importlib.util
from concurrent.futures import Future, TimeoutError
from pathlib import Path
from unittest.mock import patch

import pytest

path = Path(__file__).parents[2] / "rlinf/envs/robotwin/reset_timeout.py"
spec = importlib.util.spec_from_file_location("reset_timeout", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Pending:
    def __init__(self, finish):
        self.finish, self.calls = finish, []

    def done(self):
        return False

    def result(self, timeout=None):
        self.calls.append(timeout)
        if len(self.calls) == 1 or not self.finish:
            raise TimeoutError()
        return "same-reset-result"


def test_delayed_reset_waits_same_future():
    f = Pending(True)
    with patch.object(m.faulthandler, "dump_traceback") as dump:
        assert m.wait_for_reset(f, 3) == "same-reset-result"
    assert f.calls == [120, 480]
    dump.assert_called_once()


def test_hang_is_bounded():
    f = Pending(False)
    with patch.object(m.faulthandler, "dump_traceback"):
        with pytest.raises(TimeoutError, match="exceeded 600s"):
            m.wait_for_reset(f, 1)
    assert f.calls == [120, 480]


def test_task_exception_preserved():
    f = Future()
    error = TimeoutError("inside reset")
    f.set_exception(error)
    with pytest.raises(TimeoutError) as caught:
        m.wait_for_reset(f, 0)
    assert caught.value is error


def test_ready_result_unchanged():
    f = Future()
    value = object()
    f.set_result(value)
    assert m.wait_for_reset(f, 0) is value
