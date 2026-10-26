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

"""Optional synchronized phase timing for short smoke runs only."""

import time
from contextlib import contextmanager

import torch


class StepTiming:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.seconds = {}

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        begin = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - begin
