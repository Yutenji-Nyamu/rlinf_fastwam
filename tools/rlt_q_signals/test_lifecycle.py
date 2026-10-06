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


import unittest

from coexist_owner import priority_released


class ReleaseTests(unittest.TestCase):
    def terminal(self, lane, gpu):
        return {
            "stage": "COMPLETE",
            "released": True,
            "boot_id": "boot",
            "gpu": gpu,
            "gpu_uuid": "uuid" + str(gpu),
            "operation_id": "rlt-q-signals-g67-v1-" + lane,
        }

    def test_smoke_running_partial_and_failed_cleanup_hold_return(self):
        u, n = self.terminal("u", 6), self.terminal("norm", 7)
        uuids = {6: "uuid6", 7: "uuid7"}
        for terminals in (
            {"u": None, "norm": None},
            {"u": u, "norm": None},
            {"u": dict(u, stage="SMOKE"), "norm": n},
            {"u": dict(u, released=False), "norm": n},
        ):
            self.assertFalse(priority_released(terminals, boot_id="boot", uuids=uuids))
        self.assertTrue(
            priority_released(
                {"u": dict(u, stage="FAILED"), "norm": n}, boot_id="boot", uuids=uuids
            )
        )
        with self.assertRaises(ValueError):
            priority_released(
                {"u": dict(u, gpu=5), "norm": n}, boot_id="boot", uuids=uuids
            )
        with self.assertRaises(ValueError):
            priority_released({"u": u, "norm": n}, boot_id="other-boot", uuids=uuids)


if __name__ == "__main__":
    unittest.main()
