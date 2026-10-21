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

"""Exercise queue continuation and physical-card remapping without a GPU."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class RelocationTests(unittest.TestCase):
    def test_checkpoint_is_evaluated_before_further_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            save(
                root / "prepared.json", {"output": str(output), "physical_gpus": [6, 7]}
            )
            for gpu in (6, 7):
                save(root / f"lane-g{gpu}-released.json", {"completed": True})
            save(root / "external-smoke.json", {"host": "sz2", "passed": True})
            for method, total, interval in [("cp", 19000, 1900), ("onedp", 760, 190)]:
                config = {
                    "eval_epochs": interval,
                    "log_every": 10,
                    "checkpoint_steps": interval,
                }
                save(root / f"{method}.json", config)
                save(
                    output / f"smoke-{method}/resolved.json",
                    {"config": config, "total_steps": total, "steps_per_epoch": 1},
                )
            for name in ["resume.pt", "policy.pt"]:
                (root / name).touch()
            save(
                output / "cp/checkpoint.json",
                {
                    "step": 1900,
                    "path": str(root / "resume.pt"),
                    "policy": str(root / "policy.pt"),
                },
            )
            common = types.SimpleNamespace(
                read=lambda p: json.loads(Path(p).read_text()),
                save=save,
                proc=lambda _: {"pid": 1},
                same=lambda _: False,
                exact_signal=lambda *_: None,
                gpus=lambda: {},
            )
            spec = importlib.util.spec_from_file_location(
                "queue_under_test", Path(__file__).with_name("run_queue.py")
            )
            queue = importlib.util.module_from_spec(spec)
            with (
                patch.dict(sys.modules, common=common),
                patch.object(sys, "argv", ["queue", str(root)]),
                patch("signal.signal"),
                patch("os.getuid", return_value=20001),
            ):
                spec.loader.exec_module(queue)
                calls = []
                queue.accept_smoke = lambda _: None
                queue.evaluate = lambda method, receipt, label: calls.append(
                    ("eval", method, receipt["step"])
                )

                def train(method, output, stop, resume):
                    calls.append(("train", method, stop))
                    self.assertTrue(resume if method == "cp" else True)
                    return {"step": stop, "path": "saved.pt", "policy": "policy.pt"}

                queue.train = train
                queue.main()
            self.assertEqual(queue.CARDS, (6, 7))
            self.assertEqual(
                calls[:3],
                [("eval", "cp", 1900), ("train", "cp", 3800), ("eval", "cp", 3800)],
            )
            self.assertNotIn(("train", "cp", 1900), calls)
            self.assertEqual(sum(c[0] == "eval" and c[1] == "cp" for c in calls), 10)
            self.assertEqual(sum(c[0] == "eval" and c[1] == "onedp" for c in calls), 4)
            self.assertEqual(calls[-1], ("eval", "onedp", 760))


if __name__ == "__main__":
    unittest.main()
