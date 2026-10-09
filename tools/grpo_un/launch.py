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


"""Launch one authorized smoke after exact namespace and card availability checks."""

import argparse
import os
import subprocess
import time
from pathlib import Path

from common import actors, gpus, proc, read, save

p = argparse.ArgumentParser()
p.add_argument("--plan", required=True)
p.add_argument("--lane", choices=("u", "norm"), required=True)
a = p.parse_args()
plan = read(a.plan)
assert os.getuid() == plan["uid"]
assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == plan["boot_id"]
request = str(Path(a.plan).with_name(a.lane + "-request.json"))
q = read(request)
rt = Path(q["runtime"])
cfg = read(rt / "resolved.yaml")
assert cfg["runner"]["max_steps"] == 1 and q["gpus"] == [6, 7]
assert not (rt / "started.json").exists()
snapshot = gpus()
scope = read(Path(a.plan).parent / "scope/scope.json")
for g in (6, 7):
    assert snapshot[g]["uuid"] == scope["cards"][str(g)]["gpu_uuid"]
    assert not snapshot[g]["processes"], snapshot[g]
assert not actors(plan, q["namespace"])
env = dict(os.environ, **read(rt / "environment.json"))
for name in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
    env.pop(name, None)
probe = subprocess.run(
    [
        plan["python"],
        "-B",
        "-c",
        "import os, graphics_scope_runtime as g; print(g.verify_ray_title_scope(g.read_manifest(os.environ[g.MANIFEST_ENV])))",
    ],
    env=env,
    cwd=q["repo"],
    capture_output=True,
    text=True,
    timeout=40,
)
save(
    rt / "prelaunch.json",
    {
        "time": time.time(),
        "gpus": snapshot,
        "namespace_empty": True,
        "scope_probe_exit": probe.returncode,
        "scope_probe_stdout": probe.stdout,
        "scope_probe_stderr": probe.stderr,
    },
)
assert probe.returncode == 0, probe.stderr
command = [
    plan["python"],
    "-u",
    "-B",
    str(Path(__file__).with_name("driver.py")),
    "--plan",
    a.plan,
    "--request",
    request,
]
with (rt / "driver.log").open("x") as log:
    child = subprocess.Popen(
        command,
        cwd=q["repo"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
identity = proc(child.pid)
save(
    rt / "started.json",
    {"time": time.time(), "identity": identity, "request": request, "command": command},
)
print({"launched": a.lane, "identity": identity, "runtime": str(rt)}, flush=True)
