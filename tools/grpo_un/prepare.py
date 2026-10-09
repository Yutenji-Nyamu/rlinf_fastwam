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


"""Prepare one-round GRPO smoke configs from the exact Clean256 runtime."""

import copy
import hashlib
import json
import os
import pwd
import shutil
import socket
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

UID = 20001
ROOT = Path("/data/chenyiteng/deployment-20261009/grpo-un-smoke-g67-v1")
REPO = Path("/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-u-n-20261009")
OLD = Path(
    "/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/sz2-pi05-grpo-turn-switch-clean-256-seed42-formal200-phys45-20260923-v1"
)
OLD_REPO = (
    "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-turn256-sz2-20260923"
)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def rewrite(v, pairs):
    if isinstance(v, dict):
        return {k: rewrite(x, pairs) for k, x in v.items()}
    if isinstance(v, list):
        return [rewrite(x, pairs) for x in v]
    if isinstance(v, str):
        for a, b in pairs:
            v = v.replace(a, b)
    return v


def flat(x, p=""):
    if isinstance(x, dict):
        return {
            a: b
            for k, v in x.items()
            for a, b in flat(v, p + "." + k if p else k).items()
        }
    return {p: x}


assert os.getuid() == UID and socket.gethostname() == "h100-gpu02"
os.umask(0o077)
base = read(OLD / "runtime/resolved.yaml")
env0 = read(OLD / "runtime/environment.json")
S = ROOT / "scope"
boot = S / "bootstrap"
boot.mkdir(parents=True, exist_ok=False)
(S / "receipts").mkdir()
xml = ET.fromstring(
    subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=25)
)
cards = {}
for g in (6, 7):
    node = xml.findall("gpu")[g]
    cards[str(g)] = {
        "physical_gpu": g,
        "gpu_uuid": node.findtext("uuid"),
        "minor": int(node.findtext("minor_number")),
        "pci": node.findtext("pci/pci_bus_id"),
        "commname": f"grpo-un-g{g}",
    }
    cards[str(g)]["mask"] = 1 << cards[str(g)]["minor"]
bootstrap = """import os
path=os.environ.get("RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST")
if path:
 try:
  from graphics_scope_runtime import install
  install(path)
 except BaseException as exc:
  os.write(2,("GRPO GPU scope failed: "+str(exc)+"\\n").encode());os._exit(78)
"""
(boot / "sitecustomize.py").write_text(bootstrap)
shutil.copyfile(
    REPO / "tools/grpo_un/graphics_scope_runtime.py", boot / "graphics_scope_runtime.py"
)
marker = S / "libgrpo_un_g67_20261009.so"
(S / "marker.c").write_text("int grpo_scope_marker(void) { return 1; }\n")
subprocess.run(
    [
        "cc",
        "-shared",
        "-fPIC",
        "-nostdlib",
        "-Wl,-soname," + marker.name,
        "-o",
        str(marker),
        str(S / "marker.c"),
    ],
    check=True,
    timeout=30,
)
marker.chmod(0o500)
profile = Path(
    "/home/chenyiteng/.nv/nvidia-application-profiles-rc.d/00-grpo-un-g67-20261009.json"
)
save(
    profile,
    {
        "rules": [
            {
                "pattern": {
                    "feature": "commname",
                    "matches": cards[str(g)]["commname"],
                },
                "profile": {
                    "name": "chenyiteng-" + cards[str(g)]["commname"],
                    "settings": ["EGLVisibleDGPUDevices", cards[str(g)]["mask"]],
                },
            }
            for g in (6, 7)
        ]
    },
)
manifest = {
    "schema": 1,
    "uid": UID,
    "hostname": "h100-gpu02",
    "home": pwd.getpwuid(UID).pw_dir,
    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    "physical_gpus": [6, 7],
    "cpu_full_mask_target": 6,
    "cards": cards,
    "token": "grpo-un-g67-20261009",
    "receipts_dir": str(S / "receipts"),
    "existing_profiles": {},
    "profile_search_paths": [],
}
# Dedicated commnames cannot match another task's dedicated commname rules.
# Validate only our immutable scope files per worker; no shared profile scans.
for k, p in [
    ("marker", marker),
    ("profile", profile),
    ("bootstrap", boot / "sitecustomize.py"),
    ("runtime", boot / "graphics_scope_runtime.py"),
]:
    manifest[k + "_path"] = str(p)
    manifest[k + "_sha256"] = sha(p)
save(S / "scope.json", manifest)
frag = {
    "HOME": manifest["home"],
    "RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST": str(S / "scope.json"),
    "LD_PRELOAD": str(marker),
    "__GL_APPLICATION_PROFILE": "1",
    "PYTHONPATH": str(boot)
    + ":"
    + str(REPO)
    + ":/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support",
}
plan = {
    "uid": UID,
    "boot_id": manifest["boot_id"],
    "python": "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python",
    "repo": str(REPO),
    "plan_path": str(ROOT / "plan.json"),
    "ray_address": "127.0.0.1:26379",
    "ray_dashboard_url": "http://127.0.0.1:28266",
    "runs": {},
    "gpus": [6, 7],
    "restore_requests": {
        str(g): read(
            Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/requests")
            / f"rlt-g{g}.json"
        )
        for g in (6, 7)
    },
}
for lane in ("u", "norm"):
    run = (
        Path("/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs")
        / f"grpo-{lane}-turn256-t25-tricks-smoke1-g67-20261009-v1"
    )
    rt = run / "runtime"
    rt.mkdir(parents=True, exist_ok=False)
    cfg = rewrite(copy.deepcopy(base), [(str(OLD), str(run)), (OLD_REPO, str(REPO))])
    cfg["runner"].update(max_steps=1, val_check_interval=-1, save_interval=-1)
    cfg["runner"]["logger"].update(log_path=str(run), experiment_name=run.name)
    overlay = read(REPO / f"tools/grpo_un/{lane}-overlay.json")["algorithm"][
        "dvac_gradient_weighting"
    ]
    cfg["algorithm"]["dvac_gradient_weighting"].update(overlay)
    method = cfg["algorithm"]["dvac_gradient_weighting"]
    method["chunk_dropout"]["enabled"] = True
    method["alpha_schedule"]["enabled"] = True
    for level in ("local", "chunk"):
        method["alpha_schedule"][level]["enabled"] = True
    cfg["cluster"]["node_groups"] = [
        {
            "label": "grpo_un_g67",
            "node_ranks": "0",
            "env_configs": [
                {"node_ranks": "0", "env_vars": [{k: v} for k, v in frag.items()]}
            ],
        }
    ]
    cfg["cluster"]["component_placement"] = {
        "actor, env, rollout": {"node_group": "grpo_un_g67", "placement": "6,7"}
    }
    env = rewrite(copy.deepcopy(env0), [(str(OLD), str(run)), (OLD_REPO, str(REPO))])
    env.update(frag, CLUSTER_NAMESPACE=run.name, PYTHONDONTWRITEBYTECODE="1")
    before, after = flat(base), flat(cfg)
    diff = {
        k: [before.get(k), after.get(k)]
        for k in before.keys() | after.keys()
        if before.get(k) != after.get(k)
    }
    allowed = ("algorithm.dvac_gradient_weighting.", "cluster.", "runner.logger.")
    assert all(
        k.startswith(allowed)
        or k
        in ("runner.max_steps", "runner.val_check_interval", "runner.save_interval")
        or k.endswith(("seeds_path", "save_path", "video_base_dir"))
        for k in diff
    ), diff
    assert (
        cfg["env"]["train"]["total_num_envs"] * cfg["env"]["train"]["rollout_epoch"]
        == 256
    )
    assert (
        cfg["actor"]["global_batch_size"],
        cfg["actor"]["micro_batch_size"],
        cfg["algorithm"]["update_epoch"],
    ) == (512, 32, 2)
    for split in ("train", "eval"):
        assert sha(base["env"][split]["seeds_path"]) == sha(
            cfg["env"][split]["seeds_path"]
        )
    save(rt / "resolved.yaml", cfg)
    save(rt / "environment.json", env)
    save(rt / "baseline-diff.json", diff)
    request = {
        "kind": "grpo_smoke",
        "repo": str(REPO),
        "run": str(run),
        "runtime": str(rt),
        "namespace": run.name,
        "gpus": [6, 7],
    }
    save(ROOT / f"{lane}-request.json", request)
    plan["runs"][lane] = request
save(ROOT / "plan.json", plan)
print(
    json.dumps(
        {
            "prepared": True,
            "plan": str(ROOT / "plan.json"),
            "runs": plan["runs"],
            "gpus": cards,
        }
    )
)
