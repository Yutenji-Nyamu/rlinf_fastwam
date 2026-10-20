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


"""Prepare one signed-U formal run, preserving the accepted pillbottle200 budget."""

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

UID = os.getuid()
HOST = socket.gethostname()
TAU, CARDS, HOST_ID = {"admin": (1, [6,7], "sz1"), "h100-gpu02": (2, [6,7], "sz2"), "h100-gpu01": (3, [4,5], "sz3")}[HOST]
PAIR = "".join(map(str,CARDS))
ROOT = Path("/data/chenyiteng/deployment-20261020/grpo-u-signed-tau123-v1")
REPO = ROOT / "source"
OLD = Path(
    "/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/grpo-u-tau2-notrick-pillbottle256-sz2-g45-200-1014"
)
OLD_REPO = "/data/chenyiteng/deployment-20261013/grpo-u-tau123-v1/retry-reset1/source"


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


assert UID == (1003 if HOST == "admin" else 20001)
os.umask(0o077)
base = read(ROOT / "baseline.json")
env0 = read(ROOT / "baseline-env.json")
S = ROOT / "scope"
boot = S / "bootstrap"
boot.mkdir(parents=True, exist_ok=False)
(S / "receipts").mkdir()
xml = ET.fromstring(
    subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=25)
)
cards = {}
for g in CARDS:
    node = xml.findall("gpu")[g]
    cards[str(g)] = {
        "physical_gpu": g,
        "gpu_uuid": node.findtext("uuid"),
        "minor": int(node.findtext("minor_number")),
        "pci": node.findtext("pci/pci_bus_id"),
        "commname": f"gs20-g{g}",
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
marker = S / "libgrpo_sign_g23_20261009.so"
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
    "/home/chenyiteng/.nv/nvidia-application-profiles-rc.d/00-grpo-signed-formal-20261020.json"
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
            for g in CARDS
        ]
    },
)
manifest = {
    "schema": 1,
    "uid": UID,
    "hostname": HOST,
    "home": pwd.getpwuid(UID).pw_dir,
    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    "physical_gpus": CARDS,
    "cpu_full_mask_target": CARDS[0],
    "cards": cards,
    "token": "grpo-sign-g23-20261019",
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
site_plan = read("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/plan.json")
plan = {
    "uid": UID,
    "boot_id": manifest["boot_id"],
    "python": "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python",
    "repo": str(REPO),
    "plan_path": str(ROOT / "plan.json"),
    "ray_address": site_plan["ray_address"],
    "ray_dashboard_url": site_plan["ray_dashboard_url"],
    "runs": {},
    "gpus": CARDS,
    "restore_requests": {},
}
for lane in ("signed",):
    run = (
        Path("/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs")
        / f"grpo-u-signed-tau{TAU}-notrick-pillbottle256-{HOST_ID}-g{PAIR}-200-1020"
    )
    rt = run / "runtime"
    rt.mkdir(parents=True, exist_ok=False)
    cfg = rewrite(copy.deepcopy(base), [(str(OLD), str(run)), (OLD_REPO, str(REPO))])
    for split in ("train", "eval"):
        cfg["env"][split]["seeds_path"] = str(ROOT / f"{split}-seeds.json")
    assert cfg["runner"]["max_steps"] == 200
    assert cfg["runner"]["val_check_interval"] == 2 and cfg["runner"]["save_interval"] == 10
    cfg["runner"]["logger"].update(log_path=str(run), experiment_name=run.name)
    overlay = read(REPO / f"tools/grpo_un/sign-{lane}-overlay.json")["algorithm"][
        "dvac_gradient_weighting"
    ]
    cfg["algorithm"]["dvac_gradient_weighting"].update(overlay)
    method = cfg["algorithm"]["dvac_gradient_weighting"]
    method["temperature_local"] = method["temperature_chunk"] = float(TAU)
    method["chunk_dropout"]["enabled"] = False
    method["alpha_schedule"]["enabled"] = False
    for level in ("local", "chunk"):
        method["alpha_schedule"][level]["enabled"] = False
    cfg["cluster"]["node_groups"] = [
        {
            "label": "grpo_sign_g23",
            "node_ranks": "0",
            "env_configs": [
                {"node_ranks": "0", "env_vars": [{k: v} for k, v in frag.items()]}
            ],
        }
    ]
    cfg["cluster"]["component_placement"] = {
        "actor, env, rollout": {"node_group": "grpo_sign_g23", "placement": ",".join(map(str,CARDS))}
    }
    env = rewrite(copy.deepcopy(env0), [(str(OLD), str(run)), (OLD_REPO, str(REPO))])
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost," + HOST
    env.update(frag, CLUSTER_NAMESPACE=run.name, PYTHONDONTWRITEBYTECODE="1")
    env["RAY_ADDRESS"] = site_plan["ray_address"]
    vk = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    if not Path(vk).exists(): vk = "/etc/vulkan/icd.d/nvidia_icd.json"
    env["VK_DRIVER_FILES"] = env["VK_ICD_FILENAMES"] = vk
    assert Path(vk).is_file()
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
        assert sha(ROOT / f"{split}-seeds.json") == sha(cfg["env"][split]["seeds_path"])
    save(rt / "resolved.yaml", cfg)
    save(rt / "environment.json", env)
    save(rt / "baseline-diff.json", diff)
    request = {
        "kind": "grpo_signed_formal",
        "repo": str(REPO),
        "run": str(run),
        "runtime": str(rt),
        "namespace": run.name,
        "gpus": CARDS,
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
