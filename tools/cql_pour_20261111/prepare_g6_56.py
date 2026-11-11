"""Prepare a 56-environment successor while the 64-environment job still owns GPU6."""

import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from omegaconf import OmegaConf

old = Path("/data/chenyiteng/deployment-20261111/cql-pour/g6")
base = Path("/data/chenyiteng/deployment-20261111/cql-pour-56")
d = base / "g6"
run = Path("/home/chenyiteng/results/cql-pour-20261111/g6-56")
root = Path("/data/chenyiteng/projects/rlinf-shenzhen/exports/cql-pour-20261111/rlt")
schedule = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")
active = json.loads(
    Path(
        "/data/chenyiteng/deployment-20261105/dsrl-fp32-accel/active-runs.json"
    ).read_text()
)
oldp = json.loads((old / "plan.json").read_text())
oldrun = Path(oldp["run"])
cp = oldrun / "smoke/smoke/checkpoints/global_step_1"
assert (old / "smoke-complete.json").is_file() and (cp / "dojo-complete.json").is_file()
state = json.loads(
    (cp / "actor/sac_components/rlt_trainer_state/complete.json").read_text()
)
assert state["complete"] and state["update_step"] == 16
assert active["slots"]["6"]["deployment"] == str(old)
assert active["slots"]["6"]["identity"]["pid"] == 3605114
(d / "config").mkdir(parents=True, exist_ok=False)
(run / "stage1").mkdir(parents=True, exist_ok=False)
mapping = [
    (str(old), str(d)),
    (str(oldrun), str(run)),
    (oldp["namespace"], oldp["namespace"] + "_56"),
]


def relocate(value):
    if isinstance(value, dict):
        return {k: relocate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v) for v in value]
    if isinstance(value, str):
        for a, b in mapping:
            value = value.replace(a, b)
    return value


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


p = relocate(copy.deepcopy(oldp))
p.update(
    num_envs=56,
    pre_registered=True,
    predecessors=[active["slots"]["6"]["identity"]],
    pre_native_phases=[],
    protected_output="/home/chenyiteng/models/protected-eval/cql-pour-20261111/g6-56",
    gpu_admission=json.loads((schedule / "plan.json").read_text())["slots"]["6"],
)
p["phases"] = [x for x in p["phases"] if x["name"] in ("smoke_resume", "formal")]
for phase in p["phases"]:
    phase["timeout"] = 7200 if phase["name"] == "smoke_resume" else None
save(d / "plan.json", p)
token = json.loads((oldrun / "stage1/complete.json").read_text())
save(run / "stage1/complete.json", token)
for phase in ("smoke_resume", "formal"):
    c = OmegaConf.create(
        relocate(
            OmegaConf.to_container(
                OmegaConf.load(old / f"config/{phase}.yaml"), resolve=True
            )
        )
    )
    c.rollout.rlt_feature_model.openpi.rlt_token_path = token["token_path"]
    c.algorithm.rlt_resume.contract.stage1_token_path = token["token_path"]
    c.algorithm.rlt_resume.contract.stage1_token_sha256 = token["token_sha256"]
    for mode in ("train", "eval"):
        c.env[mode].total_num_envs = 56
    c.env.eval.fixed_layout_cycle = 56
    c.runner.resume_dir = str(cp) if phase == "smoke_resume" else None
    if phase == "formal":
        c.runner.max_steps = 90
    OmegaConf.save(c, d / f"config/{phase}.yaml")
for name in ("training-environment.json", "sim-environment.json", "native-g6.json"):
    value = relocate(json.loads((old / name).read_text()))
    if name == "native-g6.json":
        value["num_envs"] = 56
    save(d / name, value)
(d / "rpc").mkdir(mode=0o700)
(d / "rpc/authkey").write_bytes(os.urandom(32))
(d / "rpc/authkey").chmod(0o600)
for name in ("reference-eval.json", "data-contract.json"):
    shutil.copyfile(old / name, d / name)
save(
    d / "reference-origin.json",
    {
        "deployment": str(old),
        "native_envs": 64,
        "evaluated_episodes": 32,
        "reused": True,
    },
)
entries = {}
for gpu in ("4", "6"):
    a = active["slots"][gpu]
    entries[gpu] = {
        "deployment": str(d) if gpu == "6" else a["deployment"],
        "owner": str(d) if gpu == "6" else a["owner"],
        "run": str(run) if gpu == "6" else a["run"],
        "source": str(root) if gpu == "6" else a["source"],
        "old_owner": a["owner"],
        "predecessor": a["identity"],
    }
save(base / "prepared-dsrl.json", {"entries": entries})
e = json.loads((d / "training-environment.json").read_text())
assert e["DOJO_GRPO_DEPLOYMENT"] == str(d) and e["CUDA_VISIBLE_DEVICES"] == ""
os.environ.update(e)
sys.path.insert(0, str(root))
os.chdir(root)
from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster

Cluster.NAMESPACE = "RLinf_cql_pour_56_config_validation_20261111"
for name in ("smoke_resume", "formal"):
    validate_cfg(OmegaConf.load(d / f"config/{name}.yaml"))
receipt = {
    "time": time.time(),
    "prepared_while_previous_owner_running": active["slots"]["6"]["identity"],
    "parallel_before": 64,
    "parallel_after": 56,
    "observed_64_peak_mib": 79092,
    "expected_56_sampling_gib": [69, 72],
    "reason": "64-env real rollout left only about 2.4 GiB on the card; lower parallelism to retain peak headroom.",
    "formal_episode_slots": 5040,
    "original_episode_slots": 5008,
    "formal_rounds": 90,
    "evaluation_episodes": 56,
    "evaluation_unique_layouts": 55,
    "evaluation_repeated_layouts": 1,
    "unchanged": [
        "C10",
        "UTD5",
        "global512/micro256",
        "reference precollection20k",
        "preparation updates30k",
        "BC1/Q0",
        "reference reservoir20k",
    ],
    "resume_smoke_from": str(cp),
    "reused_reference_evaluation": str(old / "reference-eval.json"),
    "token": token["token_path"],
    "config_validation": True,
}
save(base / "validation.json", receipt)
save(d / "episode-budget.json", receipt)
print(json.dumps(receipt))
