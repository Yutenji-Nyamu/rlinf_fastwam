import ast
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from omegaconf import OmegaConf

B = Path("/data/chenyiteng/deployment-20261111/cql-pour")
R = Path("/data/chenyiteng/projects/rlinf-shenzhen/exports/cql-pour-20261111")
ref = R / "rlt/tools/dsrl_rlt_warmstart/reference_eval.py"
s = ref.read_text()
if "reference_eval_episodes" not in s:
    s = s.replace(
        '"""One 8-episode frozen-reference evaluation plus action adapter parity."""',
        '"""Frozen-reference evaluation plus action adapter parity (eight by default)."""',
    )
    s = s.replace(
        "    ec.total_num_envs = 8",
        '    episodes = int(p.get("reference_eval_episodes", 8))\n    assert 0 < episodes <= int(p["num_envs"])\n    ec.total_num_envs = episodes',
    )
    s = (
        s.replace("(ec, 8, 0, 1, None)", "(ec, episodes, 0, 1, None)")
        .replace("torch.zeros(8,", "torch.zeros(episodes,")
        .replace(".reshape(8, -1)", ".reshape(episodes, -1)")
        .replace('"episodes": 8,', '"episodes": episodes,')
    )
    ref.write_text(s)
os.environ.update(json.loads((B / "g4/training-environment.json").read_text()))
sys.path.insert(0, str(R / "dsrl"))
os.chdir(R / "dsrl")
from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster

Cluster.NAMESPACE = "RLinf_cql_pour_config_validation_20261111"
results = {}
for g in ["4", "6"]:
    d = B / ("g" + g)
    p = json.loads((d / "plan.json").read_text())
    for phase in p["pre_native_phases"] + p["phases"]:
        for arg in phase["command"]:
            if str(arg).endswith(".py"):
                assert Path(arg).is_file(), arg
    for f in (d / "config").glob("*.yaml"):
        c = OmegaConf.load(f)
        if f.stem != "stage1":
            validate_cfg(c)
    e = json.loads((d / "training-environment.json").read_text())
    assert e["DOJO_GRPO_DEPLOYMENT"] == str(d)
    assert e.get("DOJO_GRPO_OWNER", str(d)) == str(d)
    assert e["CUDA_VISIBLE_DEVICES"] == ""
    results[g] = {
        "config_compose_validate": True,
        "phase_scripts_present": True,
        "cuda_driver_hidden": True,
        "phase_names": [x["name"] for x in p["pre_native_phases"] + p["phases"]],
    }
for f in [
    R / "dsrl/rlinf/algorithms/conservative_q.py",
    R / "dsrl/rlinf/workers/actor/fsdp_sac_policy_worker.py",
    R / "dsrl/tools/cql_pour_20261111/gpu_smoke.py",
    R / "dsrl/tools/cql_pour_20261111/smoke_worker.py",
    ref,
]:
    ast.parse(f.read_text())
results["dataset"] = json.loads((B / "g6/data-contract.json").read_text())
results["resume"] = json.loads((B / "g4/resume-source.json").read_text())
(B / "validation.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results))
