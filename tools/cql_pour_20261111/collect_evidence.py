import difflib
import hashlib
import json
import time
from pathlib import Path

B = Path("/data/chenyiteng/deployment-20261111/cql-pour")
R = Path("/data/chenyiteng/projects/rlinf-shenzhen/exports/cql-pour-20261111")
active = json.loads(
    Path(
        "/data/chenyiteng/deployment-20261105/dsrl-fp32-accel/active-runs.json"
    ).read_text()
)
out = B / "evidence"
out.mkdir(exist_ok=True)
old = Path(
    "/data/chenyiteng/projects/rlinf-shenzhen/exports/stability-retry-20261110/dsrl"
)
path = "rlinf/workers/actor/fsdp_sac_policy_worker.py"
(out / "applied-cql.patch").write_text(
    "".join(
        difflib.unified_diff(
            (old / path).read_text().splitlines(True),
            (R / "dsrl" / path).read_text().splitlines(True),
            fromfile="a/" + path,
            tofile="b/" + path,
            n=0,
        )
    )
)
from omegaconf import OmegaConf

diffs = {}


def leaves(x, root=""):
    if isinstance(x, dict):
        z = {}
        for k, v in x.items():
            z.update(leaves(v, root + "." + k if root else k))
        return z
    return {root: x}


for g, old_d in [("4", "parallel32/g4"), ("6", "parallel64/g6")]:
    a = leaves(
        OmegaConf.to_container(
            OmegaConf.load(
                "/data/chenyiteng/deployment-20261110/" + old_d + "/config/formal.yaml"
            ),
            resolve=True,
        )
    )
    b = leaves(
        OmegaConf.to_container(
            OmegaConf.load(
                Path(active["slots"][g]["deployment"]) / "config/formal.yaml"
            ),
            resolve=True,
        )
    )
    diffs[g] = {
        k: {"before": a.get(k), "after": b.get(k)}
        for k in a.keys() | b.keys()
        if a.get(k) != b.get(k)
    }
    (out / ("formal-g" + g + ".yaml")).write_text(
        (Path(active["slots"][g]["deployment"]) / "config/formal.yaml").read_text()
    )
(out / "config-diff.json").write_text(json.dumps(diffs, indent=2))
# Verified once against upstream on 2026-11-11; keep the reviewed source fixed.
commit = "c1de826c7c9256e4d9b6900091e4b0e243d16f5e"
manifest = json.loads(
    Path(
        "/home/chenyiteng/datasets/robodojo-rlt-pour-20261111/subset-manifest.json"
    ).read_text()
)
lock = {
    "verl_vla_commit": commit,
    "dataset_metadata": {
        k: v for k, v in manifest.items() if k not in ["files", "train", "validation"]
    },
    "time": time.time(),
    "files": {},
}
for f in [
    R / "dsrl" / path,
    R / "dsrl/rlinf/algorithms/conservative_q.py",
    R / "rlt/tools/dsrl_rlt_warmstart/reference_eval.py",
]:
    lock["files"][str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
(out / "source-lock.json").write_text(json.dumps(lock, indent=2))
print(json.dumps(lock))
