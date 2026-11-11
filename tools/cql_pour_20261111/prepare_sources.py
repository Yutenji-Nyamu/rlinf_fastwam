import json
import os
import shutil
import subprocess
import time
from pathlib import Path

assert os.getuid() == 1003
root = Path("/data/chenyiteng/projects/rlinf-shenzhen/exports/cql-pour-20261111")
base = Path("/data/chenyiteng/deployment-20261111/cql-pour")
assert not root.exists() and not base.exists()
base.mkdir(parents=True)
a = json.loads(
    Path(
        "/data/chenyiteng/deployment-20261105/dsrl-fp32-accel/active-runs.json"
    ).read_text()
)
(base / "previous-active.json").write_text(json.dumps(a, indent=2))
for method, g in [("dsrl", "4"), ("rlt", "6")]:
    src = Path(a["slots"][g]["source"])
    dst = root / method
    shutil.copytree(
        src,
        dst,
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", ".git", ".pytest_cache", "*.pyc"),
    )
    package = dst / "tools/cql_pour_20261111"
    package.mkdir()
    for f in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(f, package / f.name)
download = (root / "rlt/tools/dsrl_rlt_warmstart/download_dojo.py").read_text()
download = download.replace(
    "/data/chenyiteng/datasets/robodojo-rlt-dustbin-20261106",
    "/home/chenyiteng/datasets/robodojo-rlt-pour-20261111",
).replace(
    'if "bottles" in x["task"] and "dustbin" in x["task"]',
    'if x["task"].strip().lower() == "pour the liquid from the bottle into the cup."',
)
script = root / "rlt/tools/cql_pour_20261111/download_pour.py"
script.write_text(download)
with (base / "download.log").open("x") as f:
    p = subprocess.Popen(
        ["/usr/bin/python3", "-u", "-B", str(script)],
        stdout=f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
(base / "download-process.json").write_text(
    json.dumps({"pid": p.pid, "time": time.time()})
)
print(json.dumps({"base": str(base), "root": str(root), "download_pid": p.pid}))
