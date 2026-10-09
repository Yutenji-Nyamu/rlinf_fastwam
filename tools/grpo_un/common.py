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


"""Small shared lifecycle primitives, reused from the proven RLT wrappers."""

import json
import os
import signal
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(v, indent=2))
    os.replace(tmp, p)


def proc(pid):
    try:
        p = Path("/proc") / str(pid)
        s = (p / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": int(pid),
            "uid": p.stat().st_uid,
            "start": int(s[19]),
            "state": s[0],
        }
    except OSError:
        return None


def same(q):
    a = proc(q["pid"])
    return bool(
        a
        and all(a[k] == q[k] for k in ["pid", "uid", "start"])
        and a["state"] not in ["Z", "X"]
    )


def exact_signal(q, s):
    if not same(q):
        return
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        subprocess.run(
            ["/usr/bin/python3", "-B", __file__, "signal", json.dumps(q), str(int(s))],
            check=True,
        )
        return
    fd = os.pidfd_open(q["pid"])
    try:
        assert same(q) and q["uid"] == os.getuid()
        signal.pidfd_send_signal(fd, s)
    finally:
        os.close(fd)


def actors(p, ns):
    url = p["ray_dashboard_url"].rstrip("/") + "/api/v0/actors?limit=10000&detail=1"
    with urllib.request.urlopen(url, timeout=20) as f:
        r = json.load(f)["data"]["result"]
    assert r.get("num_after_truncation", r.get("total", 0)) < 10000
    return [
        a
        for a in r["result"]
        if a.get("ray_namespace") == ns and a.get("state") != "DEAD"
    ]


def gpus():
    x = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], timeout=25))
    return {
        i: {
            "uuid": g.findtext("uuid"),
            "processes": [
                {"pid": int(p.findtext("pid")), "type": p.findtext("type")}
                for p in g.findall("processes/process_info")
            ],
        }
        for i, g in enumerate(x.findall("gpu"))
    }


def releasable(identity, rows, card):
    return not same(identity) and not rows and not card["processes"]


if __name__ == "__main__":
    import sys

    assert sys.argv[1] == "signal"
    exact_signal(json.loads(sys.argv[2]), int(sys.argv[3]))
