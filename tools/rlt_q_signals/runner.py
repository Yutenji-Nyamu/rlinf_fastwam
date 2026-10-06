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


"""One-card Q experiment: isolated smoke, fresh formal, then release receipt."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import scoped_ops as op

ST = None


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    p = Path(p)
    tmp = p.with_name(p.name + ".tmp-" + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def checked():
    p = read(ST / "plan.json")
    assert os.getuid() == p["uid"] == 20001 and p["gpu"] in (6, 7)
    assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == p["boot_id"]
    assert set(p["runs"]) == {"smoke", "formal"}
    assert all(r["gpus"] == [p["gpu"]] for r in p["runs"].values())
    assert (
        subprocess.check_output(
            ["git", "-C", p["repo"], "rev-parse", "HEAD"], text=True
        ).strip()
        == p["head"]
    )
    for name, digest in p["pins"].items():
        assert sha(name) == digest, name
    weights = Path(read(ST.parent / "stage1-copy.json")["path"])
    assert weights.is_file() and weights.stat().st_size == 10018009814
    return p


def gpu_rows():
    tree = ET.fromstring(
        subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=25)
    )
    return [
        {
            "gpu": i,
            "uuid": g.findtext("uuid"),
            "pid": int(x.findtext("pid")),
            "type": x.findtext("type"),
            "memory": x.findtext("used_memory"),
        }
        for i, g in enumerate(tree.findall("gpu"))
        for x in g.findall("processes/process_info")
    ]


def exact_signal(identity, sig):
    assert op.same(identity) and identity["uid"] == 20001
    fd = os.pidfd_open(identity["pid"])
    try:
        assert op.same(identity)
        signal.pidfd_send_signal(fd, sig)
    finally:
        os.close(fd)


def scope(p, row, identity):
    actors = op.validate_scoped_actors(p, op.active_actors(p, row["namespace"]))
    roots = {a["pid"] for a in actors if a.get("pid")}
    if identity and op.same(identity):
        roots.add(identity["pid"])
    tree = op.process_tree(roots, p["uid"])
    contexts = gpu_rows()
    outside = [g for g in contexts if g["pid"] in tree and g["gpu"] != p["gpu"]]
    assigned = [g for g in contexts if g["gpu"] == p["gpu"]]
    assert all(g["uuid"] == p["gpu_uuid"] for g in assigned)
    return {
        "actors": actors,
        "tree": list(tree.values()),
        "outside": outside,
        "gpu": assigned,
        "released": not actors and not tree and not assigned,
    }


def scalars(run):
    sys.modules.setdefault("tensorflow", None)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = {}
    for path in (Path(run) / "tensorboard", Path(run) / Path(run).name / "tensorboard"):
        if path.is_dir():
            a = EventAccumulator(str(path), size_guidance={"scalars": 0})
            a.Reload()
            for k in a.Tags()["scalars"]:
                out[k] = [
                    {"step": v.step, "value": v.value, "time": v.wall_time}
                    for v in a.Scalars(k)
                ]
    return out


def smoke_gate(row):
    data = scalars(row["run"])
    train = {k: v for k, v in data.items() if k.startswith("train/")}
    assert train and all(
        math.isfinite(x["value"]) for values in train.values() for x in values
    )
    selected = {
        k: v
        for k, v in train.items()
        if "rlt_q/" in k or "update_step" in k or "grad_norm" in k or "weighted_" in k
    }
    for suffix in ("rlt_q/weight_std", "rlt_q/weight_nonunit_fraction"):
        assert any(
            x["value"] > 0
            for k, values in selected.items()
            if k.endswith(suffix)
            for x in values
        ), suffix
    assert any(
        x["value"] > 0
        for k, values in selected.items()
        if k.endswith("update_step")
        for x in values
    )
    assert any(
        abs(x["value"]) > 0
        for k, values in selected.items()
        if k.endswith("weighted_q")
        for x in values
    )
    assert any(
        x["value"] > 0
        for k, values in selected.items()
        if "grad_norm" in k
        for x in values
    )
    cp = Path(row["run"]) / Path(row["run"]).name / "checkpoints/global_step_2/actor"
    marker = read(cp / "sac_components/rlt_trainer_state/complete.json")
    assert (
        marker["complete"]
        and marker["saved_runner_step"] == 2
        and marker["update_step"] > 0
    )
    assert (cp / "dcp_checkpoint/.metadata").is_file()
    replay = read(cp / "sac_components/replay_buffer/rank_0/metadata.json")
    assert replay["total_samples"] > 0
    return {
        "time": time.time(),
        "finite": True,
        "metrics": selected,
        "checkpoint": str(cp),
        "marker": marker,
        "replay_samples": replay["total_samples"],
    }


def cleanup(p, row, identity, child, seen):
    if identity and op.same(identity):
        exact_signal(identity, signal.SIGTERM)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        current = scope(p, row, identity)
        if current["released"] and all(not op.same(x) for x in seen.values()):
            return current
        if child is not None:
            child.poll()
        time.sleep(3)
    # Only named actors in this namespace and the identity-pinned observed tree.
    import ray

    ray.init(
        address=p["ray_address"],
        namespace=row["namespace"] + "-cleanup",
        ignore_reinit_error=True,
    )
    try:
        actors = op.validate_scoped_actors(p, op.active_actors(p, row["namespace"]))
        op.kill_named_actors(p, actors)
    finally:
        ray.shutdown()
    for ident in seen.values():
        if op.same(ident):
            exact_signal(ident, signal.SIGTERM)
    time.sleep(5)
    for ident in seen.values():
        if op.same(ident):
            exact_signal(ident, signal.SIGKILL)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        current = scope(p, row, identity)
        if current["released"] and all(not op.same(x) for x in seen.values()):
            return current
        time.sleep(3)
    raise RuntimeError("Own namespace/process/GPU release incomplete; RLT must wait")


def owner():
    p = checked()
    lock = (ST / "owner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (ST / "owner-identity.json").exists()
    me = op.proc(os.getpid())
    save(ST / "owner-identity.json", me)
    coordinator = read(ST.parent / "coexist/ready.json")
    assert op.same(coordinator["owner"]), "EXPO/RLT priority coordinator is not live"
    state = {
        "time": time.time(),
        "owner": me,
        "gpu": p["gpu"],
        "stage": "READY",
        "runs": [],
    }
    requested = [False]

    def stop(*_):
        requested[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child = row = identity = None
    seen = {}
    released = True
    try:
        for key in ("smoke", "formal"):
            assert not requested[0]
            p = checked()
            row = p["runs"][key]
            assert not [g for g in gpu_rows() if g["gpu"] == p["gpu"]]
            assert op.same(read(ST.parent / "coexist/ready.json")["owner"])
            child = op.launch(key)
            identity = op.proc(child.pid)
            seen = {}
            released = False
            state.update(stage=key.upper(), driver=identity, started=time.time())
            state["runs"].append(
                {
                    "key": key,
                    "identity": identity,
                    "namespace": row["namespace"],
                    "run": row["run"],
                }
            )
            proof = False
            while child.poll() is None:
                current = scope(p, row, identity)
                seen.update({(x["pid"], x["start"]): x for x in current["tree"]})
                assert not current["outside"], current["outside"]
                assert (
                    not requested[0]
                    and time.time() - state["started"] < p[key + "_timeout"]
                )
                assert (
                    os.statvfs(row["run"]).f_bavail * os.statvfs(row["run"]).f_frsize
                    > 40 * 2**30
                )
                state.update(
                    time=time.time(),
                    scope={"gpu": current["gpu"], "actors": len(current["actors"])},
                )
                if key == "formal" and not proof:
                    data = scalars(row["run"])
                    rounds = {
                        k: v[-1]
                        for k, v in data.items()
                        if v
                        and k
                        in (
                            "train/rlt_q/rollout_query_count",
                            "train/rlt_q/rollout_signal_mean",
                            "train/rlt/global_min_replay_size",
                        )
                    }
                    if len(rounds) == 3 and all(
                        v["value"] > 0 and v["time"] >= state["started"]
                        for v in rounds.values()
                    ):
                        save(
                            ST / "formal-first-round.json",
                            {
                                "time": time.time(),
                                "driver": identity,
                                "metrics": rounds,
                                "scope": state["scope"],
                            },
                        )
                        proof = True
                save(ST / "status.json", state)
                time.sleep(15)
            rc = child.returncode
            op.finished(key, rc)
            current = cleanup(p, row, identity, child, seen)
            released = True
            save(
                ST / (key + "-released.json"),
                {"time": time.time(), "exit_code": rc, "scope": current},
            )
            assert rc == 0, (key, rc)
            if key == "smoke":
                save(ST / "smoke-passed.json", smoke_gate(row))
        state["stage"] = "COMPLETE"
    except BaseException:
        state.update(stage="FAILED", error=traceback.format_exc())
    finally:
        if not released and row is not None:
            try:
                cleanup(p, row, identity, child, seen)
                released = True
            except BaseException:
                state["cleanup_error"] = traceback.format_exc()
        state.update(time=time.time(), released=released)
        save(ST / "status.json", state)
        save(
            ST / "terminal.json",
            dict(
                state,
                operation_id="rlt-q-signals-g67-v1-" + p["lane"],
                boot_id=p["boot_id"],
                gpu_uuid=p["gpu_uuid"],
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("action", choices=("owner", "driver"))
    parser.add_argument("key", nargs="?")
    args = parser.parse_args()
    ST = args.stage
    op.ST = ST
    op.checked = checked
    if args.action == "owner":
        owner()
    else:
        op.driver(args.key)
