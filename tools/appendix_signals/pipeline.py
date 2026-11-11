"""Sequential, bounded GPU5 jobs with exact process cleanup and durable receipts."""

import argparse
import json
import os
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import write_json


def identity(pid):
    p = Path("/proc") / str(pid)
    try:
        fields = (p / "stat").read_text().split(") ", 1)[1].split()
        return dict(
            pid=int(pid), uid=p.stat().st_uid, start=fields[19], state=fields[0]
        )
    except (FileNotFoundError, ProcessLookupError):
        return None


def exact_alive(who):
    current = identity(who["pid"])
    return (
        current is not None
        and current["state"] != "Z"
        and all(current[k] == who[k] for k in ["pid", "uid", "start"])
    )


def marked(stage):
    tag = ("APPENDIX_STAGE_ID=" + stage).encode()
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            if p.stat().st_uid == os.getuid() and tag in (
                p / "environ"
            ).read_bytes().split(b"\0"):
                who = identity(p.name)
                if who and who["state"] != "Z":
                    found.append(who)
        except (OSError, ProcessLookupError):
            continue
    return found


def cleanup(stage):
    targets = marked(stage)
    for who in targets:
        if exact_alive(who):
            os.kill(who["pid"], signal.SIGTERM)
    end = time.monotonic() + 10
    while any(exact_alive(w) for w in targets) and time.monotonic() < end:
        time.sleep(0.2)
    for who in targets:
        if exact_alive(who):
            os.kill(who["pid"], signal.SIGKILL)
    return targets


def binding(stage):
    owned = {x["pid"] for x in marked(stage)}
    xml = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], timeout=15))
    violations = []
    gpu5 = None
    for index, gpu in enumerate(xml.findall("gpu")):
        if index == 5:
            gpu5 = dict(
                memory_used=gpu.findtext("fb_memory_usage/used"),
                utilization=gpu.findtext("utilization/gpu_util"),
            )
        for proc in gpu.findall("processes/process_info"):
            if int(proc.findtext("pid")) not in owned or index == 5:
                continue
            mib = int(proc.findtext("used_memory").split()[0])
            kind = proc.findtext("type")
            # Previously reviewed Isaac G-only residue is permitted on GPU4, never 0-3.
            if index == 4 and kind == "G" and mib <= 8:
                continue
            violations.append(
                dict(gpu=index, pid=int(proc.findtext("pid")), mib=mib, type=kind)
            )
    return gpu5, violations


def main(args):
    plan = json.loads(Path(args.plan).read_text())
    out = Path(plan["output"])
    out.mkdir(parents=True, exist_ok=True)
    assert plan["gpu"] == 5
    assert Path(plan["ready_receipt"]).is_file(), "CPU preparation did not finish"
    results = []
    for stage in plan["stages"]:
        if time.time() >= plan["absolute_deadline"]:
            results.append(dict(name=stage["name"], state="skipped_global_budget"))
            break
        identifier = plan["id"] + "/" + stage["name"]
        if not all(Path(p).exists() for p in stage.get("requires", [])):
            results.append(dict(name=stage["name"], state="skipped_missing_dependency"))
            continue
        env = {
            **os.environ,
            **stage["environment"],
            "CUDA_VISIBLE_DEVICES": "5",
            "DOJO_GPU_SCOPE": "5",
            "APPENDIX_STAGE_ID": identifier,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        logpath = out / (stage["name"] + ".log")
        started = time.time()
        with logpath.open("xb") as log:
            p = subprocess.Popen(
                stage["command"],
                cwd=stage["cwd"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        deadline = min(started + stage["timeout"], plan["absolute_deadline"])
        peaks = []
        reason = None
        write_json(
            out / "status.json",
            dict(
                stage=stage["name"],
                state="running",
                identity=identity(p.pid),
                started=started,
            ),
        )
        try:
            while p.poll() is None:
                gpu, violations = binding(identifier)
                peaks.append(dict(time=time.time(), **(gpu or {})))
                if violations:
                    reason = dict(kind="binding_violation", details=violations)
                    break
                if time.time() > deadline:
                    reason = dict(kind="time_budget")
                    break
                write_json(
                    out / "gpu5-latest.json",
                    dict(stage=stage["name"], sample=peaks[-1]),
                )
                time.sleep(3)
        finally:
            stopped = cleanup(identifier)
            try:
                code = p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                code = 124
        row = dict(
            name=stage["name"],
            started=started,
            ended=time.time(),
            seconds=time.time() - started,
            returncode=code,
            state="complete" if code == 0 and reason is None else "failed_or_bounded",
            reason=reason,
            cleaned=stopped,
            command=stage["command"],
        )
        write_json(out / (stage["name"] + "-gpu.json"), peaks)
        write_json(out / (stage["name"] + "-receipt.json"), row)
        results.append(row)
        write_json(out / "results.json", results)
    write_json(
        out / "complete.json",
        dict(
            state="finished_all_available_stages",
            results=results,
            time=time.time(),
            next="same owner starts prepared DSRL checkpoint resume immediately",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    main(p.parse_args())
