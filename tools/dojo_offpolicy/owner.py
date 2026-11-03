# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Single-card Dojo phases, accepted RLT handoff and native supervisor.

Lifecycle and binding checks are taken from tools/dojo_grpo/pair_owner.py.
Any failed phase releases the exact card to the already prepared RLT fallback.
"""

import os
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, "/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops")
from common import exact_signal, gpus, proc, read, same, save


def cleanup_owned(directory, grace_seconds=15):
    """Clean only this deployment's tagged processes after its owner is gone."""
    marker = ("DOJO_GRPO_OWNER=" + str(directory)).encode()
    targets = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            if path.stat().st_uid != os.getuid() or marker not in (
                path / "environ"
            ).read_bytes().split(b"\0"):
                continue
            who = proc(int(path.name))
            if who and same(who):
                targets.append(who)
                exact_signal(who, signal.SIGTERM)
        except OSError:
            continue
    deadline = time.monotonic() + grace_seconds
    while any(same(who) for who in targets) and time.monotonic() < deadline:
        time.sleep(0.25)
    for who in targets:
        if same(who):
            exact_signal(who, signal.SIGKILL)
    return targets


def main(directory):
    d = Path(directory)
    plan = read(d / "plan.json")
    root, gpu = Path(plan["training_source"]), int(plan["gpu"])
    assert os.getuid() == 1003 and gpu in (6, 7)
    sys.path.insert(0, str(root / "tools/rlt_disk_fallback"))
    from priority_pair_handoff import handoff

    me = proc(os.getpid())
    save(d / "identity.json", me)
    children, registered = [], False

    def status(phase, **kw):
        save(d / "status.json", dict(phase=phase, time=time.time(), gpu=gpu, **kw))

    def launch(name, command, env, cwd=None):
        with (d / f"{name}.log").open("x") as log:
            p = subprocess.Popen(
                command,
                cwd=cwd or root,
                env=dict(env, DOJO_GRPO_OWNER=str(d)),
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        who = proc(p.pid)
        children.append((p, who))
        save(d / f"{name}-identity.json", who)
        return p

    def stop(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    try:
        status("HANDOFF_FROM_RLT")
        h = d / (d.name + "-handoff")
        h.mkdir()
        handoff(
            h,
            [gpu],
            me,
            reservations={
                str(gpu): {"identity": me, "release": str(d / "released.json")}
            },
        )
        registered = True
        cache = d / f"native-g{gpu}/cache"
        cache.mkdir(parents=True, exist_ok=True)
        e = dict(os.environ, **read(d / "sim-environment.json"))
        e.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            DOJO_GRPO_NATIVE_CONFIG=str(d / f"native-g{gpu}.json"),
            ROBODOJO_RENDER_GPU=str(gpu),
            ROBODOJO_RUN_ID=d.name,
            XDG_CACHE_HOME=str(cache / "xdg"),
            CUDA_CACHE_PATH=str(cache / "cuda"),
            __GL_SHADER_DISK_CACHE_PATH=str(cache / "gl"),
            EVAL_ENV_TYPE="sim",
            OMP_NUM_THREADS="1",
            EVAL_NUM="55",
            PYTHONPATH=":".join(
                [str(root), plan["native_repo"], plan["native_repo"] + "/XPolicyLab"]
            ),
        )
        command = [
            plan["native_python"],
            "-u",
            "-B",
            str(root / "tools/dojo_grpo/native_entry.py"),
            "--task_name",
            "put_bottles_into_dustbin",
            "--num_envs",
            str(plan["num_envs"]),
            "--env_cfg_type",
            "arx_x5",
            "--device_id",
            str(gpu),
            "--policy_name",
            "Pi_05",
            "--port",
            str(63460 + gpu),
            "--host",
            "127.0.0.1",
            "--additional_info",
            "offpolicy",
            "--seed",
            "1",
            "--enable_cameras",
            "--headless",
            "--kit_args",
            f"--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera --/renderer/multiGpu/enabled=false --/renderer/activeGpu={gpu} --portable --portable-root {cache}/kit --/app/userConfigPath={cache}/user.config.json",
        ]
        for folder in plan.get("native_extension_folders", []):
            command[-1] += " --ext-folder " + folder
        if plan.get("vulkan_compat"):
            command = [
                plan["native_python"],
                plan["vulkan_compat"],
                "--isaac51",
                "--gpu-index",
                str(gpu),
                "--",
            ] + command
        command = [
            plan["python"],
            "-u",
            "-B",
            str(root / "tools/dojo_grpo/native_supervisor.py"),
            str(d),
            str(gpu),
        ] + command
        native = launch(f"native-g{gpu}", command, e, plan["native_repo"])
        status("NATIVE_STARTING", envs=plan["num_envs"])
        deadline = time.monotonic() + 600
        while not (d / f"rpc/gpu{gpu}.sock").exists():
            assert native.poll() is None, "Native exited during startup"
            with (d / f"native-g{gpu}.log").open("rb") as stream:
                stream.seek(max(0, stream.seek(0, 2) - 32768))
                tail = stream.read()
            if any(
                x in tail
                for x in (
                    b"[Fatal]",
                    b"GPU crash detected",
                    b"ERROR_DEVICE_LOST",
                    b"gpu_crash_dump_",
                    b"Crash report files for upload",
                )
            ):
                raise RuntimeError("Isaac renderer fault; bounded attempt ends")
            assert time.monotonic() < deadline, "Native startup timeout"
            time.sleep(1)
        pid = read(d / f"native-g{gpu}-child.json")["pid"]
        tree = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"]))
        residuals = []
        for index, card in enumerate(tree.findall("gpu")):
            for process in card.findall("processes/process_info"):
                if int(process.findtext("pid")) != pid or index == gpu:
                    continue
                amount = int(process.findtext("used_memory").split()[0])
                assert process.findtext("type") == "G" and amount <= plan.get(
                    "reviewed_graphics_residual_mib", {}
                ).get(str(index), -1), "Native binding mismatch"
                residuals.append({"gpu": index, "mib": amount, "type": "G"})
        save(
            d / "binding.json",
            {"native": pid, "snapshot": gpus(), "residuals": residuals},
        )
        env = dict(os.environ, **read(d / "training-environment.json"))
        for phase in plan["phases"]:
            status(phase["name"], started=time.time())
            p = launch(phase["name"], phase["command"], env)
            try:
                code = p.wait(timeout=phase.get("timeout"))
            except subprocess.TimeoutExpired:
                raise TimeoutError("Phase deadline: " + phase["name"])
            assert code == 0, f"{phase['name']} exited {code}"
        status("COMPLETE")
    except BaseException:
        status("FAILED_RETURN_RLT", error=traceback.format_exc())
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for p, who in reversed(children):
            if same(who):
                exact_signal(who, signal.SIGTERM)
        for p, who in reversed(children):
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if same(who):
                    exact_signal(who, signal.SIGKILL)
                p.wait(timeout=30)
        for path in Path("/proc").iterdir():
            if not path.name.isdigit() or int(path.name) == os.getpid():
                continue
            try:
                if path.stat().st_uid != os.getuid() or (
                    "DOJO_GRPO_OWNER=" + str(d)
                ).encode() not in (path / "environ").read_bytes().split(b"\0"):
                    continue
                who = proc(int(path.name))
                if who and same(who):
                    exact_signal(who, signal.SIGTERM)
            except OSError:
                continue
        if registered:
            save(
                d / "released.json",
                {"time": time.time(), "result": read(d / "status.json")},
            )


if __name__ == "__main__":
    main(sys.argv[1])
