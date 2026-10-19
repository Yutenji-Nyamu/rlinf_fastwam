"""Bounded single-task native Dojo process lifecycle, one policy and one sim."""

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import read, save

DEVICE_ERRORS = (
    "CUDA error 700",
    "cudaErrorIllegalAddress",
    "ERROR_DEVICE_LOST",
    "VK_ERROR_DEVICE_LOST",
    "GPU page fault",
)


def check_policy_port_available(port):
    # Match the WebSocket server: allow a closed connection's TIME_WAIT,
    # while still rejecting a live listener on the same port.
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))


def device_fault(log):
    if not log.exists():
        return False
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 65536))
        tail = stream.read().decode(errors="replace")
    return any(marker in tail for marker in DEVICE_ERRORS)


def ops():
    path = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops/common.py")
    spec = importlib.util.spec_from_file_location("existing_ops", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config):
    c = read(config)
    out = Path(c["output"])
    out.mkdir(parents=True, exist_ok=False)
    save(out / "config.json", c)
    o = ops()
    g = c["gpu"]
    assert os.getuid() == c.get("uid", 20001) and not o.gpus()[g]["processes"]
    token = out.name
    start = time.time()
    save(out / "started.json", dict(time=start, identity=o.proc(os.getpid())))
    source = Path(__file__).resolve().parent
    children = []
    known = []
    code = 1
    error = None

    def environment(role):
        e = dict(os.environ, **read(c[role + "_environment"]))
        e.update(
            CUDA_VISIBLE_DEVICES=str(g),
            DOJO_GPU_SCOPE=str(g),
            DOJO_U_TOKEN=token,
            DOJO_U_REPO=c["repo"],
            DOJO_U_OUTPUT=str(out),
            DOJO_U_CONFIG=str(config),
            DOJO_U_NUM_ENVS=str(c["num_envs"]),
            ROBODOJO_RUN_ID=token,
            EVAL_NUM=str(c["episodes"]),
            ROBODOJO_FATAL_RESTART_COUNT="3",
            EVAL_ENV_TYPE="sim",
            XLA_PYTHON_CLIENT_MEM_FRACTION="0.3",
            OMP_NUM_THREADS="1",
            ROBODOJO_RENDER_GPU=str(g),
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        e["PYTHONPATH"] = ":".join(
            [
                c["bootstrap"],
                str(source),
                c["repo"],
                c["repo"] + "/XPolicyLab",
                c["repo"] + "/XPolicyLab/policy/Pi_05/openpi/src",
            ]
        )
        return e

    def launch(argv, role):
        log = (out / (role + ".log")).open("x")
        p = subprocess.Popen(
            argv,
            cwd=c["repo"],
            env=environment(role),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append(p)
        identity = o.proc(p.pid)
        known.append(identity)
        save(out / (role + "-identity.json"), identity)
        return p

    def stop(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    try:
        check_policy_port_available(c["policy"]["port"])
        policy = launch(
            [
                c["policy_python"],
                "-u",
                "-B",
                str(source / "policy_entry.py"),
                str(config),
            ],
            "policy",
        )
        deadline = time.monotonic() + 900
        while True:
            assert policy.poll() is None, "Policy server exited during initialization"
            assert time.monotonic() < deadline, "Policy initialization timeout"
            try:
                with socket.create_connection(
                    ("127.0.0.1", c["policy"]["port"]), timeout=1
                ):
                    break
            except OSError:
                time.sleep(3)
        main = [
            c["sim_python"],
            "-u",
            "-B",
            str(source / "client_entry.py"),
            "--task_name",
            c["task"],
            "--env_cfg_type",
            "arx_x5",
            "--num_envs",
            str(c["num_envs"]),
            "--device_id",
            str(g),
            "--policy_name",
            "Pi_05",
            "--port",
            str(c["policy"]["port"]),
            "--host",
            "127.0.0.1",
            "--protocol",
            "ws",
            "--policy_server_url",
            f"ws://127.0.0.1:{c['policy']['port']}",
            "--additional_info",
            "ckpt_name=sim,action_type=joint,u=10v5",
            "--seed",
            str(c["policy"]["seed"]),
            "--enable_cameras",
            "--headless",
            "--kit_args",
            f"--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera "
            f"--/renderer/multiGpu/enabled=false --/renderer/activeGpu={g}",
        ]
        if c.get("compat_script"):
            main = [
                c["sim_python"],
                c["compat_script"],
                "--isaac51",
                "--gpu-index",
                str(g),
                "--",
            ] + main
        sim = launch(main, "sim")
        save(out / "commands.json", dict(sim=main, policy=policy.args))
        duration = c.get("timeout_seconds")
        deadline = time.monotonic() + duration if duration else float("inf")
        while sim.poll() is None:
            assert policy.poll() is None, "Policy server exited during episode"
            time.sleep(10)
            if device_fault(out / "sim.log"):
                raise RuntimeError("Native GPU device fault; stop this lane")
            p = out / "progress.json"
            state = read(p) if p.exists() else {}
            age = time.time() - state.get("time", start)
            allowance = 1200 if state.get("phase") in ("policy_loading", None) else 600
            if age > allowance or time.monotonic() > deadline:
                raise TimeoutError(f"No progress ({age:.0f}s) or task deadline")
            # Observe C/G ownership, including graphics-only leaks.
            for card, info in o.gpus().items():
                for row in info["processes"]:
                    if int(card) != g:
                        p = Path("/proc") / str(row["pid"])
                        try:
                            ev = p.joinpath("environ").read_bytes().split(b"\0")
                        except OSError:
                            continue
                        if f"DOJO_U_TOKEN={token}".encode() in ev:
                            raise RuntimeError(f"Off-card GPU context: {card}")
        code = sim.returncode
        if code != 0:
            raise RuntimeError(f"Native simulator exit {code}")
    except BaseException:
        error = traceback.format_exc()
        code = 124 if "TimeoutError" in error else 1
    finally:
        # Same UID, unique run token and fresh PID/start identity, no name-wide kills.
        targets = {}
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                if p.stat().st_uid != os.getuid():
                    continue
                ev = p.joinpath("environ").read_bytes().split(b"\0")
                if f"DOJO_U_TOKEN={token}".encode() not in ev:
                    continue
                identity = o.proc(int(p.name))
                if identity:
                    targets[identity["pid"]] = identity
            except (OSError, ValueError):
                continue
        targets.update({x["pid"]: x for x in known if x})
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for identity in targets.values():
                if o.same(identity):
                    o.exact_signal(identity, sig)
            for p in children:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            time.sleep(2)
        clear = not o.gpus()[g]["processes"]
        save(out / "clearance.json", dict(time=time.time(), gpu=g, clear=clear))
        if not clear:
            raise RuntimeError("Card did not clear; stop lane before next task/RLT")
    # A simulator error may still have yielded complete, useful episodes.
    valid = None
    if list(out.glob("episodes/*/episode.json")):
        e = environment("policy")
        e["CUDA_VISIBLE_DEVICES"] = ""
        e.pop("DOJO_GPU_SCOPE", None)
        cmd = [c["policy_python"], "-B", str(source / "validate.py"), str(out)]
        if c.get("full_decode"):
            cmd.append("--full-decode")
        with (out / "validation.log").open("x") as log:
            r = subprocess.run(
                cmd, env=e, stdout=log, stderr=subprocess.STDOUT, timeout=900
            )
        if r.returncode != 0:
            save(
                out / "result.json",
                dict(state="INVALID_RECORDING", error="See validation.log"),
            )
            return 78
        valid = read(out / "validation.json")
    completed = valid["episodes"] if valid else 0
    state = (
        "COMPLETE"
        if code == 0 and completed == c["episodes"]
        else "PARTIAL"
        if completed
        else "SKIPPED"
    )
    result = dict(
        time=time.time(),
        elapsed_seconds=time.time() - start,
        state=state,
        error=error,
        returncode=code,
        task=c["task"],
        requested_episodes=c["episodes"],
        episodes=completed,
        successes=valid["successes"] if valid else 0,
    )
    save(out / "result.json", result)
    # Native device loss is different from an ordinary task failure/timeout.
    if device_fault(out / "sim.log") or (error and "GPU device fault" in error):
        save(
            out / "device-fault.json",
            dict(stop_lane=True, reason="Native GPU device fault"),
        )
        return 79
    return 0


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1]).resolve()))
