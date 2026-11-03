"""Prepared pair handoff -> bounded Dojo smoke -> formal Clean -> existing RLT.

Reuse the accepted identity-based RLT handoff and namespace-cleaning GRPO driver.
The native Isaac processes are separate, persistent, and explicitly GPU-bound.
"""

import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops")
from common import read, save, proc, same, exact_signal, gpus


def main(directory):
    d = Path(directory)
    plan = read(d / "plan.json")
    root = Path(plan["training_source"])
    sys.path.insert(0, str(root / "tools/rlt_disk_fallback"))
    from priority_pair_handoff import handoff

    me = proc(os.getpid())
    save(d / "identity.json", me)
    children = []
    native = []
    registered = False
    training_env = dict(os.environ, **read(d / "training-environment.json"))

    def status(phase, **kw):
        save(d / "status.json", dict(phase=phase, time=time.time(), **kw))

    def launch(name, command, env, cwd=None):
        environment = dict(
            env,
            DOJO_GRPO_OWNER=str(d),
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        with (d / f"{name}.log").open("x") as stream:
            p = subprocess.Popen(
                command,
                cwd=cwd or root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        who = proc(p.pid)
        children.append((p, who))
        save(d / f"{name}-identity.json", who)
        return p

    def wait(p, seconds):
        try:
            code = p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            who = next(w for c, w in children if c is p)
            exact_signal(who, signal.SIGTERM)
            p.wait(timeout=180)
            raise TimeoutError(f"Child {p.pid} exceeded {seconds}s")
        if code:
            raise RuntimeError(f"Child {p.pid} exited {code}; see its log")

    def stop(*_):
        raise SystemExit(143)

    def take_cards(cards):
        nonlocal registered
        directory = d / (d.name + "-handoff-" + "".join(map(str, cards)))
        directory.mkdir()
        reservations = {
            str(g): dict(identity=me, release=str(d / "released.json")) for g in cards
        }
        handoff(directory, cards, me, reservations=reservations)
        registered = True

    def model_parity():
        pt = launch(
            "torch-probe",
            [
                plan["python"],
                "-u",
                "-B",
                str(root / "tools/dojo_grpo/model_probe.py"),
                "--mode",
                "torch",
                "--deployment",
                str(d),
            ],
            dict(training_env, CUDA_VISIBLE_DEVICES="6"),
        )
        je = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="7",
            XLA_PYTHON_CLIENT_MEM_FRACTION="0.25",
            OMP_NUM_THREADS="8",
            OPENPI_DATA_HOME="/home/chenyiteng/cache/dojo-sz1-20261001/openpi",
            PYTHONPATH=plan["native_repo"]
            + "/XPolicyLab/policy/Pi_05/openpi/src:"
            + plan["native_repo"],
        )
        jp = launch(
            "jax-probe",
            [
                plan["jax_python"],
                "-u",
                "-B",
                str(root / "tools/dojo_grpo/model_probe.py"),
                "--mode",
                "jax",
                "--deployment",
                str(d),
            ],
            je,
        )
        wait(pt, 900)
        wait(jp, 900)
        import numpy as np

        a = np.load(d / "torch-probe.npz")
        b = np.load(d / "jax-probe.npz")
        delta = np.abs(a["actions"] - b["actions"])
        parity = {
            "rmse": float(np.sqrt((delta**2).mean())),
            "max_abs": float(delta.max()),
            "prompt_equal": bool(np.array_equal(a["prompt"], b["prompt"])),
            "state_max_abs": float(np.abs(a["state"] - b["state"]).max()),
        }
        save(d / "parity.json", parity)
        assert parity["prompt_equal"] and parity["state_max_abs"] < 1e-5, parity
        assert parity["rmse"] < 0.02 and parity["max_abs"] < 0.15, parity
        return parity

    signal.signal(signal.SIGTERM, stop)
    try:
        assert os.getuid() == 1003 and plan["gpus"] == [6, 7]
        assert (Path(plan["model"]) / "conversion-receipt.json").is_file()
        status("HANDOFF_FROM_RLT")
        # Keep useful RLT work on GPU7 during GPU6 simulator construction/reset.
        staged = bool(plan.get("parity_source")) and all(
            int(plan.get("render_gpus", {}).get(str(g), g)) == 6 for g in [6, 7]
        )
        take_cards([6] if staged else [6, 7])
        parity = None
        if plan.get("parity_source"):
            previous = Path(plan["parity_source"])
            assert read(previous / "plan.json")["model"] == plan["model"]
            parity = read(previous / "parity.json")
            assert parity["prompt_equal"] and parity["state_max_abs"] < 1e-5
            assert parity["rmse"] < 0.02 and parity["max_abs"] < 0.15
            save(d / "parity.json", dict(parity, reused_verified_source=str(previous)))
        elif plan.get("probe_observation"):
            status("MODEL_PARITY_SAVED_NATIVE_OBSERVATION")
            parity = model_parity()
        native = []
        for g in [6, 7]:
            render_gpu = int(plan.get("render_gpus", {}).get(str(g), g))
            cache = d / f"native-g{g}" / "cache"
            cache.mkdir(parents=True, exist_ok=True)
            e = dict(os.environ, **read(d / "sim-environment.json"))
            e.update(
                CUDA_VISIBLE_DEVICES=str(render_gpu),
                DOJO_GRPO_NATIVE_CONFIG=str(d / f"native-g{g}.json"),
                ROBODOJO_RENDER_GPU=str(render_gpu),
                ROBODOJO_RUN_ID=f"dojo-grpo-1103-g{g}",
                XDG_CACHE_HOME=str(cache / "xdg"),
                CUDA_CACHE_PATH=str(cache / "cuda"),
                __GL_SHADER_DISK_CACHE_PATH=str(cache / "gl"),
                EVAL_ENV_TYPE="sim",
                OMP_NUM_THREADS="1",
                EVAL_NUM="55",
                PYTHONPATH=":".join(
                    [
                        str(root),
                        plan["native_repo"],
                        plan["native_repo"] + "/XPolicyLab",
                    ]
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
                str(plan["native_envs_per_card"]),
                "--env_cfg_type",
                "arx_x5",
                "--device_id",
                str(render_gpu),
                "--policy_name",
                "Pi_05",
                "--port",
                str(63460 + g),
                "--host",
                "127.0.0.1",
                "--additional_info",
                "grpo_clean",
                "--seed",
                "1",
                "--enable_cameras",
                "--headless",
                "--kit_args",
                f"--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera --/renderer/multiGpu/enabled=false --/renderer/activeGpu={render_gpu} --portable --portable-root {cache}/kit --/app/userConfigPath={cache}/user.config.json",
            ]
            for folder in plan.get("native_extension_folders", []):
                command[-1] += " --ext-folder " + folder
            if plan.get("vulkan_compat"):
                command = [
                    plan["native_python"],
                    plan["vulkan_compat"],
                    "--isaac51",
                    "--gpu-index",
                    str(render_gpu),
                    "--",
                ] + command
            native.append(launch(f"native-g{g}", command, e, plan["native_repo"]))
        status("NATIVE_STARTING", envs_per_card=plan["native_envs_per_card"])
        deadline = time.monotonic() + 900
        while not all((d / f"rpc/gpu{g}.sock").exists() for g in [6, 7]):
            assert all(p.poll() is None for p in native), (
                "Native simulator exited during startup"
            )
            for g in [6, 7]:
                with (d / f"native-g{g}.log").open("rb") as stream:
                    stream.seek(max(0, stream.seek(0, 2) - 32768))
                    tail = stream.read()
                if (
                    b"[Fatal]" in tail
                    or b"GPU crash detected" in tail
                    or b"ERROR_DEVICE_LOST" in tail
                ):
                    raise RuntimeError(
                        f"Native GPU{g} renderer crashed during startup; see log"
                    )
            if time.monotonic() > deadline:
                raise TimeoutError("Native server startup")
            time.sleep(1)
        # Verify computing AND graphics contexts before continuing.
        owned = {p.pid for p in native}
        unexpected = [
            {"gpu": g, **r}
            for g, card in gpus().items()
            for r in card["processes"]
            if r["pid"] in owned and g not in [6, 7]
        ]
        save(
            d / "native-binding.json", {"observed": gpus(), "outside_pair": unexpected}
        )
        if unexpected:
            if any(r["type"] != "G" for r in unexpected):
                raise RuntimeError(f"Native compute binding mismatch: {unexpected}")
            status("CHECK_GRAPHICS_BINDING", outside_pair=unexpected)
            deadline = time.monotonic() + 180
            while not (d / "graphics-binding-reviewed.json").is_file():
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Unreviewed graphics context: {unexpected}")
                time.sleep(1)
        status("INITIAL_RESET_AND_MODEL_PARITY")
        wait(
            launch(
                "initial-reset",
                [
                    plan["python"],
                    "-u",
                    "-B",
                    str(root / "tools/dojo_grpo/prepare_observation.py"),
                    str(d),
                ],
                dict(training_env, CUDA_VISIBLE_DEVICES=""),
            ),
            1250,
        )
        if parity is None:
            parity = model_parity()
        # Do not run a formal job if the model/environment contract failed.
        if staged:
            status("HANDOFF_GPU7_BEFORE_TRAINING")
            take_cards([7])
        status("CLEAN_SMOKE_FULL_WAVE", parity=parity)
        wait(
            launch(
                "smoke-driver",
                [
                    plan["python"],
                    "-u",
                    "-B",
                    str(root / "tools/dojo_grpo/training_entry.py"),
                    "--config-path",
                    str(d / "config"),
                    "--config-name",
                    "smoke",
                ],
                training_env,
            ),
            7200,
        )
        status("SMOKE_COMPLETE_READY_FORMAL", parity=parity)
        # The formal launch flag is written by the same authorized execution
        # after checking the complete save + first update receipt; no user gate.
        deadline = time.monotonic() + 300
        while not (d / "launch-formal.json").is_file():
            if time.monotonic() > deadline:
                raise TimeoutError("Formal acceptance not supplied; restore RLT")
            time.sleep(0.5)
        status("FORMAL_TRAINING", rounds=200)
        wait(
            launch(
                "formal-driver",
                [
                    plan["python"],
                    "-u",
                    "-B",
                    str(root / "tools/dojo_grpo/training_entry.py"),
                    "--config-path",
                    str(d / "config"),
                    "--config-name",
                    "formal",
                ],
                training_env,
            ),
            1209600,
        )
        status("COMPLETE")
    except BaseException:
        status("FAILED_RETURN_RLT", error=traceback.format_exc())
        raise
    finally:
        # A second stop during Isaac shutdown must not interrupt card release.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for p, who in reversed(children):
            if same(who):
                exact_signal(who, signal.SIGTERM)
        for p, who in reversed(children):
            try:
                p.wait(timeout=5 if p in native else 30)
            except subprocess.TimeoutExpired:
                if same(who):
                    exact_signal(who, signal.SIGKILL)
                p.wait(timeout=30)
        # Kill only residual descendants tagged by this exact private owner.
        for path in Path("/proc").iterdir():
            if not path.name.isdigit() or int(path.name) == os.getpid():
                continue
            try:
                if path.stat().st_uid != os.getuid():
                    continue
                if ("DOJO_GRPO_OWNER=" + str(d)).encode() not in (
                    path / "environ"
                ).read_bytes().split(b"\0"):
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
