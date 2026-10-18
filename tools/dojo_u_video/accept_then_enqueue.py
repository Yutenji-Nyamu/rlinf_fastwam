"""One finite, two-hour smoke acceptance -> existing queue insertion transaction."""

import argparse
import getpass
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("C:/Users/86136/Documents/rl")
E = Path(
    "E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a/dojo-u1018"
)
sys.path.insert(0, str(ROOT / "local_scripts"))
import remote_exec_autodl as ssh  # noqa: E402

D = "/data/chenyiteng/deployment-20261018/dojo-u-video-v1"
SMOKE = (
    "/data/chenyiteng/results/dojo-u-video/20261018-smoke-sz1/smoke-g3-n25-stack-bowls"
)
W = "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dojo-u-video-20261018"
HOSTS = {
    "sz1": (
        "120.241.223.9",
        "admin",
        1003,
        "qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY",
    ),
    "sz3": (
        "221.182.146.45",
        "h100-gpu01",
        20001,
        "Pblu5EtS9HboLwJ3qVe93l2IcZ5TLAIiP9+ZC6JeN8U",
    ),
}
clients = {}
password = getpass.getpass("SSH chenyiteng password: ")


def save(name, value):
    p = E / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def client(host):
    if host in clients and clients[host].get_transport().is_active():
        return clients[host]
    address, hostname, uid, pin = HOSTS[host]
    os.environ["SEETA_SSH_PASSWORD"] = password
    try:
        c = ssh.connect(
            argparse.Namespace(
                host=address,
                port=22,
                user="chenyiteng",
                timeout=25,
                host_key_sha256=pin,
            )
        )
    finally:
        os.environ.pop("SEETA_SSH_PASSWORD", None)
    _, out, err = c.exec_command("id -un && id -u && hostname", timeout=25)
    assert out.read().decode().splitlines() == ["chenyiteng", str(uid), hostname]
    assert out.channel.recv_exit_status() == 0
    c.get_transport().set_keepalive(15)
    clients[host] = c
    return c


def read(host, path, optional=False):
    with client(host).open_sftp() as s:
        s.get_channel().settimeout(25)
        try:
            with s.open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            if optional:
                return None
            raise


assert not (E / "accept-queue-started.json").exists(), (
    "Do not launch this transaction twice"
)
started = time.time()
try:
    for host in HOSTS:
        client(host)
    assert not read("sz3", D + "/enqueue-started.json", True)
    save(
        "accept-queue-started.json",
        dict(
            time=started,
            pid=os.getpid(),
            deadline=started + 7200,
            smoke=SMOKE,
            source="tools/dojo_u_video/enqueue.py",
        ),
    )
    print(
        "READY: pinned SZ1/SZ3 identities verified; finite acceptance/queue transaction waiting.",
        flush=True,
    )
    while time.time() < started + 7200:
        result = read("sz1", SMOKE + "/result.json", True)
        if result:
            break
        save(
            "accept-queue-status.json",
            dict(time=time.time(), phase="WAIT_SMOKE", deadline=started + 7200),
        )
        time.sleep(30)
    else:
        raise TimeoutError("Smoke deadline reached; existing queues kept unchanged")
    validation = read("sz1", SMOKE + "/validation.json")
    parity = read("sz1", SMOKE + "/main-parity.json")
    clearance = read("sz1", SMOKE + "/clearance.json")
    assert result["state"] == "COMPLETE" and result["returncode"] == 0, result
    assert (
        validation["passed"]
        and validation["episodes"] == 25
        and validation["successes"] > 0
    )
    assert validation.get("full_decode_first_episode") is True
    assert parity["main_exact"] and clearance["clear"] and clearance["gpu"] == 3
    receipt = dict(
        passed=True,
        time=time.time(),
        smoke_host="sz1",
        smoke_output=SMOKE,
        episodes=25,
        successes=validation["successes"],
        parallel=25,
        main_action_exact=True,
        main_rng_exact=True,
        actual_video_frames_checked=True,
        three_cameras_first_episode_decoded=True,
        initial_three_cameras_visually_reviewed=True,
        smoke_gpu_released=True,
        sz3_runtime_acceptance="first real task after predecessor; device errors stop lane",
    )
    save("smoke-accepted.json", receipt)
    save("smoke-validation.json", validation)
    save("smoke-result.json", result)
    with client("sz3").open_sftp() as s:
        s.get_channel().settimeout(25)
        # Atomic receipt, new file only. Do not overwrite any other transaction.
        try:
            s.stat(D + "/smoke-accepted.json")
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError(
                "Acceptance receipt already exists; inspect before reuse"
            )
        with s.open(D + "/smoke-accepted.json.pending", "wx") as f:
            f.write(json.dumps(receipt))
        s.rename(D + "/smoke-accepted.json.pending", D + "/smoke-accepted.json")
    save("accept-queue-status.json", dict(time=time.time(), phase="ENQUEUE_ONCE"))
    _, out, err = client("sz3").exec_command(
        "CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B "
        + W
        + "/tools/dojo_u_video/enqueue.py",
        timeout=100,
    )
    stdout, stderr = out.read().decode(), err.read().decode()
    code = out.channel.recv_exit_status()
    (E / "accept-enqueue.stdout").write_text(stdout, encoding="utf-8")
    (E / "accept-enqueue.stderr").write_text(stderr, encoding="utf-8")
    assert code == 0, stderr[-3000:]
    enqueued = read("sz3", D + "/enqueued.json")
    assert enqueued["current_inference_retained"] and enqueued["fallbacks_preserved"]
    save("enqueued.json", enqueued)
    save(
        "accept-queue-status.json",
        dict(time=time.time(), phase="COMPLETE", receipt=enqueued),
    )
    print(
        "COMPLETE: 25-episode smoke passed; 54x125 queue inserted behind current SZ3 6/7 sweep, before original RLT.",
        flush=True,
    )
except BaseException:
    save(
        "accept-queue-status.json",
        dict(
            time=time.time(),
            phase="NEEDS_ATTENTION",
            error=traceback.format_exc(),
            note="Do not replay an uncertain enqueue; inspect server transaction receipts.",
        ),
    )
    print(traceback.format_exc(), flush=True)
    raise
finally:
    for c in clients.values():
        c.close()
