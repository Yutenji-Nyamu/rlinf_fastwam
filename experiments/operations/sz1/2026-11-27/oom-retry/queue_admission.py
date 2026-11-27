"""Wait for the exact requested slots; a busy predecessor keeps its successor queued."""
import json
import os
import re
import shlex
import subprocess
import time
import xml.etree.ElementTree as ET
from common import admission_card, gpus, read, same, save
from pathlib import Path
from urllib.parse import urlsplit


def retry_gpu_processes(cards):
    """One bounded driver probe. Even an unkillable driver query cannot hold admission."""
    probe = subprocess.Popen(["nvidia-smi", "-q", "-x", "-i", ",".join(map(str, sorted(cards)))],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        data, _ = probe.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        probe.kill()
        # A process stuck in the kernel may ignore SIGKILL until the driver recovers.
        # Do not follow it into an unbounded wait or launch another query on this attempt.
        raise
    finally:
        probe.stdout.close()
    if probe.returncode:
        raise subprocess.CalledProcessError(probe.returncode, probe.args)
    root = ET.fromstring(data)
    rows = {}
    for gpu in root.findall("gpu"):
        card = int(gpu.findtext("minor_number"))
        rows[card] = [int(p.findtext("pid")) for p in gpu.findall("processes/process_info")]
    if set(rows) != set(cards):
        raise ValueError("Incomplete GPU snapshot for recovery admission")
    return rows


def ray_jobs(url):
    # This is the existing local Ray dashboard. One HTTP request at admission only.
    local = urlsplit(url)
    if local.scheme != "http" or local.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return []
    raw = subprocess.check_output(["curl", "--noproxy", "*", "-fsS", "--max-time", "3",
                                   "--", url.rstrip("/") + "/api/jobs/"], timeout=4)
    return json.loads(raw)


def bind_ray_driver(directory, identity_path=None):
    """Resolve one exact FastWAM Ray driver; repeat calls may reuse its boot-scoped binding."""
    if not (directory / "fastwam_driver.py").is_file():
        return None
    if identity_path is None:
        current = directory / "driver-identity.json"
        archived = directory.glob("prior-oom-retry-*/driver-identity.json")
        identity_path = current if current.is_file() else max(archived, key=lambda f:f.stat().st_mtime, default=None)
    if identity_path is None or not identity_path.is_file():
        return None
    identity = read(identity_path)
    if identity.get("uid") != os.getuid():
        return None
    driver = {k:identity[k] for k in ("pid","uid","start")}
    address = read(directory / "training-environment.json").get("RAY_ADDRESS")
    if not address:
        return None
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    boot_sec = next(int(row.split()[1]) for row in Path("/proc/stat").read_text().splitlines()
                    if row.startswith("btime "))
    if identity_path.stat().st_mtime < boot_sec:
        return None
    begin = boot_sec + driver["start"] / os.sysconf("SC_CLK_TCK")
    live = same(identity)
    released = identity_path.parent / "released.json"
    end = float(read(released).get("time", 0)) if released.is_file() else None
    if not live and not end:
        return None
    cache = directory / "oom-ray-driver.json"
    expected = {"driver":driver, "boot":boot, "address":address}
    if cache.is_file():
        data = read(cache)
        if all(data.get(k)==v for k,v in expected.items()):
            return dict(data, until=end)
    plan = read(directory / "plan.json")
    try:
        url = plan.get("ray_dashboard_url") or "http://127.0.0.1:8266"
        dashboard = urlsplit(url)
        if dashboard.hostname == address.rsplit(":",1)[0] and dashboard.port and dashboard.scheme=="http":
            url = f"http://127.0.0.1:{dashboard.port}"
        jobs = ray_jobs(url)
        matches = []
        for job in jobs:
            info = job.get("driver_info") or {}
            jid = str(job.get("job_id") or info.get("id") or "").lower()
            if (str(info.get("pid")) != str(driver["pid"]) or
                    info.get("node_ip_address") != address.rsplit(":",1)[0] or
                    not re.fullmatch(r"[0-9a-f]{8}", jid) or
                    str(directory / "fastwam_driver.py") not in shlex.split(job.get("entrypoint",""))):
                continue
            start = float(job.get("start_time") or 0) / 1000
            if start and start < begin - 2:
                continue
            if not live and (not start or start > end + 1):
                continue
            matches.append(jid)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return None
    if len(matches) != 1:
        return None
    data = dict(expected, job_id=matches[0], boot_sec=boot_sec)
    save(cache, data)
    return dict(data, until=end)


def retry_owner(pid, directory, receipts, proc_root=Path("/proc"), ray_job=None):
    process = proc_root / str(pid)
    try:
        if process.stat().st_uid != os.getuid():
            return False
        if pid in receipts and same(receipts[pid]):
            return True
        # Launcher markers identify this deployment, including its child simulators.
        # The user account alone can also own unrelated jobs on the same device.
        env = set((process / "environ").read_bytes().split(b"\0"))
        if (ray_job and ("RAY_JOB_ID=" + ray_job["job_id"]).encode() in env
                and ("RAY_ADDRESS=" + ray_job["address"]).encode() in env):
            # Job numbers may repeat after a Ray restart. Match the worker lifetime
            # against the exact driver's ownership interval, including a cached binding.
            fields = (process / "stat").read_text().rsplit(")",1)[1].split()
            ticks = int(fields[19])
            since = ray_job["boot_sec"] + ticks / os.sysconf("SC_CLK_TCK")
            if ticks >= ray_job["driver"]["start"] and (not ray_job.get("until") or since <= ray_job["until"] + 1):
                return True
        exact = os.fsencode(directory)
        if any(key + exact in env for key in (
                b"DOJO_GRPO_OWNER=", b"DOJO_GRPO_DEPLOYMENT=", b"ROBOTWIN_DSRL_OWNER=",
                b"OOM_RETRY_OWNER=")):
            return True
        argv = set((process / "cmdline").read_bytes().split(b"\0"))
        return bool(argv & {exact, os.fsencode(directory / "fastwam_driver.py"),
                            os.fsencode(directory / "config")})
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def wait_retry_cleanup(directory, cards):
    receipts = {}
    prior = sorted(directory.glob("prior-oom-retry-*"), key=lambda p: p.stat().st_mtime)
    for folder in [directory, *prior[-1:]]:
        for pattern in ("*driver-identity.json", "native-g*-identity.json", "native-g*-child.json"):
            for path in folder.glob(pattern):
                who = read(path)
                if who and who.get("uid") == os.getuid():
                    receipts[who["pid"]] = who
    ray_job = bind_ray_driver(directory)
    deadline = time.monotonic() + 90
    while True:
        try:
            snapshot = retry_gpu_processes(cards)
        except (OSError, subprocess.SubprocessError, ET.ParseError, TypeError, ValueError) as error:
            save(directory / "oom-admission.json", {"phase": "GPU_QUERY_FAILED",
                 "time": time.time(), "error": str(error)})
            raise RuntimeError("GPU driver query failed during recovery admission") from error
        own = {k: [pid for pid in rows if retry_owner(pid, directory, receipts, ray_job=ray_job)]
               for k, rows in snapshot.items()}
        if not any(own.values()):
            save(directory / "oom-admission.json", {"phase": "READY", "time": time.time(),
                 "other_gpu_processes": {k: len(v) for k, v in snapshot.items()}})
            return
        expired = time.monotonic() >= deadline
        save(directory / "oom-admission.json", {"phase": "CLEANUP_TIMEOUT" if expired else "WAIT_OWN_GPU_WORKERS",
             "time": time.time(), "own_gpu_pids": own})
        if expired:
            raise TimeoutError("Old GPU workers from this deployment still hold device memory")
        time.sleep(5)


def wait_for_oom_retry(directory):
    settings = directory / "wait.json"
    if not settings.is_file():
        return
    gate = read(settings).get("oom_retry_gate")
    if not gate:
        return
    while not Path(gate["release"]).is_file():
        if not same(gate["identity"]):
            # If the CPU gate exited first, a relaunched training job can still be running.
            primary = Path(gate["primary_identity"])
            if not primary.is_file() or not same(read(primary)):
                return
        time.sleep(2)


def wait_for_cards(directory, policies):
    if os.environ.get("PAIR_OOM_RETRY_RULES"):
        return wait_retry_cleanup(directory, policies)
    wait_for_oom_retry(directory)
    previous = None
    while True:
        try:
            snapshot = gpus()
            occupied = {
                str(gpu): [(p["pid"], p["type"]) for p in admission_card(snapshot[gpu], policy)["processes"]]
                for gpu, policy in policies.items()
            }
            if not any(occupied.values()):
                return
            reason = {"occupied": occupied}
        except (OSError, subprocess.SubprocessError) as exc:
            reason = {"gpu_query": type(exc).__name__}
        if reason != previous:
            save(directory / "status.json", {"phase": "WAIT_GPU_RELEASE", "time": time.time(), **reason})
            previous = reason
        time.sleep(3)

