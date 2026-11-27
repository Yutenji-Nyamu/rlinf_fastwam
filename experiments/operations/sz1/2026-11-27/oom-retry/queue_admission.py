"""Wait for the exact requested slots; a busy predecessor keeps its successor queued."""
import subprocess
import time
from common import admission_card, gpus, read, same, save
from pathlib import Path


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

