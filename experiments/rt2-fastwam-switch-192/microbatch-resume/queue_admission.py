"""Wait for the exact requested slots; a busy predecessor keeps its successor queued."""
import subprocess
import time
from common import admission_card, gpus, save


def wait_for_cards(directory, policies):
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

