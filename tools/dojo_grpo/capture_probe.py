"""Same initial state/actions, native per-action vs chunk-boundary capture."""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
plan = json.loads((d / "plan.json").read_text())
key = (d / "rpc/authkey").read_bytes()
n = int(plan["native_envs_per_card"])
actions = np.repeat(
    np.load(plan["capture_probe_actions"])["actions"][None, :10], n, axis=0
)


def run(gpu):
    started = time.monotonic()
    with Client(
        str(d / f"rpc/gpu{gpu}.sock"), family="AF_UNIX", authkey=key
    ) as connection:
        connection.send(
            {
                "op": "step",
                "mode": "train",
                "actions": actions,
                "capture_every_action": gpu == 6,
            }
        )
        if not connection.poll(800):
            raise TimeoutError("Native capture comparison")
        result = connection.recv()
    assert result["ok"], result.get("error")
    return result["value"], time.monotonic() - started


with ThreadPoolExecutor(2) as pool:
    (original, slow), (boundary, fast) = list(pool.map(run, [6, 7]))
for name, result in [("action", original), ("boundary", boundary)]:
    np.savez_compressed(
        d / f"capture-{name}.npz",
        **{k: v for k, v in result["obs"].items() if v is not None},
    )
state_error = float(np.abs(original["obs"]["states"] - boundary["obs"]["states"]).max())
images = {}
for name in ["main_images", "wrist_images"]:
    a = original["obs"][name].astype(np.float32)
    b = boundary["obs"][name].astype(np.float32)
    images[name + "_mean_absolute_error"] = float(np.abs(a - b).mean())
same_outcome = all(
    np.array_equal(original[k], boundary[k])
    for k in ["counts", "rewards", "terminations", "truncations", "success"]
)
receipt = {
    "actions": 10,
    "environments_each": n,
    "per_action_seconds": slow,
    "boundary_seconds": fast,
    "state_max_abs": state_error,
    "same_reward_and_end_flags": same_outcome,
    **images,
}
(d / "capture-comparison.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
assert same_outcome and state_error < 0.005, receipt
# Renderer history can change pixels while physical states are identical.
# Evaluate their effect through the actual policy, not an arbitrary pixel cutoff.
