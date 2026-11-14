"""Check render-runtime resource lifetime and retain a fixed-seed observation."""

import ctypes as C
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

libc = C.CDLL(None)
libc.pthread_key_create.argtypes = [C.POINTER(C.c_uint), C.c_void_p]
libc.pthread_key_delete.argtypes = [C.c_uint]


def count():
    keys = []
    for _ in range(2048):
        key = C.c_uint()
        if libc.pthread_key_create(C.byref(key), None):
            break
        keys.append(key.value)
    for key in keys:
        libc.pthread_key_delete(key)
    return len(keys)


cfg = OmegaConf.load(sys.argv[1])
out = Path(sys.argv[2])
cycles = int(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 2
t = time.monotonic()
env = RoboTwinEnv(cfg.env.train, n, 0, 1, None)
for cycle in range(cycles):
    obs, _ = env.reset()
    active = count()
    if cycle == 0:
        np.savez_compressed(
            str(out) + ".npz",
            **{
                k: (v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v))
                for k, v in obs.items()
            },
        )
    actions = obs["states"].cpu().numpy()[:, None, :].repeat(2, axis=1)
    env.step(actions)
    env.offload()
    gc.collect()
    print(
        "RUNTIME "
        + json.dumps(
            {
                "cycle": cycle,
                "active_free_tls": active,
                "offloaded_free_tls": count(),
                "seconds": round(time.monotonic() - t, 2),
            }
        ),
        flush=True,
    )
print(
    "LIBRARIES "
    + json.dumps(
        sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "OpenImageDenoise" in line
            }
        )
    ),
    flush=True,
)
print("RUNTIME_COMPLETE", flush=True)
