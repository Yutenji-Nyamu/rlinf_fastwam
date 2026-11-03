"""Initialize both native servers concurrently, preserving complete G8 groups."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Client
from pathlib import Path

d = Path(sys.argv[1])
key = (d / "rpc/authkey").read_bytes()
n = int(json.loads((d / "plan.json").read_text())["native_envs_per_card"])
assert n % 8 == 0


def reset(gpu):
    with Client(
        str(d / f"rpc/gpu{gpu}.sock"), family="AF_UNIX", authkey=key
    ) as connection:
        connection.send(
            {
                "op": "reset",
                "mode": "train",
                "layout_ids": [i // 8 for i in range(n)],
                "group_size": 8,
            }
        )
        if not connection.poll(1200):
            raise TimeoutError("Native initial reset timed out")
        result = connection.recv()
        assert result["ok"], result.get("error")
        return {"gpu": gpu, "audit": result["value"]["audit"]}


with ThreadPoolExecutor(2) as pool:
    result = list(pool.map(reset, [6, 7]))
(d / "initial-reset.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
