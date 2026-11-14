"""CPU regression tests for process offload without a simulator dependency."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from process_vector_env import ProcessVectorEnv

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp)
    (p / "fake_vector.py").write_text("""import os
class Vector:
 def __init__(self, seeds): self.seeds=list(seeds); self.steps=0
 def reset(self, env_idx=None, env_seeds=None):
  ids=list(range(len(self.seeds))) if env_idx is None else env_idx
  for i,s in zip(ids,env_seeds):self.seeds[i]=s
 def get_obs(self):return {"seeds":self.seeds,"steps":self.steps,"pid":os.getpid()}
 def step(self, actions):
  if actions=="error":raise ValueError("known remote error")
  self.steps+=1
  return (actions,[1,0],[True,False],[False,True],{"success":[True,False]})
 def close(self, clear_cache=True):pass
 def check_seeds(self,seeds):return [s>=0 for s in seeds]
""")
    os.environ["PYTHONPATH"] = tmp + os.pathsep + os.environ.get("PYTHONPATH", "")
    e = ProcessVectorEnv(factory="fake_vector:Vector", seeds=[10, 20], rpc_timeout_s=10)
    assert e.worker_pid is None
    pids = []
    for cycle in range(16):
        e.reset(env_seeds=[100 + cycle, 200 + cycle])
        o = e.get_obs()
        pids.append(o["pid"])
        assert o["seeds"] == [100 + cycle, 200 + cycle] and o["steps"] == 0
        e.reset(env_idx=[1], env_seeds=[999])
        assert e.get_obs()["seeds"] == [100 + cycle, 999]
        assert e.step([[1], [2]]) == (
            [[1], [2]],
            [1, 0],
            [True, False],
            [False, True],
            {"success": [True, False]},
        )
        assert e.check_seeds([1, -1]) == [True, False]
        try:
            e.step("error")
        except RuntimeError as ex:
            assert "known remote error" in str(ex)
        else:
            raise AssertionError("remote exception lost")
        e.close()
        assert e.worker_pid is None and not Path("/proc/" + str(pids[-1])).exists()
    assert len(set(pids)) == 16
    # An abruptly stopped owner must not leave its simulator alive.
    record = p / "abrupt.json"
    code = (
        "import sys,time,json; from pathlib import Path; "
        + "sys.path.insert(0,"
        + repr(str(Path(__file__).parent))
        + "); "
        + "from process_vector_env import ProcessVectorEnv; "
        + "e=ProcessVectorEnv(factory='fake_vector:Vector',seeds=[1,2]); "
        + "o=e.get_obs(); Path("
        + repr(str(record))
        + ").write_text(json.dumps(o)); "
        + "time.sleep(30)"
    )
    owner = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.monotonic() + 10
        while not record.exists():
            assert owner.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        child_pid = json.loads(record.read_text())["pid"]
        owner.terminate()
        owner.wait(timeout=10)
        state = Path("/proc") / str(child_pid) / "stat"
        deadline = time.monotonic() + 5
        while state.exists() and state.read_text().split(") ", 1)[1].split()[0] != "Z":
            assert time.monotonic() < deadline, "Orphan simulator survived its owner"
            time.sleep(0.05)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=10)
print(
    json.dumps(
        {
            "cycles": 16,
            "unique_children": 16,
            "full_partial_reset": True,
            "step_reward_termination_order": True,
            "exceptions_forwarded": True,
            "all_children_reaped": True,
            "abrupt_owner_exit": True,
        }
    )
)
