set -euo pipefail
RT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

test "$(git -C "$RT" rev-parse HEAD)" = 0008ae6800df9f75fc8de7098bacb01735fd8fd2
test "$(git -C "$RT" status --porcelain | wc -l)" -eq 1
test "$(git -C "$RT" status --porcelain | awk '{print $2}')" = robotwin/envs/vector_env.py

"$VENV/bin/python" -m py_compile "$RT/robotwin/envs/vector_env.py"
PYTHONPATH="$RT:$RT/robotwin" "$VENV/bin/python" - <<'PY'
from robotwin.envs import vector_env as module

events = []

class FakeSubEnv:
    def __init__(self, index):
        self.index = index

    def close(self):
        events.append(("close", self.index))

    def reset(self, env_seed=None, close_current=True):
        events.append(("reset", self.index, env_seed, close_current))

module.gc.collect = lambda: events.append(("gc",))
module.sapien_clear_cache = lambda: events.append(("global_clear",))
module.torch.cuda.empty_cache = lambda: events.append(("cuda_empty",))

venv = module.VectorEnv.__new__(module.VectorEnv)
venv.n_envs = 2
venv.envs = [FakeSubEnv(0), FakeSubEnv(1)]
venv.full_reset_count = 0
venv.clear_cache_freq = 1

venv.reset(env_idx=None, env_seeds=[101, 202])
assert events == [
    ("close", 0),
    ("close", 1),
    ("gc",),
    ("global_clear",),
    ("cuda_empty",),
    ("reset", 0, 101, False),
    ("reset", 1, 202, False),
], events

events.clear()
venv.reset(env_idx=[1], env_seeds=[303])
assert events == [("reset", 1, 303, True)], events

events.clear()
venv.close(clear_cache=True)
assert events == [
    ("close", 0),
    ("close", 1),
    ("gc",),
    ("global_clear",),
    ("cuda_empty",),
], events
assert venv.envs == []

source = open(module.__file__, encoding="utf-8").read()
reset_source = source[source.index("    def reset(self, env_idx=None"):source.index("    def get_obs(self):", source.index("    def reset(self, env_idx=None"))]
assert "env_thread_pool.submit" not in reset_source
assert "self.env_thread_pool.submit(self.envs[i].step" in source
print("lifecycle_event_order=PASS")
print(f"vector_env={module.__file__}")
PY

git -C "$RT" diff --check
git -C "$RT" diff --stat
git -C "$RT" diff -- robotwin/envs/vector_env.py
