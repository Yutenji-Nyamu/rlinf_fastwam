set -euo pipefail
RT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
test "$(git -C "$RT" rev-parse HEAD)" = 0008ae6800df9f75fc8de7098bacb01735fd8fd2
test "$(git -C "$RT" status --porcelain | wc -l)" -eq 1
test "$(git -C "$RT" status --porcelain | awk '{print $2}')" = robotwin/envs/vector_env.py
test "$(sha256sum "$RT/robotwin/envs/vector_env.py" | awk '{print $1}')" = 863ab2a6f8f03742b4918bc049160f64200d814d5cc19c54d2fa35781407e93b
git -C "$RT" diff --check
git -C "$RT" add robotwin/envs/vector_env.py
git -C "$RT" -c user.name=Yutenji-Nyamu -c user.email=1842710211@qq.com commit -m "fix(robotwin): own renderer cache at vector scope"
git -C "$RT" rev-parse HEAD
git -C "$RT" status --short --branch
