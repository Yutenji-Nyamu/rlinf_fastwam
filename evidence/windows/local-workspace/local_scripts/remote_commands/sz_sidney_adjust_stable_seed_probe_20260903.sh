set -euo pipefail
cd /data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/third_party/RoboTwin
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/third_party/RoboTwin${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/chenyiteng/cache/huggingface-sidney
out=/data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab/adjust_bottle-stable-seed-probe-20260903.txt
/home/chenyiteng/venvs/lerobot-v060-sidney-py310/bin/python -u - "$out" <<'PY'
import contextlib
import sys
from lerobot.envs.robotwin import RoboTwinEnv

out_path = sys.argv[1]
stable = []
with open(out_path, "w", encoding="utf-8", buffering=1) as out:
    for seed in range(1003, 1021):
        env = RoboTwinEnv("adjust_bottle", episode_index=seed, n_envs=1)
        try:
            env.reset(seed=seed)
            stable.append(seed)
            line = f"STABLE {seed}"
        except Exception as exc:
            line = f"{type(exc).__name__} {seed} {exc}"
            if type(exc).__name__ != "UnStableError":
                print(line, flush=True)
                print(line, file=out, flush=True)
                raise
        finally:
            with contextlib.suppress(Exception):
                env.close()
        print(line, flush=True)
        print(line, file=out, flush=True)
        if len(stable) == 4:
            break
    summary = "STABLE_SEEDS " + " ".join(map(str, stable))
    print(summary, flush=True)
    print(summary, file=out, flush=True)
    if len(stable) != 4:
        raise SystemExit(2)
PY
