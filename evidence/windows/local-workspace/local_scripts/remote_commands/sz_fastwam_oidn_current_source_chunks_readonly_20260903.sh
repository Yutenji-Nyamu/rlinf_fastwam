set -eu
RT=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
show(){ f=$1; a=$2; b=$3; echo "=== $f:$a-$b ==="; nl -ba "$f" | sed -n "${a},${b}p"; }
show "$RT/robotwin/envs/vector_env.py" 55 210
show "$RT/robotwin/envs/vector_env.py" 310 430
show "$RT/envs/_base_task.py" 195 230
show "$RT/envs/_base_task.py" 620 670
show "$WT/rlinf/envs/robotwin/robotwin_env.py" 75 100
show "$WT/rlinf/envs/robotwin/robotwin_env.py" 235 260
show "$WT/rlinf/envs/robotwin/robotwin_env.py" 390 416
show "$WT/rlinf/workers/env/env_worker.py" 430 455
show "$WT/rlinf/workers/env/env_worker.py" 1355 1462
