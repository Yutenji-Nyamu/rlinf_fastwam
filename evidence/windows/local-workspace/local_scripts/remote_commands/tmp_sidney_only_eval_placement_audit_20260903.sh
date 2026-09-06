set -euo pipefail
cd /data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
grep -R -n -E "only_eval: [Tt]rue|task_type: embodied_eval" examples/embodiment/config | head -80 || true
grep -R -n -E "component_placement:.*env.*rollout|env, rollout" examples/embodiment/config | head -100 || true
grep -R -n "sync_model_from_actor" rlinf | head -80 || true
sed -n '1,130p' examples/embodiment/config/realworld_eval_dual_franka.yaml
sed -n '120,230p' rlinf/runners/embodied_runner.py
grep -R -n -E "class .*Eval.*Runner|only_eval" rlinf/runners rlinf/scheduler | head -120 || true
sed -n '1,130p' rlinf/runners/embodied_runner.py
sed -n '1,240p' rlinf/runners/embodied_eval_runner.py
