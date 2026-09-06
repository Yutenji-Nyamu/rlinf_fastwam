set -u
runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1/runtime
printf 'RAY_SUCCESS_MATCHES\n'
grep -RaniE 'success(_rate)?|reward_positive|episode_return|episode_reward' /tmp/ray/session_latest/logs --include='*.out' --include='*.err' 2>/dev/null | head -300 || true
printf 'RUN_OUTPUT_FILES\n'
find /root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1 -maxdepth 5 -type f -printf '%p %s\n' 2>/dev/null | sort | head -300 || true
printf 'ACTOR_SWITCH_TRACE_VALUES\n'
cd "$runtime"
/root/autodl-tmp/RLinf/.venv/bin/python -c "import glob,numpy as np; [(lambda d,p: print(p, d['actor_switch'].astype(int).reshape(-1).tolist()))(np.load(p),p) for p in sorted(glob.glob('../rlt_dvac_traces/actor_rank*/*.npz'))]" 2>/dev/null || true
