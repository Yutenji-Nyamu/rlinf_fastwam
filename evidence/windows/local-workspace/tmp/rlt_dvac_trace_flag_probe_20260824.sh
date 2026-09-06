set -u
trace_root=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1/robotwin_adjust_bottle_rlt_teacher_dvac_w0to2_smoke_8env1c_v1/rlt_dvac
cd "$trace_root"
/root/autodl-tmp/RLinf/.venv/bin/python -c 'import glob,numpy as np; [(lambda d,p: print(p, d["actor_switch"].astype(int).reshape(-1).tolist()))(np.load(p),p) for p in sorted(glob.glob("actor_rank*/*.npz"))]'
