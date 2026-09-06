set -u

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
VENV="$REPO/.venvs/robotwin"
CUROBO="$REPO/third_party/RoboTwin/envs/curobo"
date '+TIME %Y-%m-%d %H:%M:%S %Z'
printf 'BUILD_PROCESSES\n'
ps -eo pid,ppid,etimes,stat,%cpu,%mem,args | grep -E 'uv pip install.*curobo|ninja|cmake|nvcc|cc1plus|c\+\+|g\+\+|setup.py.*curobo|build.*curobo' | grep -v grep || true
printf 'CUROBO_RECENT_BUILD_FILES\n'
find "$CUROBO" -type f \( -path '*/build/*' -o -name '*.so' -o -name '*.o' -o -name '*.ninja*' \) -printf '%T@ %s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -n | tail -n 20
printf 'CUROBO_SITE_PACKAGES\n'
find "$VENV/lib/python3.10/site-packages" -maxdepth 1 \( -iname '*curobo*' -o -iname '__editable__*curobo*' \) -printf '%y %s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort
printf 'POLICY_LINK\n'
policy_link="$REPO/third_party/RoboTwin/policy/fasterwam_policy"
if test -L "$policy_link"; then printf '%s -> %s\n' "$policy_link" "$(readlink -f "$policy_link")"; else printf 'ABSENT\n'; fi
printf 'GPU3\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,pstate --format=csv,noheader,nounits | sed -n '4p'
