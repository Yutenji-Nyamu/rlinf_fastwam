set -u

printf '%s\n' PLANNER_SOURCE_BEGIN
sed -n '275,325p' /root/autodl-tmp/RoboTwin_RLinf/envs/robot/planner.py
printf '%s\n' PLANNER_SOURCE_END
printf '%s\n' CUROBO_CANDIDATES_BEGIN
find /root/autodl-tmp/RLinf/.venv/lib/python3.11/site-packages /root/autodl-tmp/RoboTwin_RLinf /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin \
  -maxdepth 4 \( -name 'curobo.py' -o -path '*/curobo/__init__.py' -o -name 'types.py' \) \
  -print 2>/dev/null | sort
printf '%s\n' CUROBO_CANDIDATES_END
