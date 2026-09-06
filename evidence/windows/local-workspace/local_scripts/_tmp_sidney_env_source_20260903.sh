set -eu
SRC=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
echo '=== configs ==='
sed -n '730,850p' "$SRC/src/lerobot/envs/configs.py"
echo '=== robotwin make env ==='
sed -n '318,680p' "$SRC/src/lerobot/envs/robotwin.py"
echo '=== factory ==='
grep -R -n "make_env_config\|make_env" "$SRC/src/lerobot/envs" "$SRC/src/lerobot/scripts/lerobot_eval.py" | head -120
echo '=== recent tmp files ==='
find /tmp -user chenyiteng -type f -newermt '2026-09-02 13:30:00' -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -120
echo '=== recent data/home scripts ==='
find /data/chenyiteng /home/chenyiteng -xdev -user chenyiteng -type f -newermt '2026-09-02 15:00:00' \( -name '*.sh' -o -name '*.py' -o -name '*.yaml' -o -name '*.toml' -o -name '*.txt' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -200
