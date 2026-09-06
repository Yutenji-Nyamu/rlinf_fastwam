set -u

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
VENV="$REPO/.venvs/robotwin"
CUROBO="$REPO/third_party/RoboTwin/envs/curobo"
date '+TIME %Y-%m-%d %H:%M:%S %Z'
printf 'INSTALL_PROCESSES\n'
ps -eo pid,ppid,etimes,stat,args | grep -E 'install_robotwin|uv sync|curobo.git|pip.*curobo' | grep -v grep || true
printf 'CUROBO_SOURCE\n'
if test -d "$CUROBO/.git"; then
  git -C "$CUROBO" rev-parse HEAD 2>/dev/null || true
  git -C "$CUROBO" describe --tags --always 2>/dev/null || true
  du -sh "$CUROBO" 2>/dev/null || true
else
  printf 'CUROBO_SOURCE_ABSENT_OR_INCOMPLETE\n'
  test -d "$CUROBO" && du -sh "$CUROBO" 2>/dev/null || true
fi
printf 'CUROBO_SITE_PACKAGES\n'
find "$VENV/lib/python3.10/site-packages" -maxdepth 1 \( -iname '*curobo*' -o -iname '__editable__*curobo*' \) -printf '%y %s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort
printf 'ASSET_LINKS\n'
for name in background_texture embodiments objects; do
  path="$REPO/third_party/RoboTwin/assets/$name"
  if test -L "$path"; then printf '%s -> %s\n' "$path" "$(readlink -f "$path")"; else printf 'NOT_LINK %s\n' "$path"; fi
done
printf 'VENV_SIZE\n'
du -sh "$VENV" 2>/dev/null || true
printf 'POLICY_LINK\n'
policy_link="$REPO/third_party/RoboTwin/policy/fasterwam_policy"
if test -L "$policy_link"; then printf '%s -> %s\n' "$policy_link" "$(readlink -f "$policy_link")"; else printf 'POLICY_LINK_ABSENT\n'; fi
