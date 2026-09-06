#!/usr/bin/env bash
set -euo pipefail

read -r -d '' ROOT_SCRIPT <<'ROOT' || true
set -euo pipefail
IFS= read -r new_password
user=zhuanghuiping
main_readme=/home/readme_to_codex.md

if getent passwd "$user" >/dev/null; then
  printf 'account already exists: %s\n' "$user" >&2
  exit 2
fi
test ! -e "/home/$user"
test ! -e "/data/$user"
getent group sudo >/dev/null
getent group labdata >/dev/null
test -f "$main_readme"

home_mode=$(stat -c '%a' /home/chenyiteng)
data_mode=$(stat -c '%a' /data/chenyiteng)
useradd -m -s /bin/bash -G sudo,labdata "$user"
printf '%s:%s\n' "$user" "$new_password" | chpasswd
unset new_password

chmod "$home_mode" "/home/$user"
install -d -o "$user" -g "$user" -m "$data_mode" "/data/$user"
ln -s "/data/$user" "/home/$user/data"
ln -s /data/shared "/home/$user/shared"
chown -h "$user:$user" "/home/$user/data" "/home/$user/shared"

python3 - "$main_readme" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
line = "- `chenyiteng` 最近在赶 ICLR，需要 4 张 GPU，通常优先使用 4、5、6、7 号卡；如需使用这些卡，请先联系他。\n"
if line not in text:
    needle = "## 协作\n\n"
    if text.count(needle) != 1:
        raise SystemExit("main README collaboration heading is not unique")
    text = text.replace(needle, needle + line, 1)
    path.write_text(text, encoding="utf-8")
PY

printf '%s\n' '=== ACCOUNT ==='
getent passwd "$user"
id "$user"
passwd -S "$user"
stat -c '%A %U:%G %n' "/home/$user" "/data/$user" "/home/$user/data" "/home/$user/shared"
printf '%s\n' '=== SUDO ==='
sudo -l -U "$user" | sed -n '1,30p'
printf '%s\n' '=== README INSERT ==='
grep -n -F '`chenyiteng` 最近在赶 ICLR' "$main_readme"
stat -c '%A %U:%G %s %n' "$main_readme"
printf 'new_account_ready=%s\n' "$user"
ROOT

printf 'MARKER=SZ_CREATE_ZHUANGHUIPING_UPDATE_README_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
IFS= read -r shared_secret
{
  printf '%s\n' "$shared_secret"
  printf '%s\n' "$shared_secret"
} | sudo -S -k -p '' /bin/bash -c "$ROOT_SCRIPT"
unset shared_secret ROOT_SCRIPT
printf 'MARKER=SZ_CREATE_ZHUANGHUIPING_UPDATE_README_OK\n'
