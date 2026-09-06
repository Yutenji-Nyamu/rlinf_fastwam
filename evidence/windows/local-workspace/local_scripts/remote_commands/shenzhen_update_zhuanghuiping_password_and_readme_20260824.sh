#!/usr/bin/env bash
set -euo pipefail

printf 'MARKER=SZ_ACCOUNT_PASSWORD_AND_README_UPDATE_V1\n'

# stdin line 1 is consumed by sudo; line 2 is the chpasswd record.
sudo -S -p '' /usr/sbin/chpasswd

sudo -n python3 - <<'PY'
from pathlib import Path

path = Path('/home/readme_to_codex.md')
old = '- `chenyiteng` 最近在赶 ICLR，需要 4 张 GPU，通常优先使用 4、5、6、7 号卡；如需使用这些卡，请先联系他。'
new = (
    '- 使用本服务器的人比较多，要使用的人尽量先联系相应人员，或者加入微信群；\n'
    '- chenyiteng 大约10月1日前在赶iclr 27，他经常使用4张卡，例如4567，请不要杀死他的任务，或者使剩余的卡过少，或者，请联系他；'
)
text = path.read_text(encoding='utf-8')
if text.count(old) != 1:
    raise SystemExit(f'expected exactly one old README line, found {text.count(old)}')
if new in text:
    raise SystemExit('replacement text already present before update')
path.write_text(text.replace(old, new), encoding='utf-8')
PY

printf '%s\n' '=== README COLLABORATION ==='
sed -n '/^## 协作$/,/^## /p' /home/readme_to_codex.md
printf '%s\n' '=== ACCOUNT ==='
getent passwd zhuanghuiping
id zhuanghuiping
sudo -n -l -U zhuanghuiping
printf 'MARKER=SZ_ACCOUNT_PASSWORD_AND_README_UPDATE_OK\n'
