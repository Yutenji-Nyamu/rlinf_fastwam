#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
date -Is
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
from pathlib import Path
import subprocess, shutil, hashlib
paths=sorted({line.split()[-1] for line in Path('/proc/1053120/maps').read_text().splitlines() if 'libsapien' in line or 'svulkan' in line})
print('LOADED_NATIVE_LIBRARIES',paths)
for path in paths:
 print('FILE_SHA256',path,hashlib.file_digest(open(path,'rb'),'sha256').hexdigest())
 if not shutil.which('nm'):continue
 symbols=subprocess.run(['nm','-D',path],capture_output=True,text=True,timeout=20).stdout.splitlines()
 for line in symbols:
  if 'RTRenderer6render' not in line:continue
  print('RENDER_SYMBOL',line)
  if shutil.which('objdump'):
   symbol=line.split()[-1]
   p=subprocess.run(['objdump','-d','--disassemble='+symbol,path],capture_output=True,text=True,timeout=30)
   # No process attach. Bounded binary text, enough to inspect dispatch and fence arguments.
   print(p.stdout[:55000])
PY
date -Is
