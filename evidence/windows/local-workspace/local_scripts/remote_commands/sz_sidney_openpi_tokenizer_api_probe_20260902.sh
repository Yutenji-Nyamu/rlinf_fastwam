set -euo pipefail
py=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
"$py" - <<'PY'
import openpi, inspect
print('openpi',openpi.__file__)
for mod in ['openpi.models.tokenizer','openpi.models.tokenizer_pytorch','openpi.shared.tokenizer']:
 try:
  m=__import__(mod,fromlist=['*'])
  print('MOD',mod,m.__file__)
  print([x for x in dir(m) if 'Token' in x or 'token' in x])
 except Exception as e: print('ERR',mod,type(e).__name__,str(e))
PY
root=$($py - <<'PY'
import openpi, pathlib
print(pathlib.Path(openpi.__file__).parent)
PY
)
grep -R -n -E 'class PaligemmaTokenizer|def tokenize' "$root" --include='*.py' | head -80 || true
sed -n '1,62p' "$root/models/tokenizer.py"
