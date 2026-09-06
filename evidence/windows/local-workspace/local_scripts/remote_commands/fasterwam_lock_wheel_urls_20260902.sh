set -euo pipefail

LOCK=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official/environments/robotwin/uv.lock
test -s "$LOCK"
printf 'TORCH_2_4_1_CU121_X86_64\n'
grep -oE 'https://[^" ]*torch-2\.4\.1[^" ]*(x86_64|manylinux[^" ]*)\.whl' "$LOCK" | sort -u
printf 'EMBREEX_X86_64\n'
grep -oE 'https://[^" ]*embreex[^" ]*(x86_64|manylinux[^" ]*)\.whl' "$LOCK" | sort -u
