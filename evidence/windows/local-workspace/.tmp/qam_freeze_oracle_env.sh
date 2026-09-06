set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/venvs/qam-oracle-2726d767
output="$repo/tests/algorithms/qam/oracle/resolved-freeze.txt"

test -x "$venv/bin/python"
test ! -e "$output"
"$venv/bin/python" -m pip freeze --all | LC_ALL=C sort > "$output"
sha256sum "$output"
wc -l "$output"
