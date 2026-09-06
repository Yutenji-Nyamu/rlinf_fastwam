set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
source_dir=/root/autodl-tmp/oracles/qam-2726d767
venv_dir=/root/autodl-tmp/venvs/qam-oracle-2726d767
output="$repo/tests/algorithms/qam/oracle/qam_official_2726d767_v1.npz"
commit=2726d767c9a0a7a46d49693f0391f73dc2cf58ac

test "$(git -C "$source_dir" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$source_dir" status --short)"
test -x "$venv_dir/bin/python"
test ! -e "$output"

"$venv_dir/bin/python" - <<'PY'
from importlib.metadata import version

for name in (
    "numpy",
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "distrax",
    "tensorflow-probability",
    "ml-collections",
):
    print(name, version(name))
PY

JAX_PLATFORMS=cpu \
TF_CPP_MIN_LOG_LEVEL=2 \
"$venv_dir/bin/python" \
  "$repo/tests/algorithms/qam/oracle/export_official_fixture.py" \
  --source "$source_dir" \
  --output "$output"

sha256sum "$output"
stat --format='%s %y %n' "$output"
du -sx --block-size=1 "$source_dir" "$venv_dir"
