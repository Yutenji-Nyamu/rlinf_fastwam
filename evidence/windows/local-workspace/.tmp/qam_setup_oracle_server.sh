set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
source_dir=/root/autodl-tmp/oracles/qam-2726d767
venv_dir=/root/autodl-tmp/venvs/qam-oracle-2726d767
commit=2726d767c9a0a7a46d49693f0391f73dc2cf58ac

test -d "$source_dir/.git"
test ! -e "$venv_dir"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$source_dir" status --short)"

/root/autodl-tmp/RLinf/.venv/bin/python -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install \
  --disable-pip-version-check \
  --no-input \
  --no-cache-dir \
  -r "$repo/tests/algorithms/qam/oracle/requirements.lock.txt"

"$venv_dir/bin/python" - <<'PY'
import distrax
import flax
import jax
import jaxlib
import ml_collections
import numpy
import optax
import tensorflow_probability

print("numpy", numpy.__version__)
print("jax", jax.__version__)
print("jaxlib", jaxlib.__version__)
print("flax", flax.__version__)
print("optax", optax.__version__)
print("distrax", distrax.__version__)
print("tensorflow_probability", tensorflow_probability.__version__)
print("ml_collections", ml_collections.__version__)
PY

du -sx --block-size=1 "$source_dir" "$venv_dir"
