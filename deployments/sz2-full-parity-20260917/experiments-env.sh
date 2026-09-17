# Generated full-parity environment selector. Does not start any process.
_family="${1:-pi05-grpo-exp-controls-20260917}"
_parity_base=/data/chenyiteng/deployment-20260917/full-parity/code
if [ ! -x /data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin/python ]; then
  echo "Parity environment is still being prepared" >&2; return 1
fi
_eval_exports=$( /data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin/python "$_parity_base/experiment_setup.py" env "$_family" ) || return $?
eval "$_eval_exports"
unset _family _parity_base _eval_exports
