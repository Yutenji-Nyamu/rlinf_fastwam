set -eu
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
CFG="$WT/examples/embodiment/config/robotwin_move_stapler_pad_grpo_openpi_pi05_sidney.yaml"
RES=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/b1-adjust-badseed-retry-m10-phys4-v11/resolved.yaml
echo '=== YAML model/rollout references ==='
grep -n -C 5 -E '^(actor|rollout):|model:|model_type|model_path|openpi:' "$CFG" | head -240
echo '=== resolved top-level actor/rollout keys ==='
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - "$RES" <<'PY'
import sys,yaml
c=yaml.safe_load(open(sys.argv[1]))
for k in ('actor','rollout'):
 x=c.get(k,{})
 print(k, 'keys=',sorted(x))
 print(k+'.model=',x.get('model'))
PY
echo '=== official pi05 configs rollout model forms ==='
grep -R -n -m 30 -E '^rollout:|^[[:space:]]+model:.*actor|model_type: openpi' "$WT/examples/embodiment/config" | head -160
