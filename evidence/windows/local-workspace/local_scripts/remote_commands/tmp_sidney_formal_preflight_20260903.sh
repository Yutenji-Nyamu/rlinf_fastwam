set -eu
echo '== identity =='
whoami
hostname
echo '== gpu =='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo '== compute =='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
echo '== relevant processes =='
ps -eo pid,ppid,pgid,etime,user,args --sort=pid | grep -E 'train_embodied_agent|eval_embodied_agent|sidney-pi05|fastwam-grpo' | grep -v grep || true
echo '== host =='
free -h | sed -n '1,2p'
df -h / /home /data | tail -n +2
echo '== ray =='
ss -ltnp 2>/dev/null | grep ':6389' || true
echo '== source =='
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
git -C "$WT" status --short
git -C "$WT" rev-parse HEAD
git -C "$WT" branch --show-current
echo '== config =='
CFG="$WT/examples/embodiment/config/robotwin_move_stapler_pad_grpo_openpi_pi05_sidney.yaml"
test -f "$CFG"
sed -n '1,260p' "$CFG"
echo '== smoke packet command =='
P=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v12
sed -n '1,240p' "$P/run.sh"
echo '== readmes =='
find /home -maxdepth 1 -type f -iname '*readme*' -printf '%p %s\n' | sort
