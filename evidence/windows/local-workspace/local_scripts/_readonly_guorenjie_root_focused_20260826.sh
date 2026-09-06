set -eu
export LC_ALL=C

sudo -S -p '' bash -c '
set -eu
export LC_ALL=C

echo "=== GPU UUID MAP ==="
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo "=== GPU PROCESS MAP ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits

echo "=== ACTIVE LEROBOT RANK ENV / SECURITY ==="
for pid in $(pgrep -u guorenjie -f "/lerobot-train" || true); do
  echo "--- pid=$pid"
  tr "\000" "\n" < "/proc/$pid/environ" 2>/dev/null | grep -E "^(CUDA_VISIBLE_DEVICES|LOCAL_RANK|RANK|WORLD_SIZE|MASTER_ADDR|MASTER_PORT)=" || true
  awk "/^(Name|Uid|Gid|TracerPid|CapEff|NoNewPrivs|Seccomp):/{print}" "/proc/$pid/status" 2>/dev/null || true
done

echo "=== ACTIVE LEROBOT NETWORK SOCKETS ==="
ss -tpn 2>/dev/null | grep -E "pid=(1661533|1661536|2642550|2642551)" || true

echo "=== CURRENT ACTIVE OUTPUTS ==="
runuser -u guorenjie -- bash -c "
project=/home/guorenjie/research/smolvla-libero-clp
cd /home/guorenjie
for name in \\
  pi0_libero_fullft_rel_vis_baseline_s30k \\
  pi0_libero_fullft_rel_vis_3m_v2_equal_s30k; do
  d=\"\$project/outputs/train/\$name\"
  if [ -d \"\$d\" ]; then
    du -sh \"\$d\"
    find \"\$d\" -path \"*/training_step.json\" -type f -printf \"%T@ %p\\n\" | sort -n | cut -d\" \" -f2- | while IFS= read -r f; do printf \"%s: \" \"\$f\"; tr -d \"\\n\" < \"\$f\"; echo; done
  else
    echo \"\$name: no checkpoint directory yet\"
  fi
done
"

echo "=== CLP REFERENCE IDENTITY ==="
runuser -u guorenjie -- bash -c "
project=/home/guorenjie/research/smolvla-libero-clp
cd /home/guorenjie
if [ -d \"\$project/refs/CLP_VLA/.git\" ]; then
  git -C \"\$project/refs/CLP_VLA\" rev-parse HEAD
  git -C \"\$project/refs/CLP_VLA\" remote -v | sed -E \"s#(https?://)[^/@]+@#\\1REDACTED@#g\"
fi
sed -n \"1,180p\" \"\$project/refs/CLP_VLA/README.md\" 2>/dev/null || true
"

echo "=== CUSTOM CODE SAFETY MATCHES ==="
runuser -u guorenjie -- bash -c "
project=/home/guorenjie/research/smolvla-libero-clp
repo=\"\$project/lerobot\"
cd /home/guorenjie
grep -RInE \"sudo|rm[[:space:]]+-rf|pkill|kill[[:space:]]|/etc/|chmod|chown|curl|wget|requests\\\\.|subprocess|os\\\\.system\" \\
  \"\$project\"/*.sh \"\$project\"/scripts \\
  \"\$repo/src/lerobot/policies/pi0/clp_prune_pi0.py\" \\
  \"\$repo/src/lerobot/policies/pi0/pi0_vlm_lora.py\" 2>/dev/null | head -n 120 || true
"
'
