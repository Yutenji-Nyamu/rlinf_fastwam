set -u
root=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_pair_v9
echo '== files =='
find "$root" -maxdepth 3 -type f -printf '%p\n' 2>/dev/null | sort
echo '== shell and command contents =='
while IFS= read -r path; do
  echo "===== $path ====="
  sed -n '1,260p' "$path"
done < <(find "$root" -maxdepth 3 -type f \( -name '*.sh' -o -name '*command*' -o -name '*launch*' \) 2>/dev/null | sort)
echo '== adjacent formal helpers =='
find /root/autodl-tmp/experiment_exports -maxdepth 3 -type f \( -name '*rlt*formal*.sh' -o -name '*success*bc*.sh' \) -printf '%p\n' 2>/dev/null | sort | tail -40

echo '== per-job v9 runtime =='
for root in \
  /root/autodl-tmp/experiment_exports/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v9/runtime \
  /root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v9/runtime
do
  find "$root" -maxdepth 2 -type f -printf '%p\n' 2>/dev/null | sort
  while IFS= read -r path; do
    echo "===== $path ====="
    sed -n '1,280p' "$path"
  done < <(find "$root" -maxdepth 2 -type f \( -name '*.sh' -o -name '*command*' -o -name '*env*' \) 2>/dev/null | sort)
done

echo '== v9 runtime metadata and log heads =='
for root in \
  /root/autodl-tmp/experiment_exports/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v9/runtime \
  /root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v9/runtime
do
  for name in ray_address.txt cluster_namespace.txt expected_physical_gpu.txt started_at.txt source_head.txt; do
    echo "===== $root/$name ====="
    cat "$root/$name" 2>/dev/null || true
  done
  echo "===== $root/foreground.log head ====="
  sed -n '1,160p' "$root/foreground.log" 2>/dev/null || true
done

echo '== ray head log head =='
sed -n '1,160p' /root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_pair_v9/ray_head.log 2>/dev/null || true
