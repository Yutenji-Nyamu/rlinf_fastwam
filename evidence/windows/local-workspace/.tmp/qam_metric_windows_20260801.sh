set -u
log=/root/autodl-tmp/experiment_exports/qam_formal_resume25_to100_20260801_v3/runtime/driver.log

for metric in \
  qam/am_loss \
  qam/fine_grad_norm \
  qam/terminal_adjoint_norm \
  qam/critic_loss \
  qam/critic_grad_norm \
  qam/q_mean \
  qam/q_std_heads \
  qam/td_target_mean \
  qam/prefix_roundtrip_mean_abs \
  qam/prefix_roundtrip_min_cosine; do
  printf '%s\t' "$metric"
  grep -aoE "$metric=[-+0-9.e]+" "$log" 2>/dev/null | cut -d= -f2 | \
    awk '{a[++n]=$1; if(n==1||$1<mn)mn=$1; if(n==1||$1>mx)mx=$1}
         END {k=(n<5?n:5); for(i=1;i<=k;i++)f+=a[i]; for(i=n-k+1;i<=n;i++)l+=a[i];
              printf "n=%d first5=%.6g last5=%.6g min=%.6g max=%.6g\n",n,f/k,l/k,mn,mx}'
done
