set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
printf 'CONFIG_CANDIDATES\n'
find "$repo" -type f \( -name '*teacher_dvac*.yaml' -o -name '*rlt_stage2*8env*250*.yaml' \) -printf '%p\n' | sort
printf 'HISTORICAL_CANDIDATES\n'
find /root/autodl-tmp/RLinf_rlt_pi0_robotwin -type f \( -name '*resume*480*.yaml' -o -name '*rlt_stage2*8env*250*.yaml' \) -printf '%p\n' | sort
printf 'CONFIG_REFERENCES\n'
rg -n 'teacher_dvac|rlt_stage2_8env250|resume250|480' "$repo/examples" /root/autodl-tmp/RLinf_rlt_pi0_robotwin/examples 2>/dev/null | head -n 300 || true
printf 'RECENT_RUN_EXPORTS\n'
find /root/autodl-tmp/experiment_exports -maxdepth 2 -type f \( -name 'resolved.yaml' -o -name 'exact_command*.txt' \) -path '*rlt*' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort | tail -n 80
