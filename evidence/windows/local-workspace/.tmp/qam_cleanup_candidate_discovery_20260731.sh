set -eu

printf 'TIME=%s\n' "$(date '+%F %T %Z')"
printf 'FASTWAM_SMOKES_BEGIN\n'
find /root/autodl-tmp/RLinf_fastwam_rlinf/logs \
  -mindepth 1 -maxdepth 1 -type d \
  \( -iname '*smoke*' -o -iname '*thin*' \) \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
printf 'FASTWAM_SMOKES_END\n'

printf 'RLT_STAGE1_BEGIN\n'
find /root/autodl-tmp/experiments \
  -mindepth 1 -maxdepth 2 -type d \
  -iname '*rlt*stage1*' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
printf 'RLT_STAGE1_END\n'

printf 'WAM_BACKUP_BEGIN\n'
find /root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40 \
  -mindepth 1 -maxdepth 2 \
  -printf '%y %TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort | head -120
printf 'WAM_BACKUP_END\n'
