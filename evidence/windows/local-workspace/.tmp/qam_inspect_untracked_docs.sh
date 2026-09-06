set -eu
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
find "$repo/docs/rlinf-robotwin-pi0-qam" \
  -maxdepth 3 -type f -printf '%s %TY-%Tm-%Td %TH:%TM:%TS %p\n' |
  sort
