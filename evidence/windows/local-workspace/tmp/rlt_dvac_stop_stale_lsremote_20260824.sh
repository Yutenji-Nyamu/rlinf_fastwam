#!/usr/bin/env bash
set -euo pipefail

pid=495982
if [ -r "/proc/$pid/cmdline" ]; then
  command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
  case "$command_line" in
    *"git -C /root/autodl-tmp/RLinf_rlt_teacher_dvac ls-remote --heads personal refs/heads/codex/rlt-teacher-dvac-weighting"*)
      kill -TERM "$pid"
      ;;
    *)
      printf 'REFUSE_UNEXPECTED_PID=%s CMD=%s\n' "$pid" "$command_line"
      exit 1
      ;;
  esac
fi
sleep 1
if [ -r "/proc/$pid/cmdline" ]; then
  echo STALE_LSREMOTE_ALIVE
  exit 1
fi
echo STALE_LSREMOTE_STOPPED
git -C /root/autodl-tmp/RLinf_rlt_teacher_dvac status --short --branch
git -C /root/autodl-tmp/RLinf_rlt_teacher_dvac rev-list --left-right --count '@{upstream}...HEAD'
