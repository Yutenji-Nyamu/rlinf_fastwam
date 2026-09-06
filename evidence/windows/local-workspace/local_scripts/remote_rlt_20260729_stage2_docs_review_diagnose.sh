#!/usr/bin/env bash
set -u
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
code_commit=3b610cb4685a1d41c97da64df67ab86561697dfd
status=0

for script in \
  "${repo}"/docs/rlinf-robotwin-pi0-rltoken/evidence/\
stage2_pre_smoke_20260729/scripts/*.sh
do
  printf 'bash_n\t%s\t' "${script#${repo}/}"
  if bash -n "${script}"; then
    printf '%s\n' OK
  else
    printf '%s\n' FAIL
    status=1
  fi
done

printf '%s\n' "non_docs_diff_current_pathspec"
git -C "${repo}" diff --name-only "${code_commit}" -- . ':(exclude)docs/**'
if git -C "${repo}" diff --quiet \
  "${code_commit}" -- . ':(exclude)docs/**'
then
  printf '%s\n' "non_docs_gate_current=OK"
else
  printf '%s\n' "non_docs_gate_current=FAIL"
  status=1
fi

printf '%s\n' "non_docs_diff_with_handoff_excluded"
git -C "${repo}" diff --name-only \
  "${code_commit}" -- . ':(exclude)docs/**' ':(exclude)HANDOFF.md'
if git -C "${repo}" diff --quiet \
  "${code_commit}" -- . ':(exclude)docs/**' ':(exclude)HANDOFF.md'
then
  printf '%s\n' "non_docs_gate_with_handoff=OK"
else
  printf '%s\n' "non_docs_gate_with_handoff=FAIL"
  status=1
fi

printf 'diagnostic_status\t%s\n' "${status}"
exit 0
