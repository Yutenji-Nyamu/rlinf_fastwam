#!/usr/bin/env bash
set -euo pipefail

bundle=/root/autodl-tmp/experiment_exports/rlt_base_48a_20260729.bundle
prefix=/root/autodl-tmp/experiment_exports/rlt_base_48a_20260729.bundle.part.

if compgen -G "${prefix}*" >/dev/null; then
  echo "refusing: bundle parts already exist" >&2
  exit 10
fi

split -b 4000000 -d -a 2 "$bundle" "$prefix"
sha256sum "${prefix}"*
stat -c '%s %n' "${prefix}"*
