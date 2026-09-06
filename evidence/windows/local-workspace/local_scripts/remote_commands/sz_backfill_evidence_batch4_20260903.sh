#!/usr/bin/env bash
set -euo pipefail
STAMP=completed_runs_20260822_20260903

copy_run() {
  local run=$1 out=$2 file rel root packet
  test -d "$run"; mkdir -p "$out/run"
  while IFS= read -r -d '' file; do
    rel=${file#"$run"/}; mkdir -p "$out/run/$(dirname "$rel")"; cp -- "$file" "$out/run/$rel"
  done < <(find "$run" \
    -path '*/checkpoints/*' -prune -o -path '*/video/*' -prune -o -path '*/videos/*' -prune -o \
    -path '*/robotwin_data/*' -prune -o -path '*/ray_logs*' -prune -o \
    -type f -size -20M \
    \( -name '*.log' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \
       -o -name '*.txt' -o -name '*.jsonl' -o -name '*.sh' -o -name 'events.out.tfevents*' -o -name '*.png' \) -print0)
  case "$run" in */runs/*)
    root=${run%%/runs/*}; packet="$root/packets/$(basename "$run")"
    if test -d "$packet"; then
      mkdir -p "$out/packet"
      while IFS= read -r -d '' file; do
        rel=${file#"$packet"/}; mkdir -p "$out/packet/$(dirname "$rel")"; cp -- "$file" "$out/packet/$rel"
      done < <(find "$packet" -type f -size -2M \
        \( -name '*.log' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \
           -o -name '*.txt' -o -name '*.jsonl' -o -name '*.sh' \) -print0)
    fi;; esac
  printf '%s\n' "$run" > "$out/SERVER_RUN_PATH.txt"
  (cd "$out" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
}

stage_push() {
  local wt=$1 branch=$2 expected=$3; shift 3
  local out="$wt/evidence/$STAMP" run head remote
  test "$(git -C "$wt" branch --show-current)" = "$branch"
  test "$(git -C "$wt" rev-parse HEAD)" = "$expected"
  test -z "$(git -C "$wt" status --porcelain --untracked-files=all)"
  test ! -e "$out"; mkdir -p "$out"
  printf '# Completed Shenzhen run evidence\n\nCopied from immutable server paths named in each `SERVER_RUN_PATH.txt`; checkpoints, models, videos, simulator data and full Ray logs are excluded.\n' > "$out/README.md"
  for run in "$@"; do copy_run "$run" "$out/$(basename "$run")"; done
  test -z "$(find "$out" -type f \( -iname '*.pt' -o -iname '*.pth' -o -iname '*.bin' -o -iname '*.safetensors' \
    -o -iname '*.ckpt' -o -iname '*.mp4' -o -iname '*.npz' -o -iname '*.npy' -o -iname '*.h5' \
    -o -iname '*.hdf5' -o -iname '*.zip' -o -iname '*.tar.gz' \) -print -quit)"
  if grep -ERil --binary-files=without-match -e '[REDACTED]' -e 'github_pat_' \
      -e 'hf_[A-Za-z0-9]\{20,\}' -e 'authorization:[[:space:]]*bearer' \
      -e 'BEGIN \(RSA\|OPENSSH\|EC\) PRIVATE KEY' "$out" | grep -q .; then
    echo "credential-like material in $branch" >&2; exit 22
  fi
  git -C "$wt" add -f -- "evidence/$STAMP"
  git -C "$wt" commit -q -m 'Add completed Shenzhen run evidence'
  HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 \
    timeout 60s git -C "$wt" -c http.version=HTTP/1.1 push -u personal "$branch"
  head=$(git -C "$wt" rev-parse HEAD)
  remote=$(git -C "$wt" ls-remote personal "refs/heads/$branch" | awk '{print $1}')
  test "$head" = "$remote"; test -z "$(git -C "$wt" status --porcelain --untracked-files=all)"
  printf 'PUSHED\t%s\t%s\t%s\t%s\n' "$branch" "$head" \
    "$(git -C "$wt" ls-files "evidence/$STAMP" | wc -l)" "$(du -sb "$out" | awk '{print $1}')"
}

stage_push /data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421 \
  codex/sz-rlt-pi0-robotwin-ar 8bbd0216113ec6eca8bbc67e79d9d642d48b8ed1 \
  /data/chenyiteng/results/rlinf-rlt/smoke-stage1-current-ar-2step-20260823 \
  /data/chenyiteng/results/rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823 \
  /data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2 \
  /data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v3 \
  /data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix

stage_push /data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin \
  codex/sz-current-dsrl-pi0-robotwin 4b609178d10d2534f3f972435ad972e4e015c392 \
  /data/chenyiteng/results/rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823 \
  /data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v1 \
  /data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2

stage_push /data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-pure-single-gpu-7d07a421 \
  codex/sz-rlt-dvac-pure-single-gpu b1e01364b01a9f6d6072e2645cd7ba3bdf0df8fd \
  /data/chenyiteng/results/rlinf-rlt/smokes/current-single-gpu-control-phys2-1cycle-20260830-v2 \
  /data/chenyiteng/results/rlinf-rlt/smokes/current-single-gpu-pure04-phys3-1cycle-20260830-v2

stage_push /data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-action-dvac-adv \
  codex/sz-fastwam-action-dvac-adv a6ad77ea9ee9bf0b355251324c4cf88b6e9a47e7 \
  /data/chenyiteng/results/rlinf-shenzhen/fastwam-action-dvac-adv/runs/fastwam-action-dvac-adv-w05to15-smoke2-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v2 \
  /data/chenyiteng/results/rlinf-shenzhen/fastwam-action-dvac-adv/runs/fastwam-action-dvac-adv-w05to15-smoke2-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v2-reload-step2

echo BATCH4_COMPLETE
