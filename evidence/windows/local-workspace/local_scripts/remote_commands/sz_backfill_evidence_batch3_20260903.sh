#!/usr/bin/env bash
set -euo pipefail
RESULT=/data/chenyiteng/results/rlinf-shenzhen
STAMP=completed_runs_20260822_20260903

copy_run() {
  local run=$1 out=$2 file rel root packet
  test -d "$run"
  mkdir -p "$out/run"
  while IFS= read -r -d '' file; do
    rel=${file#"$run"/}
    mkdir -p "$out/run/$(dirname "$rel")"
    cp -- "$file" "$out/run/$rel"
  done < <(find "$run" \
    -path '*/checkpoints/*' -prune -o -path '*/video/*' -prune -o -path '*/videos/*' -prune -o \
    -path '*/robotwin_data/*' -prune -o -path '*/ray_logs*' -prune -o \
    -type f -size -20M \
    \( -name '*.log' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \
       -o -name '*.txt' -o -name '*.jsonl' -o -name '*.sh' -o -name 'events.out.tfevents*' -o -name '*.png' \) -print0)
  case "$run" in
    */runs/*)
      root=${run%%/runs/*}
      packet="$root/packets/$(basename "$run")"
      if test -d "$packet"; then
        mkdir -p "$out/packet"
        while IFS= read -r -d '' file; do
          rel=${file#"$packet"/}
          mkdir -p "$out/packet/$(dirname "$rel")"
          cp -- "$file" "$out/packet/$rel"
        done < <(find "$packet" -type f -size -2M \
          \( -name '*.log' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \
             -o -name '*.txt' -o -name '*.jsonl' -o -name '*.sh' \) -print0)
      fi
      ;;
  esac
  printf '%s\n' "$run" > "$out/SERVER_RUN_PATH.txt"
  (cd "$out" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
}

stage_push() {
  local wt=$1 branch=$2 expected=$3; shift 3
  local out="$wt/evidence/$STAMP" run head remote
  test "$(git -C "$wt" branch --show-current)" = "$branch"
  test "$(git -C "$wt" rev-parse HEAD)" = "$expected"
  test -z "$(git -C "$wt" status --porcelain --untracked-files=all)"
  test ! -e "$out"
  mkdir -p "$out"
  printf '# Completed Shenzhen run evidence\n\nCopied from the immutable server paths named in each `SERVER_RUN_PATH.txt`. Checkpoints, models, videos, simulator data and full Ray logs are excluded.\n' > "$out/README.md"
  for run in "$@"; do copy_run "$run" "$out/$(basename "$run")"; done
  test -z "$(find "$out" -type f \( -iname '*.pt' -o -iname '*.pth' -o -iname '*.bin' -o -iname '*.safetensors' \
    -o -iname '*.ckpt' -o -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.npz' \
    -o -iname '*.npy' -o -iname '*.h5' -o -iname '*.hdf5' -o -iname '*.zip' -o -iname '*.tar.gz' \) -print -quit)"
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
  test "$head" = "$remote"
  test -z "$(git -C "$wt" status --porcelain --untracked-files=all)"
  printf 'PUSHED\t%s\t%s\t%s\t%s\n' "$branch" "$head" \
    "$(git -C "$wt" ls-files "evidence/$STAMP" | wc -l)" "$(du -sb "$out" | awk '{print $1}')"
}

stage_push /data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv \
  codex/sz-grpo-dvac-action-adv a5b94b6f10a9212502d6930f07543f61e31af52e \
  "$RESULT/grpo/runs/dvac-action-adv-w0to2-smoke2-2gpu64x4-b1024-noeval-phys23-v1" \
  "$RESULT/grpo/runs/dvac-action-adv-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1"

stage_push /data/chenyiteng/projects/rlinf-shenzhen/worktrees/st-dvac-local-shard \
  codex/sz-st-dvac-local-shard f2a543da87afb7d5e3aa030ef52e53df9a266b28 \
  "$RESULT/grpo/runs/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1" \
  "$RESULT/grpo/runs/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v2"

stage_push /data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin \
  codex/sz-ppo-pi0-robotwin 7d07a4212ee6858cc333e1d4fab7a37256d1f839 \
  "$RESULT/ppo/ppo-formal100-4gpu128train64eval-official-v1"

echo BATCH3_COMPLETE
