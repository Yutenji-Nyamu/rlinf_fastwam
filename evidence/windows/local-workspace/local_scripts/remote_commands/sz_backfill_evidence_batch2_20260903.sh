#!/usr/bin/env bash
set -euo pipefail
RESULT=/data/chenyiteng/results/rlinf-shenzhen
STAMP=completed_runs_20260822_20260903

copy_run() {
  local run=$1 out=$2 file rel packet
  test -d "$run"; mkdir -p "$out/run"
  while IFS= read -r -d '' file; do
    rel=${file#"$run"/}; mkdir -p "$out/run/$(dirname "$rel")"; cp -- "$file" "$out/run/$rel"
  done < <(find "$run" \
    -path '*/checkpoints/*' -prune -o -path '*/video/*' -prune -o -path '*/videos/*' -prune -o \
    -path '*/robotwin_data/*' -prune -o -path '*/ray_logs*' -prune -o \
    -type f -size -20M \
    \( -name '*.log' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \
       -o -name '*.txt' -o -name '*.jsonl' -o -name 'events.out.tfevents*' -o -name '*.png' \) -print0)
  packet="$(dirname "$(dirname "$run")")/packets/$(basename "$run")"
  if test -d "$packet"; then
    mkdir -p "$out/packet"
    while IFS= read -r -d '' file; do
      rel=${file#"$packet"/}; mkdir -p "$out/packet/$(dirname "$rel")"; cp -- "$file" "$out/packet/$rel"
    done < <(find "$packet" -type f -size -2M \
      \( -name '*.log' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \
         -o -name '*.txt' -o -name '*.jsonl' \) -print0)
  fi
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
  printf '# Completed Shenzhen run evidence\n\nCopied from the server run directories named in each `SERVER_RUN_PATH.txt`; checkpoints, models, videos, simulator data and full Ray logs are excluded.\n' > "$out/README.md"
  for run in "$@"; do copy_run "$run" "$out/$(basename "$run")"; done
  test -z "$(find "$out" -type f \( -iname '*.pt' -o -iname '*.pth' -o -iname '*.bin' -o -iname '*.safetensors' -o -iname '*.ckpt' -o -iname '*.mp4' -o -iname '*.npz' -o -iname '*.npy' -o -iname '*.h5' -o -iname '*.hdf5' -o -iname '*.zip' \) -print -quit)"
  if grep -ERil --binary-files=without-match -e '[REDACTED]' -e 'password[[:space:]]*[:=]' -e 'authorization:[[:space:]]*bearer' -e 'hf_[A-Za-z0-9]\{20,\}' -e 'github_pat_' "$out" | grep -q .; then
    echo "credential-like material in $branch" >&2; exit 22
  fi
  git -C "$wt" add -f -- "evidence/$STAMP"
  git -C "$wt" commit -q -m 'Add completed Shenzhen run evidence'
  git -C "$wt" push --quiet personal "$branch"
  head=$(git -C "$wt" rev-parse HEAD); remote=$(git -C "$wt" ls-remote personal "refs/heads/$branch" | awk '{print $1}')
  test "$head" = "$remote"; test -z "$(git -C "$wt" status --porcelain --untracked-files=all)"
  printf 'PUSHED\t%s\t%s\t%s\t%s\n' "$branch" "$head" "$(git -C "$wt" ls-files "evidence/$STAMP" | wc -l)" "$(du -sb "$out" | awk '{print $1}')"
}

stage_push /data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv-fix \
  codex/sz-grpo-dvac-action-adv-fix e434f409b21d281ce883df29487ecae7cb3e4839 \
  "$RESULT/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2" \
  "$RESULT/grpo/runs/dvac-action-adv-fix-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1" \
  "$RESULT/grpo/runs/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1"

stage_push /data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo \
  codex/sz-prism-dvac-rank-rloo 306ce2e98a06b6f439a1070d8942e20132e48d49 \
  "$RESULT/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2" \
  "$RESULT/grpo/runs/prism-dvac-rank-rloo-smoke1-2gpu64x4-b1024-noeval-phys23-v1" \
  "$RESULT/grpo/runs/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1" \
  "$RESULT/grpo/runs/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2"

stage_push /data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-dvac-action-adv-fix \
  codex/sz-ppo-dvac-action-adv-fix 74617ced87d64045ab6850d0efd90956a494af66 \
  "$RESULT/ppo/runs/ppo-control-smoke1-2gpu64x4-b1024-noeval-localshard-phys23-v1" \
  "$RESULT/ppo/runs/ppo-dvac-action-adv-fix-w0to2-smoke2-2gpu64x4-b1024-noeval-localshard-phys23-v1" \
  "$RESULT/ppo/runs/ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1" \
  "$RESULT/ppo/runs/ppo-dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v1"

stage_push /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl \
  codex/sz-pi05-robotwin-rl 256eeeb4459b4bd5db85bfc6a0eb315771e8c38c \
  "$RESULT/pi05/runs/pi05-ppo-smoke1-2gpu64x4-b512-u5-m5-phys23-localshard-v2" \
  "$RESULT/pi05/runs/pi05-grpo-smoke1-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2" \
  "$RESULT/pi05/runs/pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2" \
  "$RESULT/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1" \
  "$RESULT/pi05/runs/pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys67-localshard-v1" \
  "$RESULT/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2" \
  "$RESULT/pi05/runs/pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys67-localshard-v2"

echo BATCH2_COMPLETE
