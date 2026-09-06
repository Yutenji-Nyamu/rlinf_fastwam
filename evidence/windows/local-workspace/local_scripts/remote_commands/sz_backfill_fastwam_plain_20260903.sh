#!/usr/bin/env bash
set -euo pipefail
wt=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
branch=codex/sz-fastwam-current-rlinf-grpo
expected=7b2331c55d14397cfb4cb16181470ddc8afae44a
root=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
out="$wt/evidence/completed_runs_20260822_20260903"

copy_run() {
  local run=$1 dest="$out/$(basename "$1")" file rel packet
  test -d "$run"; mkdir -p "$dest/run"
  while IFS= read -r -d '' file; do
    rel=${file#"$run"/}; mkdir -p "$dest/run/$(dirname "$rel")"; cp -- "$file" "$dest/run/$rel"
  done < <(find "$run" \
    -path '*/checkpoints/*' -prune -o -path '*/video/*' -prune -o -path '*/videos/*' -prune -o \
    -path '*/robotwin_data/*' -prune -o -path '*/ray_logs*' -prune -o \
    -type f -size -20M \
    \( -name '*.log' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \
       -o -name '*.txt' -o -name '*.jsonl' -o -name '*.sh' -o -name 'events.out.tfevents*' -o -name '*.png' \) -print0)
  packet="$root/packets/$(basename "$run")"
  if test -d "$packet"; then
    mkdir -p "$dest/packet"
    while IFS= read -r -d '' file; do
      rel=${file#"$packet"/}; mkdir -p "$dest/packet/$(dirname "$rel")"; cp -- "$file" "$dest/packet/$rel"
    done < <(find "$packet" -type f -size -2M \
      \( -name '*.log' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \
         -o -name '*.txt' -o -name '*.jsonl' -o -name '*.sh' \) -print0)
  fi
  printf '%s\n' "$run" > "$dest/SERVER_RUN_PATH.txt"
  (cd "$dest" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
}

test "$(git -C "$wt" branch --show-current)" = "$branch"
test "$(git -C "$wt" rev-parse HEAD)" = "$expected"
test -z "$(git -C "$wt" status --porcelain --untracked-files=all)"
test ! -e "$out"; mkdir -p "$out"
printf '# Completed Fast-WAM Shenzhen run evidence\n\nOnly ended runs are included. The active renderer-lifecycle resume is deliberately excluded. Checkpoints, models, videos, simulator data and full Ray logs are excluded.\n' > "$out/README.md"
for name in \
  fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v1 \
  fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v2-envoffload \
  fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v2-envoffload-reloadcheck \
  fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v3-roundtrip \
  fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v4-roundtrip \
  fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v4-roundtrip-reloadcheck \
  fastwam-grpo-pi0style-offload-smoke1-2gpu32x8-g8-b2048-u2-m10-phys67-v1 \
  fastwam-grpo-pi0style-offload-smoke1-2gpu16x16-g8-b2048-u2-m10-phys67-v1 \
  fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1 \
  fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v1 \
  fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2 \
  fastwam-grpo-control-formal100-2gpu16x16-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-pi0style-v3; do
  copy_run "$root/runs/$name"
done

test -z "$(find "$out" -type f \( -iname '*.pt' -o -iname '*.pth' -o -iname '*.bin' -o -iname '*.safetensors' \
  -o -iname '*.ckpt' -o -iname '*.mp4' -o -iname '*.npz' -o -iname '*.npy' -o -iname '*.h5' \
  -o -iname '*.hdf5' -o -iname '*.zip' -o -iname '*.tar.gz' \) -print -quit)"
if grep -ERil --binary-files=without-match -e '[REDACTED]' -e 'github_pat_' \
    -e 'hf_[A-Za-z0-9]\{20,\}' -e 'authorization:[[:space:]]*bearer' \
    -e 'BEGIN \(RSA\|OPENSSH\|EC\) PRIVATE KEY' "$out" | grep -q .; then
  echo credential-like-material >&2; exit 22
fi
git -C "$wt" add -f -- evidence/completed_runs_20260822_20260903
git -C "$wt" commit -q -m 'Add completed Shenzhen run evidence'
HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 \
  timeout 60s git -C "$wt" -c http.version=HTTP/1.1 push -u personal "$branch"
head=$(git -C "$wt" rev-parse HEAD); remote=$(git -C "$wt" ls-remote personal "refs/heads/$branch" | awk '{print $1}')
test "$head" = "$remote"; test -z "$(git -C "$wt" status --porcelain --untracked-files=all)"
printf 'PUSHED\t%s\t%s\t%s\t%s\n' "$branch" "$head" \
  "$(git -C "$wt" ls-files evidence/completed_runs_20260822_20260903 | wc -l)" "$(du -sb "$out" | awk '{print $1}')"
