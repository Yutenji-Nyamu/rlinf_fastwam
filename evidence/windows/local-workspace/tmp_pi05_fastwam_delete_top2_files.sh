#!/usr/bin/env bash
set -euo pipefail

# Fixed, live-verified targets derived from
# tmp_pi05_fastwam_smoke_checkpoint_targets.tsv on 2026-09-01.
# This script deletes files only. It never removes a directory and never uses a
# glob to select a deletion target.
readarray -t targets <<'TARGETS'
10156334861|/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-ppo-smoke1-2gpu64x4-b512-u5-m5-phys23-localshard-v2/pi05_ppo_smoke1/checkpoints/global_step_1/actor/local_shard_checkpoint/checkpoint_rank_1.pt
10156334797|/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-ppo-smoke1-2gpu64x4-b512-u5-m5-phys23-localshard-v2/pi05_ppo_smoke1/checkpoints/global_step_1/actor/local_shard_checkpoint/checkpoint_rank_0.pt
10150818027|/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-smoke1-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/pi05_grpo_smoke1/checkpoints/global_step_1/actor/local_shard_checkpoint/checkpoint_rank_1.pt
10150818027|/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-smoke1-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/pi05_grpo_smoke1/checkpoints/global_step_1/actor/local_shard_checkpoint/checkpoint_rank_0.pt
10150817963|/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/pi05_grpo_dvac_action_adv_w0p5to1p5_smoke2/checkpoints/global_step_1/actor/local_shard_checkpoint/checkpoint_rank_1.pt
10150817899|/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/pi05_grpo_dvac_action_adv_w0p5to1p5_smoke2/checkpoints/global_step_1/actor/local_shard_checkpoint/checkpoint_rank_0.pt
10150817963|/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/pi05_grpo_dvac_action_adv_w0p5to1p5_smoke2/checkpoints/global_step_2/actor/local_shard_checkpoint/checkpoint_rank_1.pt
10150817899|/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/pi05_grpo_dvac_action_adv_w0p5to1p5_smoke2/checkpoints/global_step_2/actor/local_shard_checkpoint/checkpoint_rank_0.pt
14450986115|/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v2-envoffload/fastwam_grpo_smoke1_current/checkpoints/global_step_1/actor/local_shard_checkpoint/checkpoint_rank_0.pt
14448760195|/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v2-envoffload/fastwam_grpo_smoke1_current/checkpoints/global_step_1/actor/local_shard_checkpoint/checkpoint_rank_1.pt
14455154049|/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v4-roundtrip/fastwam_grpo_smoke1_dcp_fresh/checkpoints/global_step_1/actor/dcp_checkpoint/__0_0.distcp
14454317014|/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v4-roundtrip/fastwam_grpo_smoke1_dcp_fresh/checkpoints/global_step_1/actor/dcp_checkpoint/__1_0.distcp
14455154049|/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v4-roundtrip-reloadcheck/fastwam_grpo_smoke1_dcp_reload_step2/checkpoints/global_step_2/actor/dcp_checkpoint/__0_0.distcp
14454317014|/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v4-roundtrip-reloadcheck/fastwam_grpo_smoke1_dcp_reload_step2/checkpoints/global_step_2/actor/dcp_checkpoint/__1_0.distcp
14455154049|/data/chenyiteng/results/rlinf-shenzhen/fastwam-action-dvac-adv/runs/fastwam-action-dvac-adv-w05to15-smoke2-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v2/fastwam_action_dvac_adv_smoke_fresh/checkpoints/global_step_1/actor/dcp_checkpoint/__0_0.distcp
14454317014|/data/chenyiteng/results/rlinf-shenzhen/fastwam-action-dvac-adv/runs/fastwam-action-dvac-adv-w05to15-smoke2-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v2/fastwam_action_dvac_adv_smoke_fresh/checkpoints/global_step_1/actor/dcp_checkpoint/__1_0.distcp
14455154049|/data/chenyiteng/results/rlinf-shenzhen/fastwam-action-dvac-adv/runs/fastwam-action-dvac-adv-w05to15-smoke2-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v2-reload-step2/fastwam_action_dvac_adv_smoke_reload_step2/checkpoints/global_step_2/actor/dcp_checkpoint/__0_0.distcp
14454317014|/data/chenyiteng/results/rlinf-shenzhen/fastwam-action-dvac-adv/runs/fastwam-action-dvac-adv-w05to15-smoke2-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v2-reload-step2/fastwam_action_dvac_adv_smoke_reload_step2/checkpoints/global_step_2/actor/dcp_checkpoint/__1_0.distcp
TARGETS

approved_roots=(
  '/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-ppo-smoke1-2gpu64x4-b512-u5-m5-phys23-localshard-v2/pi05_ppo_smoke1/checkpoints/global_step_1'
  '/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-smoke1-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/pi05_grpo_smoke1/checkpoints/global_step_1'
  '/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/pi05_grpo_dvac_action_adv_w0p5to1p5_smoke2/checkpoints/global_step_1'
  '/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/pi05_grpo_dvac_action_adv_w0p5to1p5_smoke2/checkpoints/global_step_2'
  '/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v2-envoffload/fastwam_grpo_smoke1_current/checkpoints/global_step_1'
  '/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v4-roundtrip/fastwam_grpo_smoke1_dcp_fresh/checkpoints/global_step_1'
  '/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v4-roundtrip-reloadcheck/fastwam_grpo_smoke1_dcp_reload_step2/checkpoints/global_step_2'
  '/data/chenyiteng/results/rlinf-shenzhen/fastwam-action-dvac-adv/runs/fastwam-action-dvac-adv-w05to15-smoke2-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v2/fastwam_action_dvac_adv_smoke_fresh/checkpoints/global_step_1'
  '/data/chenyiteng/results/rlinf-shenzhen/fastwam-action-dvac-adv/runs/fastwam-action-dvac-adv-w05to15-smoke2-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v2-reload-step2/fastwam_action_dvac_adv_smoke_reload_step2/checkpoints/global_step_2'
)

formal_control='/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1'
formal_dvac='/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys67-localshard-v1'
expected_count=18
expected_bytes=225755207998

formal_alive_and_clean() {
  local run_root=$1
  local label=$2
  local driver_log="$run_root/runtime/driver.log"
  local wrapper_pattern="$run_root/runtime/wrapper.sh"
  local pids
  pids=$(pgrep -f -- "$wrapper_pattern" || true)
  if [[ -z "$pids" ]]; then
    printf 'FORMAL_NOT_ALIVE\t%s\t%s\n' "$label" "$run_root" >&2
    return 1
  fi
  if [[ -f "$driver_log" ]] && grep -Eaiq \
    'Traceback|OutOfMemoryError|CUDA out of memory|WorkerCrashedError|SIGKILL|NCCL[^[:cntrl:]]*(error|timeout)|nonfinite' \
    "$driver_log"; then
    printf 'FORMAL_FATAL_PATTERN\t%s\t%s\n' "$label" "$driver_log" >&2
    return 1
  fi
  printf 'FORMAL_ALIVE\t%s\tpids=%s\tdriver_bytes=%s\n' \
    "$label" "$(tr '\n' ',' <<<"$pids" | sed 's/,$//')" \
    "$(stat -c '%s' "$driver_log" 2>/dev/null || printf 'NA')"
}

[[ ${#targets[@]} -eq $expected_count ]] || {
  printf 'BAD_TARGET_COUNT\tgot=%s\texpected=%s\n' "${#targets[@]}" "$expected_count" >&2
  exit 1
}

declare -A seen=()
checked_bytes=0
for entry in "${targets[@]}"; do
  expected_size=${entry%%|*}
  path=${entry#*|}

  [[ "$expected_size" =~ ^[0-9]+$ ]] || {
    printf 'BAD_SIZE_FIELD\t%s\n' "$entry" >&2
    exit 1
  }
  [[ "$path" == /* ]] || {
    printf 'NOT_ABSOLUTE\t%s\n' "$path" >&2
    exit 1
  }
  [[ -z ${seen["$path"]+x} ]] || {
    printf 'DUPLICATE_TARGET\t%s\n' "$path" >&2
    exit 1
  }
  seen["$path"]=1

  [[ "$path" != "$formal_control" && "$path" != "$formal_control/"* ]] || {
    printf 'FORMAL_TARGET_REJECTED\t%s\n' "$path" >&2
    exit 1
  }
  [[ "$path" != "$formal_dvac" && "$path" != "$formal_dvac/"* ]] || {
    printf 'FORMAL_TARGET_REJECTED\t%s\n' "$path" >&2
    exit 1
  }

  approved=0
  for root in "${approved_roots[@]}"; do
    if [[ "$path" == "$root/"* ]]; then
      approved=1
      break
    fi
  done
  [[ $approved -eq 1 ]] || {
    printf 'UNAPPROVED_ROOT\t%s\n' "$path" >&2
    exit 1
  }

  [[ -f "$path" && ! -L "$path" ]] || {
    printf 'NOT_REGULAR_OR_IS_SYMLINK\t%s\n' "$path" >&2
    exit 1
  }
  canonical=$(realpath -e -- "$path")
  [[ "$canonical" == "$path" ]] || {
    printf 'CANONICAL_PATH_MISMATCH\t%s\t%s\n' "$path" "$canonical" >&2
    exit 1
  }
  actual_size=$(stat -c '%s' -- "$path")
  [[ "$actual_size" == "$expected_size" ]] || {
    printf 'SIZE_MISMATCH\t%s\texpected=%s\tactual=%s\n' "$path" "$expected_size" "$actual_size" >&2
    exit 1
  }
  checked_bytes=$((checked_bytes + actual_size))
done

[[ $checked_bytes -eq $expected_bytes ]] || {
  printf 'TOTAL_BYTES_MISMATCH\texpected=%s\tactual=%s\n' "$expected_bytes" "$checked_bytes" >&2
  exit 1
}

# Both current formal runs must be healthy before any destructive action.
formal_alive_and_clean "$formal_control" control_pre
formal_alive_and_clean "$formal_dvac" dvac_pre

printf 'PRECHECK_OK\ttargets=%s\tbytes=%s\n' "${#targets[@]}" "$checked_bytes"
echo 'DF_BEFORE'
df -B1 --output=source,size,used,avail,pcent,target /data

deleted_count=0
deleted_bytes=0
for entry in "${targets[@]}"; do
  expected_size=${entry%%|*}
  path=${entry#*|}
  rm -- "$path"
  [[ ! -e "$path" ]] || {
    printf 'DELETE_FAILED\t%s\n' "$path" >&2
    exit 1
  }
  deleted_count=$((deleted_count + 1))
  deleted_bytes=$((deleted_bytes + expected_size))
  printf 'DELETED\t%s\t%s\n' "$expected_size" "$path"
done

printf 'DELETE_SUMMARY\tcount=%s\tbytes=%s\n' "$deleted_count" "$deleted_bytes"
echo 'DF_AFTER'
df -B1 --output=source,size,used,avail,pcent,target /data

# Post-delete checks are read-only and target only the two active formal runs.
formal_alive_and_clean "$formal_control" control_post
formal_alive_and_clean "$formal_dvac" dvac_post
printf 'POSTCHECK_OK\tformal_runs=2\tdeleted=%s\tbytes=%s\n' "$deleted_count" "$deleted_bytes"
