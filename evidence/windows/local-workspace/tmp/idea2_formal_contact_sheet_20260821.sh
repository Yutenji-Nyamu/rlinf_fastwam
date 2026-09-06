set -e
run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821
out=/root/autodl-tmp/idea2_dvac_train_analysis/formal_live_step23_20260821
video="$run/control_trace/adjust_bottle/worker_000/env_slot_000/recording_0000_episode_0000_reset_57/head_camera.mp4"
mkdir -p "$out"
ffmpeg -y -v error -i "$video" \
  -vf "select='not(mod(n\,10))',scale=320:240,tile=5x4:padding=4:margin=4" \
  -frames:v 1 "$out/CONTROL_TRACE_CONTACT_SHEET.png"
file "$out/CONTROL_TRACE_CONTACT_SHEET.png"
