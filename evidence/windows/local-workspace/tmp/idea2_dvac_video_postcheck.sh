set -euo pipefail

output=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin/outputs/idea2_dvac_sft_smoke_2gpu_16env_v1
video_root="$output/video/eval"
probe_dir=/root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1/video_probe_v3
test ! -e "$probe_dir"
mkdir -p "$probe_dir"

video_count=$(find "$video_root" -type f -name '*.mp4' | wc -l)
test "$video_count" -eq 2
printf 'VIDEO_COUNT=%s\n' "$video_count"
while IFS= read -r video; do
  seed_dir=$(basename "$(dirname "$video")")
  printf 'VIDEO_FILE=%s\n' "$video"
  ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration:format=duration,size \
    -of json "$video"
  ffmpeg -nostdin -v error -i "$video" -vf 'tile=3x2:padding=4:margin=4' -frames:v 1 \
    "$probe_dir/${seed_dir}_contact.png"
  ffmpeg -nostdin -v error -i "$video" -vsync 0 \
    "$probe_dir/${seed_dir}_frame_%02d.png"
done < <(find "$video_root" -type f -name '*.mp4' | sort)

printf 'EXTRACTED_FRAME_COUNT=%s\n' "$(find "$probe_dir" -type f -name '*_frame_*.png' | wc -l)"
printf 'CONTACT_SHEET_COUNT=%s\n' "$(find "$probe_dir" -type f -name '*_contact.png' | wc -l)"
find "$probe_dir" -type f -printf '%f\t%s\n' | sort
printf 'VIDEO_POSTCHECK_PASS=1\n'
