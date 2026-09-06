set -u

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
RUN_ID=official-move-stapler-pad-random1-20260902-v1
RUN_ROOT=/data/chenyiteng/results/fasterwam-standalone/$RUN_ID
OFFICIAL_ROOT="$REPO/evaluate_results/robotwin/step_029355/$RUN_ID"

date '+TIME %Y-%m-%d %H:%M:%S %Z'

printf 'OWNED_PROCESSES\n'
ps -eo user,pid,ppid,etimes,stat,%cpu,%mem,args | grep -E 'eval_robotwin_single.py|official-move-stapler-pad-random1-20260902-v1' | grep -v grep || true

printf 'GPU3\n'
nvidia-smi -i 3 --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate --format=csv,noheader

printf 'RUN_ROOT\n'
ls -ld "$RUN_ROOT" "$RUN_ROOT/official_output" 2>&1 || true
stat -c '%s %y %n' "$RUN_ROOT/driver.log" 2>&1 || true

printf 'DRIVER_TAIL\n'
tail -n 100 "$RUN_ROOT/driver.log" 2>&1 || true

printf 'OFFICIAL_FILES\n'
find "$OFFICIAL_ROOT" -maxdepth 5 -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort

printf 'RESULT_TEXT\n'
find "$OFFICIAL_ROOT" -type f -name '_result_random.txt' -exec sh -c 'printf "FILE %s\n" "$1"; sed -n "1,120p" "$1"' _ {} \; 2>/dev/null || true

printf 'VIDEO_PROBE\n'
find "$OFFICIAL_ROOT" -type f \( -iname '*.mp4' -o -iname '*.avi' \) -print0 2>/dev/null | while IFS= read -r -d '' video; do
  stat -c 'VIDEO %s %y %n' "$video"
  if command -v ffprobe >/dev/null 2>&1; then
    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,nb_frames,r_frame_rate,duration -of default=noprint_wrappers=1 "$video" 2>&1 | sed 's/^/  /'
  fi
done

printf 'FATAL_SCAN\n'
grep -Ein 'traceback|fatal|cuda out of memory|out of memory|segmentation fault|runtimeerror|exception' "$RUN_ROOT/driver.log" 2>/dev/null | tail -n 40 || true
