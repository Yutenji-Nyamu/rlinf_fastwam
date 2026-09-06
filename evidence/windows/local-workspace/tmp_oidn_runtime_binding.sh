set -eu
for pid in 2813140 2813142; do
  test -d "/proc/$pid"
  printf 'PID=%s\n' "$pid"
  tr '\0' '\n' < "/proc/$pid/environ" | grep -E '^(PYTHONPATH|ROBOTWIN_PATH|RLINF_CODE_WORKING_DIR|CUDA_VISIBLE_DEVICES)=' | sort
done
