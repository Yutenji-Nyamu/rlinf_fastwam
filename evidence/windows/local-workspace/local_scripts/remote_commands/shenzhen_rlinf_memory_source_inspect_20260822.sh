#!/usr/bin/env bash
set -u

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

echo '=== RECORDVIDEO_DEFINITION ==='
grep -R -n --include='*.py' 'class RecordVideo' "$ROOT/rlinf" 2>/dev/null || true

echo '=== RECORDVIDEO_FLUSH_REFERENCES ==='
grep -R -n --include='*.py' -E 'def flush_video|flush_video\(|video_frames|frames\.(append|clear)|render_images' "$ROOT/rlinf" 2>/dev/null | head -n 240

echo '=== ROBOTWIN_CACHE_REFERENCES ==='
grep -R -n --include='*.py' -E 'clear_cache_freq|eval_video_log|collect_data|clear_cache|gc\.collect|empty_cache|video' "$ROBOTWIN" 2>/dev/null | head -n 300

echo '=== RECORDVIDEO_SOURCE ==='
record_file=$(grep -R -l --include='*.py' 'class RecordVideo' "$ROOT/rlinf" 2>/dev/null | head -n 1)
if test -n "$record_file"; then
  echo "file=$record_file"
  sed -n '1,300p' "$record_file"
fi

echo '=== ROBOTWIN_ENV_SOURCE_CANDIDATES ==='
find "$ROOT/rlinf" "$ROBOTWIN" -type f \( -name '*robotwin*.py' -o -name 'vector_env.py' -o -name '*wrapper*.py' \) -print 2>/dev/null | head -n 120

echo 'MEMORY_SOURCE_INSPECT_DONE'
