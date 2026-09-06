set -u
target='/data/chenyiteng/models/fasterwam/release-6bf9471'
repo='/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official'
for i in $(seq 1 20); do
  ts=$(date --iso-8601=seconds)
  dl_alive=0; ps -p 46237 >/dev/null 2>&1 && dl_alive=1
  install_alive=0; ps -p 46226 >/dev/null 2>&1 && install_alive=1
  uv_alive=0; pgrep -P 46238 -f '/uv sync ' >/dev/null 2>&1 && uv_alive=1
  inc=$(find "$target" -maxdepth 5 -type f -name '*.incomplete' -printf '%s' 2>/dev/null | head -n 1)
  [ -n "$inc" ] || inc=0
  final=0; [ -f "$target/robotwin/step_029355.pt" ] && final=$(stat -c %s "$target/robotwin/step_029355.pt")
  vbytes=0; [ -d "$repo/.venvs/robotwin" ] && vbytes=$(du -sb "$repo/.venvs/robotwin" 2>/dev/null | awk '{print $1}')
  gpu=$(nvidia-smi -i 3 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  printf '%s i=%s dl_alive=%s inc=%s final=%s install_alive=%s uv_alive=%s venv_bytes=%s gpu3=%s\n' "$ts" "$i" "$dl_alive" "$inc" "$final" "$install_alive" "$uv_alive" "$vbytes" "$gpu"
  if [ "$dl_alive" -eq 0 ] || [ "$install_alive" -eq 0 ]; then
    echo '--- transition detail ---'
    ps -p 46226,46237,46238,46247 -o user=,pid=,ppid=,etime=,%cpu=,%mem=,rss=,stat=,cmd= 2>/dev/null || true
    find "$target" -maxdepth 5 -type f \( -name '*.incomplete' -o -name '*.lock' -o -name 'step_029355.pt' -o -name 'dataset_stats.json' \) -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null | sort -k3
    if [ -x "$repo/.venvs/robotwin/bin/python" ]; then
      "$repo/.venvs/robotwin/bin/python" -c 'import importlib.util as u; print("torch",bool(u.find_spec("torch")),"hydra",bool(u.find_spec("hydra")),"sapien",bool(u.find_spec("sapien")))' 2>/dev/null || true
    fi
    for name in background_texture embodiments objects; do
      p="$repo/third_party/RoboTwin/assets/$name"
      [ -L "$p" ] && printf '%s -> %s\n' "$p" "$(readlink -f "$p")"
    done
  fi
  if [ "$dl_alive" -eq 0 ] && [ "$install_alive" -eq 0 ]; then
    break
  fi
  sleep 45
done

