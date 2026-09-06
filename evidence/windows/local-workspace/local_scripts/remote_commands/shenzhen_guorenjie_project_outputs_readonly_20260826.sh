#!/usr/bin/env bash
set -euo pipefail

sudo -S -p '' true
sudo -n -u guorenjie bash <<'INNER'
set -euo pipefail
export HOME=/home/guorenjie
export GIT_OPTIONAL_LOCKS=0
cd /home/guorenjie
PROJ=/home/guorenjie/research/smolvla-libero-clp
REPO="$PROJ/lerobot"

echo '===== git ====='
git -C "$REPO" rev-parse --show-toplevel
printf 'head='; git -C "$REPO" rev-parse HEAD
printf 'branch='; git -C "$REPO" branch --show-current
echo '-- remotes (names only) --'; git -C "$REPO" remote || true
echo '-- recent commits --'; git -C "$REPO" log -8 --date=iso --pretty='format:%h %ad %an %s'
echo; echo '-- status --'; git -C "$REPO" status --short --untracked-files=normal || true
echo '-- diff stat --'; git -C "$REPO" diff --stat || true

echo '===== project_sizes ====='
du -sh "$PROJ" "$REPO" "$PROJ/refs" "$PROJ/configs" "$PROJ/scripts" "$PROJ/outputs" 2>/dev/null || true
echo '-- output roots --'
du -sh "$PROJ"/outputs/* 2>/dev/null | sort -h || true

echo '===== active_output_dirs ====='
for out in \
  "$PROJ/outputs/train/pi0_libero_fullft_rel_vis_3m_pure_s30k" \
  "$PROJ/outputs/train/pi0_libero_fullft_rel_vis_3m_v2_default_s30k" \
  "$PROJ/outputs/train/pi0_libero_fullft_rel_vis_paper_s30k"; do
  echo "-- $out --"
  if [[ ! -d "$out" ]]; then echo missing; continue; fi
  stat -c '%A %U:%G %s %y %n' "$out"
  du -sh "$out"
  find "$out" -maxdepth 4 -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort -k2 | tail -n 120
done

echo '===== prior_result_candidates ====='
find "$PROJ/outputs" -type f \
  \( -iname '*.json' -o -iname '*.jsonl' -o -iname '*.csv' -o -iname '*.md' -o -iname '*.log' -o -iname '*.txt' \) \
  -size -4M -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort -k2 | tail -n 260

echo '===== dataset_locations ====='
for root in /home/guorenjie/.cache/huggingface /home/guorenjie/.cache /home/guorenjie/research/smolvla-libero-clp/outputs/cache; do
  [[ -e "$root" ]] && du -sh "$root" 2>/dev/null || true
done
find /home/guorenjie/.cache/huggingface "$PROJ/outputs/cache" -maxdepth 8 \
  \( -iname '*libero*' -o -iname 'info.json' -o -iname 'stats.json' -o -iname 'tasks.jsonl' -o -iname 'episodes.jsonl' \) \
  -printf '%y %s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort | head -n 320 || true

echo '===== relevant_code_hits ====='
grep -RIn --exclude-dir=.git --exclude-dir=outputs --exclude='*.pyc' \
  -E 'clp_remove_indices|relative_exclude_joints|CLP|layer pruning|redundan|similarity|CKA' \
  "$PROJ/configs" "$PROJ/scripts" "$REPO/src" 2>/dev/null | head -n 260 || true

echo SZ_GUORENJIE_PROJECT_OUTPUTS_READONLY_OK
INNER
