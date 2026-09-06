#!/usr/bin/env bash
set -euo pipefail

target=/root/autodl-tmp/datasets/robotwin2/source/9dc9299c163db059931898a9f0852098a61155a1/dataset/adjust_bottle/aloha-agilex_clean_50.zip
listing=/root/autodl-tmp/tmp/rlt_clean50_archive_listing_20260729.txt

unzip -Z1 "$target" >"$listing"
echo "INSPECT_TIME $(date -Is)"
echo "listing=$listing"
echo "entries=$(wc -l <"$listing")"

unsafe_count=$(awk '
  /^\/|^[A-Za-z]:[\\\/]|(^|\/)\.\.(\/|$)|\\/ {count++}
  END {print count + 0}
' "$listing")
echo "unsafe_archive_paths=$unsafe_count"
if [[ "$unsafe_count" -ne 0 ]]; then
  echo "FAIL: archive contains absolute, parent-traversal, drive, or backslash paths" >&2
  awk '/^\/|^[A-Za-z]:[\\\/]|(^|\/)\.\.(\/|$)|\\/' "$listing" >&2
  exit 50
fi

echo "=== directory_prefix_counts ==="
awk -F/ '
  NF >= 2 {print $1 "/" $2}
' "$listing" | sort | uniq -c | sort -nr

echo "=== extension_counts ==="
awk '
  /\/$/ {next}
  {
    n=split($0, parts, ".");
    if (n == 1) {ext="<none>"} else {ext=tolower(parts[n])}
    count[ext]++
  }
  END {
    for (ext in count) {
      print count[ext], ext
    }
  }
' "$listing" | sort -nr

echo "=== file_samples_by_group ==="
for pattern in \
  '/_traj_data/' \
  '/data/' \
  '/instructions/' \
  '/camera'
do
  echo "--- $pattern"
  grep -F "$pattern" "$listing" | awk 'NR <= 8 {print}'
done
