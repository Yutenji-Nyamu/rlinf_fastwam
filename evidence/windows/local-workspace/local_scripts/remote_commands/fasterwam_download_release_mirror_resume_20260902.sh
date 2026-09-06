set -euo pipefail

REVISION=6bf9471ced6919a15ab8fded89f7772f5060c44b
TARGET=/data/chenyiteng/models/fasterwam/release-6bf9471
CKPT="$TARGET/robotwin/step_029355.pt"
STATS="$TARGET/robotwin/dataset_stats.json"
POINTER_URL="https://hf-mirror.com/hustvl/FasterWAM/raw/$REVISION/robotwin/step_029355.pt"
FILE_URL="https://hf-mirror.com/hustvl/FasterWAM/resolve/$REVISION/robotwin/step_029355.pt"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
test ! -e "$CKPT"
test -s "$STATS"

pointer="$(curl -L --fail --silent --show-error --max-time 30 "$POINTER_URL")"
printf '%s\n' "$pointer"
expected_sha="$(printf '%s\n' "$pointer" | awk -F: '/^oid sha256:/{print $2}')"
expected_size="$(printf '%s\n' "$pointer" | awk '/^size /{print $2}')"
test -n "$expected_sha"
test -n "$expected_size"

mapfile -t partials < <(find "$TARGET/.cache/huggingface/download/robotwin" -maxdepth 1 -type f -name "*.$expected_sha.incomplete" -print)
test "${#partials[@]}" -eq 1
partial="${partials[0]}"
current_size="$(stat -c %s "$partial")"
test "$current_size" -gt 0
test "$current_size" -lt "$expected_size"
printf 'RESUME_FROM %s / %s %s\n' "$current_size" "$expected_size" "$partial"

download_ok=0
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    printf 'MIRROR_ATTEMPT %s\n' "$attempt"
    if wget \
        --continue \
        --tries=5 \
        --timeout=45 \
        --read-timeout=45 \
        --waitretry=2 \
        --no-use-server-timestamps \
        --progress=dot:giga \
        --output-document="$partial" \
        "$FILE_URL"; then
        download_ok=1
        break
    fi
    sleep 5
done
test "$download_ok" = 1
test "$(stat -c %s "$partial")" = "$expected_size"
actual_sha="$(sha256sum "$partial" | awk '{print $1}')"
test "$actual_sha" = "$expected_sha"

mv -- "$partial" "$CKPT"
printf 'RELEASE_FILES\n'
stat -c '%s %n' "$CKPT" "$STATS"
sha256sum "$CKPT" "$STATS"
printf 'RELEASE_TOTAL\n'
du -sh "$TARGET"
