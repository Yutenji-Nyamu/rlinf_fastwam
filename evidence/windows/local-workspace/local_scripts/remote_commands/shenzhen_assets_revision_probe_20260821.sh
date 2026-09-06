set -euo pipefail

root=/data/chenyiteng/projects/robotwin-native/RoboTwin

cd "$root"
test "$(git rev-parse HEAD)" = 30954692d06ba7e89f7a6b76064f4062c488fa81
printf '%s\n' '=== Hugging Face local-dir metadata ==='
while IFS= read -r metadata; do
  printf '%s\n' "--- $metadata"
  sed -n '1,8p' "$metadata"
done < <(find assets/.cache/huggingface/download -maxdepth 1 -type f -name '*.metadata' | sort)

printf '%s\n' '=== final assets and Git status ==='
du -sh assets assets/background_texture assets/embodiments assets/objects
git status --short --branch
