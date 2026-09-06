set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh

installer_dir=/home/chenyiteng/installers
installer_path="$installer_dir/Miniforge3-26.3.2-3-Linux-x86_64.sh"
install_prefix=/home/chenyiteng/miniforge3
installer_url=https://github.com/conda-forge/miniforge/releases/download/26.3.2-3/Miniforge3-Linux-x86_64.sh
installer_sha256=848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c

printf '%s\n' '=== PRECHECK ==='
id
df -h /home/chenyiteng
if [ -e "$install_prefix" ]; then
  printf 'REFUSE_EXISTING_PREFIX=%s\n' "$install_prefix" >&2
  exit 40
fi
mkdir -p "$installer_dir"

printf '%s\n' '=== DOWNLOAD AND VERIFY ==='
curl --fail --location --retry 5 --retry-all-errors --continue-at - \
  --output "$installer_path" "$installer_url"
printf '%s  %s\n' "$installer_sha256" "$installer_path" | sha256sum --check --strict
stat -c 'installer=%n bytes=%s owner=%U:%G mode=%a' "$installer_path"

printf '%s\n' '=== INSTALL ==='
bash "$installer_path" -b -p "$install_prefix"

printf '%s\n' '=== VERIFY ==='
"$install_prefix/bin/conda" --version
"$install_prefix/bin/conda" info --base
"$install_prefix/bin/python" --version
du -sh "$install_prefix" "$installer_path"
