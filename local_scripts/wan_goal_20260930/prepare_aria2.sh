#!/bin/bash
set -euo pipefail
P=/data/chenyiteng/projects/wan-goal-sz3/tools/aria2
mkdir -p "$P/debs" "$P/root"
cd "$P/debs"
apt-get download aria2=1.36.0-1 libaria2-0=1.36.0-1 libssh2-1=1.10.0-3ubuntu0.1 libc-ares2=1.18.1-1ubuntu0.22.04.3
for f in ./*.deb; do dpkg-deb -x "$f" "$P/root"; done
LD_LIBRARY_PATH="$P/root/usr/lib/x86_64-linux-gnu" "$P/root/usr/bin/aria2c" --version | head -8
sha256sum ./*.deb
