#!/usr/bin/env bash
set -u
url=/api/models/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle/revision/fa8df6ed103db0f5549c122f3a17c00ba6426c98
source /etc/profile.d/mihomo-proxy.sh 2>/dev/null || true
printf 'proxy_official '; curl -L -sS -o /dev/null -w 'http=%{http_code} time=%{time_total}\n' --connect-timeout 8 --max-time 20 "https://huggingface.co$url" || echo failed
printf 'proxy_mirror '; curl -L -sS -o /dev/null -w 'http=%{http_code} time=%{time_total}\n' --connect-timeout 8 --max-time 20 "https://hf-mirror.com$url" || echo failed
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
printf 'direct_official '; curl -L -sS -o /dev/null -w 'http=%{http_code} time=%{time_total}\n' --connect-timeout 8 --max-time 20 "https://huggingface.co$url" || echo failed
printf 'direct_mirror '; curl -L -sS -o /dev/null -w 'http=%{http_code} time=%{time_total}\n' --connect-timeout 8 --max-time 20 "https://hf-mirror.com$url" || echo failed
