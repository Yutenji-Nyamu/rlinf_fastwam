#!/usr/bin/env bash
set -u

printf '%s\n' '--- hostname -I ---'
hostname -I || true
printf '%s\n' '--- interfaces ---'
ip -br -4 addr show || true
printf '%s\n' '--- routes ---'
ip -4 route show || true
printf '%s\n' '--- route to public self ---'
ip -4 route get 120.241.223.9 || true
printf '%s\n' '--- ray processes ---'
pgrep -u "$(id -u)" -af 'raylet|gcs_server|dashboard|monitor.py' || true
printf '%s\n' '--- listeners ---'
ss -ltnp 2>/dev/null | grep -E ':(6389|8266)[[:space:]]' || true
printf '%s\n' '--- gpu compute ---'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader,nounits || true
