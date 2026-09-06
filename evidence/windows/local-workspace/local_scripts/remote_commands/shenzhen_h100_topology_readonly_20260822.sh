#!/usr/bin/env bash
set -uo pipefail
printf 'timestamp=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)"
printf '%s\n' '=== exact GPU identity ==='
nvidia-smi --query-gpu=index,name,pci.bus_id,pci.device_id,uuid,serial,vbios_version,driver_version,memory.total --format=csv,noheader,nounits
printf '%s\n' '=== nvidia-smi topo -m ==='
nvidia-smi topo -m
printf '%s\n' '=== NVLink status ==='
nvidia-smi nvlink --status
exit 0
