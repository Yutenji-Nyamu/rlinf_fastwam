#!/usr/bin/env bash
set -u
date --iso-8601=seconds
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
printf 'memory.peak='; cat /sys/fs/cgroup/memory.peak
printf 'memory.max='; cat /sys/fs/cgroup/memory.max
printf 'memory.swap.current='; cat /sys/fs/cgroup/memory.swap.current
cat /sys/fs/cgroup/memory.events
df -h /root/autodl-tmp
