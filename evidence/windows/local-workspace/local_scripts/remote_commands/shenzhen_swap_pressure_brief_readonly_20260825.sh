#!/usr/bin/env bash
set -u
date --iso-8601=seconds
free -h
vmstat 1 3
for f in cpu memory io; do printf '%s_pressure=' "$f"; tr '\n' ' ' <"/proc/pressure/$f"; echo; done
