#!/usr/bin/env bash
set -u
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
find "$ROBOTWIN" -type f -iname '*vector*env*.py' -printf '%p\n'
grep -RIn -C 16 -E 'trial_seed|except UnStableError' "$ROBOTWIN" --include='*.py' 2>/dev/null
