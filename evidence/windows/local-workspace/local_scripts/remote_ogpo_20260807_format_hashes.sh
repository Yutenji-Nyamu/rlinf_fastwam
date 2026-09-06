#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
cd "$repo"
sha256sum \
  rlinf/data/ogpo_replay.py \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/models/embodiment/openpi/openpi_ogpo.py
