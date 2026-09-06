#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
upload=/root/autodl-tmp/tmp/rlt_formal250_upload_20260730
cd "$repo"

expected_head=2df23e7f4b3d19d4f0dedab32168767a32845a58
expected_test_sha=12d2f5a4e6d4acb70809412b38d4de28ce367f724b6bc2fb3d7b9c02d8f9a7e0
expected_base_sha=f089f333839c99b87d546e8bcf0d5bddbb7da380e8cc1597e1de4c4450592850

test_path=tests/unit_tests/test_robotwin_seed_partition.py
base_path=examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml
bank_path=rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json
formal_path=examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml
smoke_path=examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_resource_smoke.yaml

test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --porcelain=v1)"
test "$(sha256sum "$test_path" | awk '{print $1}')" = "$expected_test_sha"
test "$(sha256sum "$base_path" | awk '{print $1}')" = "$expected_base_sha"
test ! -e "$bank_path"
test ! -e "$formal_path"
test ! -e "$smoke_path"

test "$(sha256sum "$upload/$bank_path" | awk '{print $1}')" = \
  fb9c3353e27b83aad6fe7ff778437d960b084de9d981c2af68615d52769952a7
test "$(sha256sum "$upload/$formal_path" | awk '{print $1}')" = \
  d9ee30f8c776b349cc3e5e08cb17c97f46151c9c62bffada7f514e9522ffb315
test "$(sha256sum "$upload/$smoke_path" | awk '{print $1}')" = \
  47bbdd231d53d69aad5aee35ef6c04787d6d9c7610f02df9f945961466d6acb2
test "$(sha256sum "$upload/$test_path" | awk '{print $1}')" = \
  e4b5686e5d4bc68961cf7238d7b42be96f724e6bc948c530e2ddb4f4c8d3c093

install -m 0644 "$upload/$bank_path" "$bank_path"
install -m 0644 "$upload/$formal_path" "$formal_path"
install -m 0644 "$upload/$smoke_path" "$smoke_path"
install -m 0644 "$upload/$test_path" "$test_path"

git diff --check
git status --short
sha256sum "$bank_path" "$formal_path" "$smoke_path" "$test_path"
