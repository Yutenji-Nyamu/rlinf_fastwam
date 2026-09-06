#!/usr/bin/env bash
set -euo pipefail

results=/data/chenyiteng/results
bundle_dir=$results/rlinf-rlt/evidence-bundles
bundle=$bundle_dir/shenzhen-current-rlt-dsrl-smokes-light-20260823.tar.gz

files=(
  rlinf-rlt/smoke-stage1-current-ar-2step-20260823/runtime/driver.log
  rlinf-rlt/smoke-stage1-current-ar-2step-20260823/runtime/resource.csv
  rlinf-rlt/smoke-stage1-current-ar-2step-20260823/runtime/resolved.yaml
  rlinf-rlt/smoke-stage1-current-ar-2step-20260823/runtime/resolved.yaml.sha256
  rlinf-rlt/smoke-stage1-current-ar-2step-20260823/runtime/command.txt
  rlinf-rlt/smoke-stage1-current-ar-2step-20260823/runtime/launch_manifest.txt
  rlinf-rlt/smoke-stage1-current-ar-2step-20260823/runtime/exit_code.txt
  rlinf-rlt/smoke-stage1-current-ar-2step-20260823/runtime/started_at.txt
  rlinf-rlt/smoke-stage1-current-ar-2step-20260823/runtime/finished_at.txt
  rlinf-rlt/smoke-stage1-current-ar-2step-20260823/artifacts/stage1_artifact_manifest.json
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/fresh/driver.log
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/fresh/resource.csv
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/fresh/resolved.yaml
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/fresh/resolved.yaml.sha256
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/fresh/command.txt
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/fresh/launch_manifest.txt
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/fresh/exit_code.txt
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/fresh/started_at.txt
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/fresh/finished_at.txt
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/resume/driver.log
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/resume/resource.csv
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/resume/resolved.yaml
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/resume/resolved.yaml.sha256
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/resume/command.txt
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/resume/launch_manifest.txt
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/resume/preflight.txt
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/resume/exit_code.txt
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/resume/started_at.txt
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/resume/finished_at.txt
  rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823/postflight.json
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/fresh/driver.log
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/fresh/resource.csv
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/fresh/resolved.yaml
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/fresh/resolved.yaml.sha256
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/fresh/command.txt
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/fresh/launch_manifest.txt
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/fresh/exit_code.txt
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/fresh/started_at.txt
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/fresh/finished_at.txt
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/resume/driver.log
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/resume/resource.csv
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/resume/resolved.yaml
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/resume/resolved.yaml.sha256
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/resume/command.txt
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/resume/launch_manifest.txt
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/resume/exit_code.txt
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/resume/started_at.txt
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/resume/finished_at.txt
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/postflight.json
  rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823/resource_summary.json
)

for file in "${files[@]}"; do
  test -f "$results/$file"
done
mkdir -p "$bundle_dir"
test ! -e "$bundle"
tar -czf "$bundle" -C "$results" "${files[@]}"
sha256sum "$bundle"
du -h "$bundle"
tar -tzf "$bundle" | wc -l
printf 'bundle=%s\n' "$bundle"
