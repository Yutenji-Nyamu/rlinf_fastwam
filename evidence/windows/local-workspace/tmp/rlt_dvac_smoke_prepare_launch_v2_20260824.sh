set -euo pipefail
runtime_root=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v2/runtime
mkdir -p "${runtime_root}"
chmod 755 "${runtime_root}/run_foreground.sh" "${runtime_root}/launch.sh"
cp /root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1/runtime/resolved.yaml "${runtime_root}/resolved.yaml"
cp /root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1/runtime/exact_command.txt "${runtime_root}/exact_command_v1_reference.txt"
"${runtime_root}/launch.sh"
