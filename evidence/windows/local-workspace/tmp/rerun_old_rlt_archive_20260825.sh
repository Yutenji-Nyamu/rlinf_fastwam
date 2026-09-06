set -euo pipefail

runtime_package="/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_runtime_package"
export_root="/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/high_info_archive_20260825"
stage="$export_root/staging"
archive="$export_root/rlt_teacher_dvac_w0to2_formal_fresh480_high_info_raw_20260825.tar.gz"

[ "$stage" = "/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/high_info_archive_20260825/staging" ]
[ ! -e "$archive" ]

if [ -d "$stage" ]; then
  rm -rf -- "$stage"
fi

bash -n "$runtime_package/archive_old.sh"
bash "$runtime_package/archive_old.sh"

echo "=== archive result ==="
ls -lh "$archive" "${archive}.sha256"
sha256sum -c "${archive}.sha256"
tar -tzf "$archive" | wc -l
tar -tzf "$archive" | sed -n '1,80p'
