set -euo pipefail

python_bin=/root/autodl-tmp/RLinf/.venv/bin/python
script=/root/autodl-tmp/idea2_dvac_analysis_scripts/analyze_dvac_first_collection.py
source_dir=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin/outputs/idea2_dvac_sft_smoke_2gpu_16env_v1/dvac_telemetry
output_dir=/root/autodl-tmp/idea2_dvac_analysis/first_collection_v3_20260820

sha256sum "$script"
"$python_bin" -m py_compile "$script"
"$python_bin" "$script" --source "$source_dir" --output "$output_dir"
