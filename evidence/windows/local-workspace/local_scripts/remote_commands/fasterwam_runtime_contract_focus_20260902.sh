set -euo pipefail

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
cd "$REPO"

printf 'ROBOTWIN_ENV_PYPROJECT\n'
sed -n '1,260p' environments/robotwin/pyproject.toml
printf 'FASTERWAM_MODEL_CONFIG\n'
sed -n '1,260p' configs/model/fasterwam.yaml
printf 'MODEL_RESOLVER\n'
grep -RInE 'DIFFSYNTH_MODEL_BASE_PATH|model_id_with_origin_paths|download_models|Wan2.2|models_t5|Wan2.2_VAE' src/fasterwam configs | sed -n '1,320p'
printf 'MODEL_IO_IMPLEMENTATION\n'
sed -n '1,240p' src/fasterwam/models/wan22/helpers/io.py
printf 'MODEL_LOADER_IMPLEMENTATION\n'
sed -n '110,230p' src/fasterwam/models/wan22/helpers/loader.py
