set -euo pipefail

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin

cuda_context_header="$CONDA_PREFIX/lib/python3.10/site-packages/torch/include/ATen/cuda/CUDAContextLight.h"
printf '%s\n' '=== TORCH CUDA HEADER CONTRACT ==='
sed -n '1,120p' "$cuda_context_header"

printf '%s\n' '=== REQUIRED HEADER PRESENCE ==='
for cuda_header in cublas_v2.h cusparse.h cusolverDn.h cusolverMg.h curand.h cufft.h nvrtc.h nvToolsExt.h; do
  header_path=$(find "$CONDA_PREFIX/include" -name "$cuda_header" -print -quit 2>/dev/null || true)
  printf '%s=%s\n' "$cuda_header" "${header_path:-missing}"
done

printf '%s\n' '=== RELEVANT CUDA DEV PACKAGES DRY RUN ==='
/home/chenyiteng/miniforge3/bin/conda install --dry-run --json --prefix "$CONDA_PREFIX" \
  --channel nvidia/label/cuda-12.1.1 --channel conda-forge \
  libcublas-dev=12.1.3.1 libcusparse-dev=12.1.0.106 libcusolver-dev=11.4.5.107 \
  | python -c 'import json,sys; d=json.load(sys.stdin); actions=d.get("actions",{}); fetch=actions.get("FETCH",[]); print("success=",d.get("success")); print("download_bytes=",sum(int(x.get("size",0)) for x in fetch)); print("packages=",",".join(x.get("name","")+"="+x.get("version","") for x in actions.get("LINK",[])))'
