set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh

conda_bin=/home/chenyiteng/miniforge3/bin/conda
robotwin_prefix=/home/chenyiteng/miniforge3/envs/RoboTwin

"$conda_bin" install --dry-run --json --prefix "$robotwin_prefix" \
  --channel nvidia/label/cuda-12.1.1 --channel conda-forge \
  cuda-nvcc=12.1.105 cuda-cudart-dev=12.1.105 cuda-cccl=12.1.109 \
  | "$robotwin_prefix/bin/python" -c 'import json,sys; d=json.load(sys.stdin); actions=d.get("actions",{}); fetch=actions.get("FETCH",[]); print("success=",d.get("success")); print("package_count=",len(actions.get("LINK",[]))); print("download_bytes=",sum(int(x.get("size",0)) for x in fetch)); print("fetch_count=",len(fetch)); print("packages=",",".join(x.get("name","")+"="+x.get("version","") for x in actions.get("LINK",[])))'
