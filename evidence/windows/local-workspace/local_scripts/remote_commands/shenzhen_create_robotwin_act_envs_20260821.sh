set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh

conda_bin=/home/chenyiteng/miniforge3/bin/conda
robotwin_prefix=/home/chenyiteng/miniforge3/envs/RoboTwin
act_prefix=/home/chenyiteng/miniforge3/envs/act

printf '%s\n' '=== PRECHECK ==='
"$conda_bin" --version
for environment_prefix in "$robotwin_prefix" "$act_prefix"; do
  if [ -e "$environment_prefix" ]; then
    printf 'REFUSE_EXISTING_PREFIX=%s\n' "$environment_prefix" >&2
    exit 40
  fi
done
df -h /home/chenyiteng

printf '%s\n' '=== CREATE ROBOTWIN PYTHON 3.10 ENV ==='
"$conda_bin" create --yes --name RoboTwin python=3.10 pip

printf '%s\n' '=== CREATE ACT PYTHON 3.10 ENV ==='
"$conda_bin" create --yes --name act python=3.10 pip

printf '%s\n' '=== VERIFY ==='
"$conda_bin" env list
"$robotwin_prefix/bin/python" --version
"$act_prefix/bin/python" --version
"$robotwin_prefix/bin/python" -m pip --version
"$act_prefix/bin/python" -m pip --version
du -sh "$robotwin_prefix" "$act_prefix"

printf '%s\n' '=== CUDA 12.1 TOOLKIT DRY RUN FOR ROBOTWIN ==='
"$conda_bin" install --dry-run --json --prefix "$robotwin_prefix" \
  --channel nvidia/label/cuda-12.1.1 --channel conda-forge \
  cuda-toolkit=12.1.1 \
  | "$robotwin_prefix/bin/python" -c 'import json,sys; d=json.load(sys.stdin); actions=d.get("actions",{}); fetch=actions.get("FETCH",[]); print("success=",d.get("success")); print("package_count=",len(actions.get("LINK",[]))); print("download_bytes=",sum(int(x.get("size",0)) for x in fetch)); print("fetch_count=",len(fetch)); print("packages=",",".join(x.get("name","")+"="+x.get("version","") for x in actions.get("LINK",[])))'
