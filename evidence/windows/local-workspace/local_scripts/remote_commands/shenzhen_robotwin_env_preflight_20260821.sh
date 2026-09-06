set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh

printf '%s\n' '=== IDENTITY AND SPACE ==='
id
printf 'home='; printf '%s\n' "$HOME"
df -h "$HOME" /data/chenyiteng /tmp
df -i "$HOME" /data/chenyiteng /tmp

printf '%s\n' '=== PYTHON AND ENV MANAGERS ==='
python3 --version
command -v python3
for env_tool in conda mamba micromamba uv pip pip3; do
  if command -v "$env_tool" >/dev/null 2>&1; then
    printf '%s=' "$env_tool"
    command -v "$env_tool"
  else
    printf '%s=missing\n' "$env_tool"
  fi
done

printf '%s\n' '=== CUDA TOOLKITS AND COMPILERS ==='
ls -ld /usr/local/cuda /usr/local/cuda-* 2>/dev/null || true
for nvcc_path in /usr/local/cuda-*/bin/nvcc; do
  if [ -x "$nvcc_path" ]; then
    printf '\n-- %s --\n' "$nvcc_path"
    "$nvcc_path" --version
  fi
done
gcc --version | sed -n '1p'
g++ --version | sed -n '1p'
cmake --version | sed -n '1p'
command -v ninja || true
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader

printf '%s\n' '=== REQUIRED SYSTEM PACKAGES ==='
dpkg-query -W -f='${binary:Package}\t${Version}\t${Status}\n' \
  libvulkan1 mesa-vulkan-drivers vulkan-tools ffmpeg unzip git-lfs build-essential ninja-build \
  2>&1 || true

printf '%s\n' '=== MINIFORGE LATEST INSTALLER METADATA ==='
curl -fsSL --max-time 60 https://api.github.com/repos/conda-forge/miniforge/releases/latest \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("tag=",d["tag_name"]); a=next(x for x in d["assets"] if x["name"]=="Miniforge3-Linux-x86_64.sh"); print("name=",a["name"]); print("size_bytes=",a["size"]); print("digest=",a.get("digest")); print("url=",a["browser_download_url"])'

printf '%s\n' '=== HUGGING FACE ROOT ASSET METADATA ==='
curl -fsSL --max-time 60 'https://huggingface.co/api/datasets/TianxingChen/RoboTwin2.0/tree/main?recursive=false&expand=true' \
  | python3 -c 'import json,sys; rows=json.load(sys.stdin); wanted={"background_texture.zip","embodiments.zip","objects.zip"}; [(print(x.get("path"),x.get("size"),x.get("lfs",{}).get("oid"))) for x in rows if x.get("path") in wanted]'

printf '%s\n' '=== ADJUST_BOTTLE DATA ARCHIVE METADATA ==='
curl -fsSL --max-time 60 'https://huggingface.co/api/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset/adjust_bottle?recursive=false&expand=true' \
  | python3 -c 'import json,sys; rows=json.load(sys.stdin); [(print(x.get("path"),x.get("size"),x.get("lfs",{}).get("oid"))) for x in rows if x.get("path","").endswith("demo_clean.zip")]'
