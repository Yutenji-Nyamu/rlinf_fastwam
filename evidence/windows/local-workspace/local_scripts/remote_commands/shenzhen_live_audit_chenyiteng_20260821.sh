#!/usr/bin/env bash

set +e

section() {
  printf '\n[%s]\n' "$1"
}

section identity
date --iso-8601=seconds
hostnamectl 2>/dev/null || true
pwd
id
groups
umask
sudo -n -l 2>&1

section operating_system
cat /etc/os-release
uname -a
uptime
lscpu | sed -n '1,28p'
free -h

section filesystems
df -hT / /home /data /scratch 2>&1
df -ih / /home /data /scratch 2>&1
findmnt -T /home/chenyiteng -o TARGET,SOURCE,FSTYPE,OPTIONS
findmnt -T /data/chenyiteng -o TARGET,SOURCE,FSTYPE,OPTIONS
findmnt -T /scratch/chenyiteng -o TARGET,SOURCE,FSTYPE,OPTIONS
ls -ld /home/chenyiteng /data /data/chenyiteng /data/shared /scratch /scratch/chenyiteng 2>&1
readlink -f /home/chenyiteng/data /home/chenyiteng/scratch /home/chenyiteng/shared 2>&1
command -v getfacl >/dev/null && getfacl -cp /data/chenyiteng /scratch/chenyiteng /data/shared 2>&1
command -v quota >/dev/null && quota -s 2>&1

section user_workspace_inventory
find /home/chenyiteng -mindepth 1 -maxdepth 2 -printf '%M %u:%g %s %TY-%Tm-%TdT%TH:%TM %p\n' 2>&1 | sort
find /data/chenyiteng -mindepth 1 -maxdepth 2 -printf '%M %u:%g %s %TY-%Tm-%TdT%TH:%TM %p\n' 2>&1 | sort | head -n 300
find /scratch/chenyiteng -mindepth 1 -maxdepth 2 -printf '%M %u:%g %s %TY-%Tm-%TdT%TH:%TM %p\n' 2>&1 | sort | head -n 300

section gpu_driver_cuda
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,driver_version,pci.bus_id,memory.total,memory.used,utilization.gpu,temperature.gpu,pstate,compute_mode --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>&1
nvidia-smi topo -m
command -v nvcc 2>&1
nvcc --version 2>&1
ls -ld /usr/local/cuda /usr/local/cuda-* 2>&1

section processes_services
ps -eo user,pid,ppid,stat,%cpu,%mem,rss,etimes,cmd --sort=-rss | head -n 50
pgrep -af 'raylet|gcs_server|train_embodied_agent|torchrun|python.*train|RoboTwin|sapien' 2>&1
systemctl is-active docker containerd 2>&1
ls -l /var/run/docker.sock 2>&1

section runtimes_tools
for tool in git git-lfs uv conda mamba micromamba python3 pip3 cmake ninja gcc g++ ffmpeg vulkaninfo eglinfo; do
  command -v "$tool" 2>/dev/null || printf '%s: not found\n' "$tool"
done
git --version 2>&1
git lfs version 2>&1
python3 --version 2>&1
uv --version 2>&1
cmake --version 2>&1 | head -n 1
gcc --version 2>&1 | head -n 1
ffmpeg -version 2>&1 | head -n 1

section graphics_stack
ldconfig -p 2>/dev/null | grep -E 'lib(EGL|GLX|GL|vulkan|nvidia-egl)' | head -n 120
find /usr/share/glvnd/egl_vendor.d /usr/share/vulkan/icd.d -maxdepth 1 -type f -printf '%p\n' 2>&1 | sort
if command -v vulkaninfo >/dev/null; then timeout 25s vulkaninfo --summary 2>&1 | head -n 200; fi
if command -v eglinfo >/dev/null; then timeout 25s eglinfo -B 2>&1 | head -n 200; fi

section limits_kernel
ulimit -a
sysctl kernel.pid_max fs.file-max fs.inotify.max_user_watches fs.inotify.max_user_instances 2>&1
cat /proc/sys/vm/overcommit_memory 2>&1
cat /proc/sys/vm/max_map_count 2>&1

section network_configuration
ip -brief address
ip route
getent hosts github.com pypi.org files.pythonhosted.org huggingface.co hf-mirror.com
env | grep -iE '(^|_)(http|https|all|no)_proxy=|HF_ENDPOINT|PIP_INDEX|UV_INDEX|CONDA' | sort
git config --global --get-regexp '^(http\.|https\.|url\.|credential\.|core\.sshCommand)' 2>&1

section network_probes
timeout 20s curl -fsSIL --max-time 15 https://github.com/RoboTwin-Platform/RoboTwin 2>&1 | sed -n '1,12p'
timeout 20s git ls-remote https://github.com/RoboTwin-Platform/RoboTwin.git HEAD refs/heads/main refs/heads/RoboTwin2.0 2>&1
timeout 20s git ls-remote https://github.com/RLinf/RLinf.git HEAD refs/heads/main 2>&1
timeout 20s curl -fsSIL --max-time 15 https://pypi.org/simple/ 2>&1 | sed -n '1,12p'
timeout 20s curl -fsSIL --max-time 15 https://huggingface.co/robots.txt 2>&1 | sed -n '1,12p'
timeout 20s curl -fsSIL --max-time 15 https://hf-mirror.com/robots.txt 2>&1 | sed -n '1,12p'

section package_presence
dpkg-query -W -f='${binary:Package}\t${Version}\n' build-essential cmake ninja-build git git-lfs ffmpeg libegl1 libgl1 libglvnd0 libvulkan1 vulkan-tools pkg-config 2>&1

section end
date --iso-8601=seconds
exit 0
