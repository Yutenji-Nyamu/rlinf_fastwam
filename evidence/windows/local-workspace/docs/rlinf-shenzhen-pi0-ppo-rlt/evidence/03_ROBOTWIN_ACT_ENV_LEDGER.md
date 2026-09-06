# RoboTwin 2.0 / ACT 环境安装流水账

## 环境边界

- 普通账号安装，不用 `toom` 创建项目文件。
- 环境根：扩容后的 `/home/chenyiteng`；大源码、资产、数据与结果仍在 `/data/chenyiteng`。
- 原生仿真环境与 ACT policy 环境分开，避免 PyTorch/MuJoCo/SAPIEN 依赖互相污染。
- 先按锁定源码核对 Python、CUDA toolkit、编译链和官方脚本副作用；只有确有缺失的系统依赖才使用精确 sudo 命令。

## 操作记录

> 从安装前 CUDA/编译链与 installer 元数据预检起按实际顺序追加。

### SZ-ENV-001 — 环境、CUDA 与下载体积预检

- 时间：2026-08-21 17:05 CST
- command file：`local_scripts/remote_commands/shenzhen_robotwin_env_preflight_20260821.sh`
- SHA256：`0a640cbe6ad877e5ed202e27dd4347916c11614de7f408952ce77efbbe9c9dca`
- 结果：退出码 0，全程只读。
- 现场：home 约 2.3 TiB 可用，data 约 3.2 TiB 可用，root `/tmp` 所在卷约 237 GiB 可用；inode 均充足。系统 Python 3.10.12，但没有 conda/mamba/uv/pip。
- 系统依赖：Vulkan、Mesa Vulkan driver、vulkan-tools、FFmpeg、unzip、Git LFS、build-essential 已安装；`ninja-build` 未安装。当前不需要 sudo 安装 RoboTwin 文档列出的系统图形包。
- CUDA：只有 `/usr/local/cuda-12.9`，`nvcc 12.9.86`；8 张 H100 仍空闲。RoboTwin 锁定 requirements 是 `torch==2.4.1`，PyTorch3D 需编译，不能未经验证就混用 system nvcc 12.9；计划在 conda 环境内提供与 torch 对齐的 12.1 编译工具链，不修改系统 CUDA symlink。
- Miniforge 官方 latest 元数据：release `26.3.2-3`，installer 105,172,629 bytes，SHA256 `848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c`。
- HF 精确压缩体积：
  - `background_texture.zip` 10,970,687,027 bytes；
  - `embodiments.zip` 219,859,313 bytes；
  - `objects.zip` 3,737,778,549 bytes；
  - 合计 14,928,324,889 bytes（十进制 14.93 GB，约 13.90 GiB），展开量官方未给出；
  - `dataset/adjust_bottle/demo_clean.zip` 293,694,934 bytes（约 280.09 MiB）。
- 配额决策：不下载 1.62 TB 全量数据；后续只取上述基础资产与 `adjust_bottle/demo_clean.zip`。基础资产下载约占当前代理剩余额度的 20%，执行前仍需指定目录并保留空间复核。

### SZ-ENV-002 — 用户态 Miniforge 安装

- 时间：2026-08-21 17:07 CST
- command file：`local_scripts/remote_commands/shenzhen_install_miniforge_20260821.sh`
- SHA256：`4303926f9c7a238193a4eef35bca543b67172d038b215b029791e0a7623598ee`
- 精确动作：从 conda-forge/Miniforge release `26.3.2-3` 下载 105,172,629-byte installer 到 `/home/chenyiteng/installers/`，以官方 SHA256 严格校验，然后 batch install 到 `/home/chenyiteng/miniforge3`；未运行 `conda init`，未修改 login shell。
- 结果：退出码 0；installer 校验通过；Conda `26.3.2`、base Python `3.13.13`，prefix 约 562 MiB；installer 保留约 101 MiB，未删除任何文件。
- 后续调用固定使用 `/home/chenyiteng/miniforge3/bin/conda` 或显式 `source .../etc/profile.d/conda.sh`，不依赖交互 shell 状态。

### SZ-ENV-003 — 两个 Python 3.10 环境与 CUDA toolkit 事务估算

- 时间：2026-08-21 17:09 CST
- command file：`local_scripts/remote_commands/shenzhen_create_robotwin_act_envs_20260821.sh`
- SHA256：`7f884cf2b3b7261d39c9ea002ec905bf2de705ab1be962ba766d9e7c65cf75f5`
- 完整动作：

```bash
/home/chenyiteng/miniforge3/bin/conda create --yes --name RoboTwin python=3.10 pip
/home/chenyiteng/miniforge3/bin/conda create --yes --name act python=3.10 pip
```

- 结果：退出码 0；两个环境均为 Python 3.10.20。`RoboTwin` prefix 约 223 MiB，`act` prefix 约 59 MiB（包 cache 共享造成落盘差异）。
- 完整 `cuda-toolkit=12.1.1` dry-run：需 link 64 packages、下载 3,530,474,803 bytes，并含 Nsight、文档、profilers 与大量 static libraries。它能对齐官方推荐版本，但对编译 PyTorch3D/CuRobo 明显过宽；先估算最小 `nvcc + cudart-dev + CCCL` 事务，再按观测到的缺失窄增量处理。

### SZ-ENV-004 — 最小 CUDA 12.1 编译工具链 dry-run

- 时间：2026-08-21 17:11 CST
- command file：`local_scripts/remote_commands/shenzhen_cuda121_minimal_dryrun_20260821.sh`
- SHA256：`bbdca64a0f82adbd0256c4dfd34ec4270d2f23a6b042504e7f9dff1492bb97de`
- 结果：退出码 0；`cuda-nvcc=12.1.105 + cuda-cudart-dev=12.1.105 + cuda-cccl=12.1.109` 实际只需 4 packages（另含 runtime）、下载 56,762,248 bytes。
- 决策：先安装这组最小工具链，并令官方安装进程的 `CUDA_HOME` 指向 RoboTwin conda prefix。若编译实际报告缺少特定 CUDA library/header，再按错误补对应 dev 包；不先装包含 Nsight/文档/全部静态库的 3.53 GB toolkit。

### SZ-ENV-005 — 首次官方 `_install.sh` 执行与 PyTorch3D 单点失败

- 时间：2026-08-21 17:12–17:28 CST
- command file：`local_scripts/remote_commands/shenzhen_install_robotwin_official_20260821.sh`
- SHA256：`97121f38e993319cbdaecbc356106a6058966baa704309a773b3247a2319b4a4`
- 外围约束：确认 RoboTwin/XPolicyLab pre-lock 与 clean tree；确认 XPolicyLab `origin/main=c07a096...`；设置 `CUDA_HOME=$CONDA_PREFIX`、`MAX_JOBS=16`、单卡可见；先安装并验证 `nvcc 12.1.105`，再原样调用 `bash scripts/_install.sh`。
- 成功部分：
  - 官方 requirements 安装完成；`torch 2.4.1`、resolver 对齐的 `torchvision 0.19.1`、SAPIEN/MPLib/Open3D 等已安装。
  - 官方 update 把 XPolicyLab 从 parent pin `c37109c...` 更新到预检的 current main `c07a096...`；顶层只显示预期的 `M XPolicyLab`。
  - editable XPolicyLab、SAPIEN/MPLib 官方 sed 修改完成。
  - CuRobo v0.7.8 checkout 为 `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`，editable build 完成；官方随后固定 `warp-lang=1.12.0`、`setuptools=69.5.1`。
  - CuRobo 的无上界依赖把 SciPy 1.10.1 升到 1.15.3；先作为官方实况保留，后续 import/仿真验证决定是否需要处理。
- 失败：PyTorch3D stable `75ebeeaea0908c5527e7b1e305fbc7681382db47` 编译在首个 CUDA source 上报：

```text
ATen/cuda/CUDAContextLight.h:7:10: fatal error: cusparse.h: No such file or directory
```

- 重要脚本行为：官方 `_install.sh` 没有 `set -e`，PyTorch3D 子命令失败后仍继续其余安装，最后还打印“Installation basic environment complete”；不能把这行当成成功证据。外围最终 import probe 因 `ModuleNotFoundError: pytorch3d` 退出 1，正确阻止了误报完成。
- 原因复核 command file：`local_scripts/remote_commands/shenzhen_pytorch3d_failure_probe_20260821.sh`，SHA256 `33eeae0c10d7d094ef65cc582b7d25f5f71658f04b9dd30404cb485e7f20cde1`，退出码 0。Torch 编译 header 明确需要 `cusparse.h`、`cublas_v2.h`、`cublasLt.h` 和 `cusolverDn.h`；当前 conda include 均缺失。
- 三个 conda dev packages 的 dry-run 需额外下载 837,545,289 bytes。Torch 已安装的 NVIDIA wheel 通常携带相同 12.1 版本 headers/libs；先现场定位并尝试把这些已下载目录仅注入本次编译，避免重复下载，失败再安装 conda dev packages。

### SZ-ENV-006 — 复用 Torch wheel headers 修复 PyTorch3D 并闭合核心 imports

- 时间：2026-08-21 17:30–17:32 CST
- command file：`local_scripts/remote_commands/shenzhen_repair_pytorch3d_headers_20260821.sh`
- SHA256：`4581e306f7cbeebf9800b6c3a265b7493e2100bf66670519f835c3f2b33473d5`
- 处理：现场确认 Torch 的 NVIDIA 12.1 wheels 已包含 `cusparse.h`、`cublas_v2.h`、`cublasLt.h`、`cusolverDn.h` 与对应 libs；仅在本次 build 的 `CPATH/CPLUS_INCLUDE_PATH/LIBRARY_PATH/LD_LIBRARY_PATH` 注入这些现有目录。安装 180-KB `ninja 1.13.0`，设置 `TORCH_CUDA_ARCH_LIST=9.0`、`MAX_JOBS=16`，重建精确 PyTorch3D commit `75ebeea...`。
- 结果：退出码 0；PyTorch3D wheel 构建并安装成功，避免了 837.5 MB 重复 CUDA dev 下载。
- 核心验证：
  - `torch 2.4.1+cu121` / `torch.cuda.is_available()=True` / device `NVIDIA H100 80GB HBM3`；
  - `torchvision 0.19.1+cu121`、`pytorch3d 0.7.8`、`sapien 3.0.0b1`、`mplib 0.2.1` 均 import 成功；
  - editable XPolicyLab 指向当前源码树；CuRobo `TensorDeviceType()` 成功选到 CUDA；
  - 双 SHA：RoboTwin `30954692...`，XPolicyLab `c07a096...`；CuRobo `d64c4b0...`；顶层仅有预期的 submodule gitlink modified。
- 环境落盘：RoboTwin conda prefix 约 8.5 GiB，CuRobo source/build 约 240 MiB。
- 待验证：SAPIEN import 提示未找到 Vulkan ICD 文件，但系统 `vulkaninfo` 已成功枚举全部 H100；不凭 warning 改驱动/系统配置，留给官方 render gate 判断。

### SZ-ENV-007 — 当前 XPolicyLab ACT 官方环境安装

- 时间：2026-08-21 17:34–17:36 CST
- command file：`local_scripts/remote_commands/shenzhen_install_act_official_20260821.sh`
- SHA256：`1d4d8cc593466519bc15672eea340c940c97a10f450e854f5785a5a520048f97`
- 动作：激活独立 `act` Python 3.10 环境，在锁定的 `XPolicyLab/policy/ACT` 中原样运行 `bash install.sh`，再执行 imports 与 `pip check`。
- 结果：退出码 0；Torch `2.4.1+cu121` 与 resolver 匹配的 torchvision `0.19.1+cu121` 可见 H100；MuJoCo 2.3.7、dm_control 1.0.14、OpenCV 5.0.0、h5py 3.16.0、editable DETR/XPolicyLab 均成功加载；`pip check` 无 broken requirements。
- 当前官方未固定 NumPy/OpenCV，现场解析到 NumPy 2.2.6、OpenCV 5.0.0；不先做经验降级，留给 ACT preprocess/debug wiring 的真实结果判断。
- ACT prefix 落盘约 5.9 GiB。至此两个独立 Python 环境安装完成；尚未做仿真 render、数据处理或训练。
