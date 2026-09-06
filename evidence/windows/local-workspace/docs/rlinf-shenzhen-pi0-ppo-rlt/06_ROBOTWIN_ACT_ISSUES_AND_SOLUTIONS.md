# 深圳 RoboTwin 2.0 / ACT：官方流程之外的问题与解决

> 只列本次真实发生、且会影响复现的偏差。完整命令、退出码、SHA 与原始关键输出见分阶段 ledger；
> 未发生的可能性和备用诊断树不写入本文。

## 1. 旧服务器路径已经失效

**症状**：8 月 19 日材料中的 `/scratch/chenyiteng/...` 与 `~/scratch` 在 8 月 21 日现场已不存在；
继续照旧会把源码、环境或大数据放错盘。

**处理**：

- source、assets、dataset、model、run result 放 `/data/chenyiteng`；
- Conda env 与用户 cache 放扩容后的 `/home/chenyiteng`；
- 不使用 `/scratch`，也不把 GB/TB 级产物落到 `/`。

**复测**：ACT postflight 时 `/` 约 237 GiB、`/home` 约 2.3 TiB、`/data` 约 3.2 TiB 可用；本轮
大文件均在 `/home`/`/data`。

## 2. Paramiko 非登录 shell 没有自动代理

**症状**：系统 Mihomo 正常监听 `127.0.0.1:7890`，但 Paramiko `exec_command` 不是 login shell，
不会自动加载 `/etc/profile.d/mihomo-proxy.sh`。HF 直连出现 timeout/TLS reset；这曾让 postflight
小请求失败，但不代表配额耗尽。

**处理**：所有联网 command file 在同一 shell 显式：

```bash
source /etc/profile.d/mihomo-proxy.sh
```

然后再执行 Git/HF/pip/Conda 请求。没有把代理 URL、订阅 token 或密码写入文档/脚本。

**复测**：同一 pinned HF API，direct exit 35，显式 Mihomo HTTP 200；GitHub/HF 最终 postflight
均 HTTP 200。20:20 CST 代理配额为 100 GiB total、45.251 GiB used、54.749 GiB remaining，
到期时间为 2026-08-23 14:01:48 CST。

## 3. Official installer 会前移 submodule，且内部失败不一定让脚本失败

**症状 A**：RoboTwin parent commit pin 的 XPolicyLab 是 `c37109c...`，但 official `_install.sh`
调用 updater，把它更新到当时 `origin/main=c07a096...`。只记录 parent commit 会误报实际运行源码。

**处理 A**：clone 后和安装后分别记录 RoboTwin HEAD、XPolicyLab HEAD 与 target-scope git status；
完成态明确写成 RoboTwin `30954692...` + installed XPolicyLab `c07a096...`。

**症状 B**：official `_install.sh` 没有 `set -e`。PyTorch3D 编译失败后它仍继续，并打印
`Installation basic environment complete`；只看最后一行会假阳性。

**处理 B**：外围 command file 使用严格 shell 语义，并在 official script 后执行核心 import/CUDA probe。
同类地，`collect_data.sh` 捕获 episode 异常后还会继续处理 seed/cleanup，成功以 HDF5/JSON/MP4/seed
产物为准，不只看 shell code。

## 4. PyTorch3D 编译缺少 `cusparse.h`

**精确错误**：

```text
ATen/cuda/CUDAContextLight.h:7:10: fatal error: cusparse.h: No such file or directory
```

**现场原因**：为了对齐 torch 2.4.1+cu121，只在 Conda env 中先装了最小
`cuda-nvcc + cuda-cudart-dev + cuda-cccl`；PyTorch3D 还需要 cuSPARSE/cuBLAS/cuSOLVER headers/libs。
直接补全 Conda dev packages 需额外下载约 837.5 MB。

**选定修复**：现场确认已经安装的 PyTorch NVIDIA 12.1 wheels 自带所需 headers/libs；只在本次
PyTorch3D build 的 `CPATH/CPLUS_INCLUDE_PATH/LIBRARY_PATH/LD_LIBRARY_PATH` 注入这些目录，安装小型
`ninja`，设 `TORCH_CUDA_ARCH_LIST=9.0` 后重建锁定 commit。精确脚本：
[`local_scripts/remote_commands/shenzhen_repair_pytorch3d_headers_20260821.sh`](../../local_scripts/remote_commands/shenzhen_repair_pytorch3d_headers_20260821.sh)。

**复测**：PyTorch3D wheel build/install exit 0；torch、torchvision、PyTorch3D、SAPIEN、MPLib、
CuRobo 全部 import 并看到 H100。没有安装 3.53-GB 完整 CUDA toolkit，也没有改系统 CUDA symlink。

## 5. CuRobo `lbfgs_step_cu` 在 H100 上触发 CUDA illegal instruction

**原始症状**：official render 已通过，但 collect 的每个候选 seed 都在
`[Start Seed and Pre Motion Data Collection]` 同一点报：

```text
CUDA error: an illegal instruction was encountered
```

**精确定位**：最小复现将栈收敛为：

```text
LBFGSOpt._get_step_direction
  -> LBFGScu.apply
  -> lbfgs_step_cu.forward
  -> CUDA error 715
```

现场组合是 H100/sm90、torch 2.4.1+cu121、CuRobo v0.7.8、Warp 1.12.0。因为 render 正常、
最小 MotionGen 复现直接命中 fused kernel，故不是 ACT、HDF5 写盘或 SAPIEN renderer 问题。

**选定局部兼容修复**：只在 `GPU capability >= 9` 且 PyTorch CUDA runtime `< 12.6` 时，把已构造的
5 个 IK/trajopt/finetune LBFGS optimizer 的 `use_cuda_kernel` 设为 `False`，随后再构造两个
`MotionGen`。完整 patch：
[`local_scripts/remote_patches/robotwin_h100_cuda121_unfused_lbfgs.patch`](../../local_scripts/remote_patches/robotwin_h100_cuda121_unfused_lbfgs.patch)。

这不是关闭 CuRobo 或改用 CPU/另一规划器：仍使用 CuRobo 的 GPU LBFGS、原 objective/iteration/
line-search，只旁路失败的 fused step-direction kernel；非 Hopper 或 CUDA runtime >=12.6 保持原行为。

**复测链**：

1. 不改 source 的同配置 unfused MotionGen warmup：40.33 s，`UNFUSED_LBFGS_WARMUP_OK`；
2. patch `git apply --check`、`py_compile`、`git diff --check` 通过；
3. official collect seed 0：1/1 pre-motion simulation success，141 rows、142-frame video，exit 0；
4. 后续 official HF ACT simulator eval actual seed 100001 在 step 147 成功，scheduler exit 0。

**长期边界**：若以后需要大量 standalone native collection，可另建 torch 2.6/cu126 sibling env，
重新编译 CuRobo 后比较性能；本次没有在原 env 全栈升级。进入 RLinf compatibility tree 时不自动复制
此 patch，只有真实复现同一个 `lbfgs_step_cu` 栈才移植窄修复。

## 6. Official 数据与训练默认值不适合“快速功能闭环”

**数据**：HF release 全量约 1.62 TB，本轮只取三项基础 assets、`adjust_bottle/demo_clean` 与目标
ACT checkpoint leaf；checkpoint 使用 `allow_patterns` 精确下载 `policy_last.ckpt` 和
`dataset_stats.pkl`，没有下载整个 dataset repo。

**训练**：current official `train.sh` 固定 6000 epochs，不能为证明“训练入口可用”而跑完整训练或
启动后强杀。选用同一个 official `imitate_episodes.py`，保持 ACT/KL/chunk/hidden/FFN/batch/lr 参数，
只把预算显式设为 `num_epochs=1, save_freq=1` 并写入 `not_for_eval` 目录。

**结果边界**：1 epoch 正常完成并生成两个 checkpoint，证明 preprocess/dataloader/forward/backward/
optimizer/save 链闭合；真实 simulator eval 只使用 TianxingChen official HF weights，不把 debug 权重
包装成 baseline。

## 7. Expert check 会改变实际 eval seed

`--seed 0` 是 scheduler 的 seed 序列入口，不等于最终策略一定跑 candidate seed 100000。official
`--expert-check` 先筛掉 candidate 100000，实际策略运行 seed 100001。因此 ledger、result 和视频都标记
actual seed 100001；不能只根据 CLI 的 `--seed 0` 命名结果。

本次 actual seed 100001 的 1/1 成功仅是工程闭环事实，不外推成 official ACT 稳定成功率。
