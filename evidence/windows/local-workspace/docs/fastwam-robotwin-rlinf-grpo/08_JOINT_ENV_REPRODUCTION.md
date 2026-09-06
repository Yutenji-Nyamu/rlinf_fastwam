# Fast-WAM × RLinf 联合环境复现记录

更新时间：2026-07-17

本文只记录联合训练环境 `/root/autodl-tmp/conda/envs/FastWAM-RLinf`。官方单模型推理环境仍以 `07_OFFICIAL_STANDALONE_RUNBOOK.md` 为准，两者不要混装。

## 1. 已验证结果

- Python 3.11.15，Torch 2.7.1+cu128，Torchvision 0.22.1+cu128。
- Fast-WAM 0.1.0，commit `45d8e1458921d83f8ad6cf9ce993d371208dabd0`，editable 安装。
- RLinf 0.3.0，基线 commit `6d0db56bf26f972cd27fa29535f5eb939e80e5bf`，独立 worktree editable 安装。
- CUDA 12.8 下重新编译 CuRobo 0.7.8，源码 commit `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`。
- Ray 2.55.1、Transformers 4.49.0、Hydra 1.3.2、OmegaConf 2.3.0、NumPy 1.26.4、timm 1.0.27、SwanLab 0.8.2、TensorBoard 2.20.0、torchcodec 0.5、setuptools 80.9.0、warp-lang 1.11.1、toppra 0.6.3。
- `pip check`、RoboTwin render、CuRobo/Warp import 与 `adjust_bottle.play_once()` 均已通过。

## 2. 实际构建顺序

1. 从空的 Python 3.11 Conda 环境创建联合 venv；没有复制或修改现役 π0 venv。
2. cache、临时目录和 Torch extension 全部放在 `/root/autodl-tmp`；普通 Python 包清代理后使用清华源。
3. 先固定 `setuptools==80.9.0`，安装 `ninja`、`setuptools_scm`；再从 PyTorch cu128 索引安装 Torch/Torchvision。
4. 使用 `--no-build-isolation` editable 安装锁定 commit 的 Fast-WAM 和 RLinf worktree，避免 AutoDL 镜像缺少构建依赖。
5. 安装 RoboTwin 依赖；固定 `toppra==0.6.3`、`warp-lang==1.11.1`。
6. 在最终 Torch/CUDA 12.8 环境下，以 `MAX_JOBS=8`、A800 架构 8.0 编译并 editable 安装 CuRobo 0.7.8。
7. 对 SAPIEN 3.0.0b1 的 UTF-8 URDF/SRDF 路径和 MPLib planner 应用 RoboTwin 所需的精确补丁；补丁后做 control-byte scan 与 `py_compile`。
8. 将 RLinf 基础设施版本对齐已跑通 π0 环境：Ray 2.55.1、timm 1.0.27、SwanLab 0.8.2、TensorBoard 2.20.0。
9. 最后再次固定 setuptools/warp，执行 `pip check`、import、render、scripted expert 和 Fast-WAM 真实 checkpoint probe。
10. π0 venv 后续另做 freeze/list/editable/pip-check 快照和 14 GiB `rsync -aH` golden 备份；源环境未被改动。

## 3. 中途问题与最终处理

| 问题 | 根因 | 最终处理 |
|---|---|---|
| `ninja`、隔离构建的 `setuptools>=68` 找不到 | AutoDL Aliyun 镜像不全 | 清代理，普通包用清华源；editable 安装使用 `--no-build-isolation` |
| SAPIEN 缺 `pkg_resources` | 新 setuptools 已移除旧 API | 固定 `setuptools==80.9.0`，不重装 Torch/CuRobo |
| `toppra==0.6.8` 无 wheel | Python 3.11 与当前镜像只提供到 0.6.3 | 采用 π0/RoboTwin 已验证的 0.6.3，并以真实仿真验证 |
| `warp 1.15` 无 `wp.torch`，随后出现 `left_planner` | CuRobo 对 Warp 无上界，planner 首次初始化失败后留下连锁错误 | 固定 `warp-lang==1.11.1`；独立 device probe 先调用 `wp.init()` |
| SAPIEN/MPLib 文件出现控制字节 | 首版 shell `sed` 反向引用在脚本生成层被破坏 | 用同版本已跑通环境文件恢复；改为 fail-fast Python exact-string patch，并做 hash/compile 检查 |
| RLinf `[embodied]` 拉入较新基础设施版本 | 依赖解析漂移 | 按已跑通 π0 环境降回上述 Ray/timm/SwanLab/TensorBoard pins |

## 4. 当前复现边界

服务器旧的 `install_joint_env.sh` 含 `toppra==0.6.8` 和已损坏的 `sed`，禁止复用。`install_joint_env_continue1.sh` 是修复后的续跑脚本，不是从空环境的一键安装器，也没有做过 clean-room 全量重建。

因此，当前状态是“环境本身和关键行为已验证、构建过程已完整解释”，但还不能宣称存在经过 clean-room 验证的一键重建脚本。下一次确需重建时，应把上述顺序整合成唯一脚本，加入：锁定 commit 检查、infra pins、完整 `pip freeze`/editable 快照、control-byte/compile 检查以及 import/render/play_once 验收；在干净路径完整跑通后再标为 canonical。不要把两个历史脚本直接交给新机器执行。

## 5. 与既有项目经验的关系

- Fast-WAM 官方环境提供 Torch 2.7.1+cu128、setuptools/Warp、ModelScope 与 RoboTwin/CuRobo 的真实兼容事实。
- π0 + RLinf 已跑通环境提供 Python 3.11、Ray/timm/SwanLab/TensorBoard 和 RoboTwin 依赖的稳定版本参考。
- Motus/LaWAM 的经验提供“模型环境与 RLinf 环境隔离、CUDA extension 在最终 Torch 后重编译、先 render 再真实 task smoke”的实施顺序。
- 联合 venv 是第三个独立环境；它没有用“复制 π0 venv 后直接覆盖 Torch”的高风险做法。
