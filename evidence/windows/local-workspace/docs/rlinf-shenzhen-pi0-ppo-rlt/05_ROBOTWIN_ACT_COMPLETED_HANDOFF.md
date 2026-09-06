# 深圳 RoboTwin 2.0 / ACT 完成态交接与复现入口

> 截止点：2026-08-21，`adjust_bottle` 的 standalone RoboTwin render、单条数据采集、ACT
> 1-epoch training smoke、official HF checkpoint 推理与真实 simulator eval/video 均已闭环。
> 本文只整理完成事实和复现顺序；逐条命令输出由
> [`evidence/00_SERVER_OPERATION_LEDGER_INDEX.md`](evidence/00_SERVER_OPERATION_LEDGER_INDEX.md) 路由。

## 1. 一页结论

| 环节 | 实际结果 | 证据边界 |
|---|---|---|
| Render | official `scripts/test_render.py` 输出 `Render Well`，exit 0 | 证明 Vulkan/SAPIEN 基础渲染可用 |
| Collect | 修复 H100/CuRobo fused LBFGS 后，seed 0，1/1 成功；141 行 HDF5、142 帧 MP4 | 临时数据验收后按批准精确删除，manifest/日志/patch 保留 |
| Data | official `adjust_bottle/demo_clean`：50 episodes、7,188 frames、三相机、14D state/action | 只下载单任务，不是 1.62 TB 全量数据 |
| Preprocess | 50 个 ACT HDF5，约 19 GiB | 位于 `/data`，不是根分区 |
| Train smoke | official `imitate_episodes.py`，1 epoch，40/10 split、3 train batches，exit 0 | checkpoint 只证明训练机制，不用于成功率 |
| Offline inference | official HF ACT checkpoint 完成 10×20 action steps，exit 0 | 证明权重、stats、14D action 与 policy transport 可用 |
| Simulator eval | actual seed 100001，step 147/400 success，scheduler 1/1，exit 0 | 单条 100% 不是稳定成功率 |
| 视频 | H.264，320×240，10 fps，148 帧，14.8 s | [`evidence/act_adjust_bottle_official_hf_seed100001_20260821.mp4`](evidence/act_adjust_bottle_official_hf_seed100001_20260821.mp4) |

ACT 通过证明的是：

```text
driver/Vulkan -> SAPIEN -> CuRobo -> assets/task -> HDF5 -> ACT -> simulator eval/video
```

它不等于 RLinf、Ray、FSDP、OpenPI 或 PPO 已通过。下一阶段仍需独立验证。

## 2. 机器、源码与环境锁

| 项目 | 完成态值 |
|---|---|
| 机器标签 | `SZ-H100`，Ubuntu 22.04.5，8×H100 80GB |
| 日常账号 | `chenyiteng`；系统管理才使用 sudo/管理员入口 |
| SSH | `120.241.223.9:22`；固定 host key `SHA256:qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY` |
| RoboTwin source | `/data/chenyiteng/projects/robotwin-native/RoboTwin@30954692d06ba7e89f7a6b76064f4062c488fa81` |
| XPolicyLab parent pin | `c37109c500be67d0dea6b36bf7337bbd26e763cd` |
| XPolicyLab installed | `c07a09614dd44cc4a67483bcb9a82e7439d99926`；official installer 按设计前移 |
| CuRobo | v0.7.8 checkout `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` |
| PyTorch3D | `75ebeeaea0908c5527e7b1e305fbc7681382db47` |
| Simulator env | `/home/chenyiteng/miniforge3/envs/RoboTwin`，Python 3.10，torch 2.4.1+cu121 |
| ACT env | `/home/chenyiteng/miniforge3/envs/act`，Python 3.10，torch 2.4.1+cu121 |
| HF release | `TianxingChen/RoboTwin2.0@a967b852afa21a9cbf19a198f7e653109042e87c` |

凭据不在本文、脚本、账本或 ZIP 中。Windows helper 只在当前进程无回显提示密码，并在认证前核对
固定 host key。

## 3. 完成态服务器布局

```text
/data/chenyiteng/projects/robotwin-native/RoboTwin/     # source、assets、raw/processed data
/home/chenyiteng/miniforge3/                            # Conda 与两个 env
/home/chenyiteng/.cache/                                # torch/HF 等用户缓存
/data/chenyiteng/models/robotwin2-hf-a967b852/          # official ACT checkpoint
/data/chenyiteng/runs/robotwin-native/adjust_bottle/    # logs、manifest、debug train output
```

2026-08-21 20:27 CST postflight：`/` 约 237 GiB 可用，`/home` 约 2.3 TiB、`/data` 约
3.2 TiB 可用；`/scratch` 已删除。大文件全部在 `/home` 或 `/data`：

- assets：约 16 GiB extracted；
- raw clean-50：约 423 MiB，另保留 293,694,934-byte archive；
- processed ACT data：约 19 GiB；
- official ACT checkpoint：335,918,106 bytes；
- 1-epoch debug train output：约 641 MiB。

## 4. 复现方式

### 4.1 本地执行器

ZIP 中 `local_scripts/` 下的 Paramiko 执行器不含密码。以任一 command file 为例：

```powershell
python local_scripts/verified_password_ssh.py `
  --host 120.241.223.9 --port 22 --user chenyiteng `
  --host-key-sha256 qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY `
  run --command-file local_scripts/remote_commands/shenzhen_act_postflight_20260821.sh
```

原工作区和解压后的 ZIP 都保留上述相对目录结构。不要把密码放到命令行、文档或 shell history。

`local_scripts/remote_exec_autodl.py` 是历史文件名，内部仍保留旧 AutoDL 默认 endpoint；它实际支持任意 SSH
endpoint。本包复现深圳服务器时必须像上例一样显式提供 `--host/--port/--user/--host-key-sha256`，
不能依赖默认值。

需要上传 config/patch 时使用同一 helper 的 `put` 子命令。三个实际 remote target 是：

```text
configs/sz_collect_smoke_1ep_20260821.yml
  -> /data/chenyiteng/projects/robotwin-native/RoboTwin/env_cfg/task_config/sz_collect_smoke_1ep_20260821.yml

configs/sz_adjust_bottle_act_1ep.yml
  -> /data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1/configs/sz_adjust_bottle_act_1ep.yml

patches/robotwin_h100_cuda121_unfused_lbfgs.patch
  -> /data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1/robotwin_h100_cuda121_unfused_lbfgs.patch
```

### 4.2 从空服务器到 ACT 闭环的顺序

每个文件包含当次完整 cwd、环境变量、路径、timeout 与产物校验。下面是复现主序列；失败时只进入
与实际错误对应的修复脚本。

| 顺序 | Command file | 作用 |
|---:|---|---|
| 1 | `shenzhen_live_audit_chenyiteng_20260821.sh`、`shenzhen_proxy_storage_refresh_20260821.sh` | 身份、磁盘、GPU、Vulkan、代理现场 |
| 2 | `shenzhen_clone_robotwin_native_20260821.sh` | recursive clone 并冻结 RoboTwin/XPolicyLab 双 SHA |
| 3 | `shenzhen_robotwin_env_preflight_20260821.sh` | Python/CUDA/build tools、下载量与空间 |
| 4 | `shenzhen_install_miniforge_20260821.sh`、`shenzhen_create_robotwin_act_envs_20260821.sh` | 用户态 Conda 与两个 Python 3.10 env |
| 5 | `shenzhen_cuda121_minimal_dryrun_20260821.sh`、`shenzhen_install_robotwin_official_20260821.sh` | 最小 CUDA 12.1 编译链与 official RoboTwin install |
| 6 | `shenzhen_repair_pytorch3d_headers_20260821.sh` | 仅当复现相同 `cusparse.h` 缺失时重建 PyTorch3D |
| 7 | `shenzhen_install_act_official_20260821.sh` | 独立 ACT env |
| 8 | `shenzhen_download_robotwin_assets_20260821.sh` | official 三项基础 assets，约 14.93 GB 压缩下载 |
| 9 | `shenzhen_download_adjust_bottle_demo_clean_20260821.sh`、`shenzhen_run_adjust_bottle_inventory_20260821.sh` | 单任务 clean-50 与 schema |
| 10 | `shenzhen_preprocess_adjust_bottle_act_20260821.sh` | official ACT preprocess |
| 11 | `shenzhen_download_official_act_adjust_bottle_20260821.sh` | revision-pinned official ACT leaf，仅两个文件 |
| 12 | `shenzhen_act_gate_a_render_20260821.sh` | official render gate |
| 13 | 上传 collect config；运行 `shenzhen_act_gate_a_collect_20260821.sh` | 单条 official collect |
| 14 | 若出现同一 CUDA 715 栈：最小复现、warmup、上传/apply patch、retry | 详见问题文档第 5 节 |
| 15 | `shenzhen_act_gate_a_verify_collection_20260821.sh` | HDF5/JSON/MP4/seed 验收与 manifest |
| 16 | （可选）`shenzhen_act_gate_a_cleanup_exact_smoke_20260821.sh` | 只适用于本轮新建且已验收的精确临时根；不是成功必要步骤 |
| 17 | `shenzhen_act_train_smoke_1epoch_20260821.sh` | official 1-epoch training mechanism smoke |
| 18 | `shenzhen_act_hf_offline_debug_20260821.sh` | official HF checkpoint offline action loop |
| 19 | 上传 eval config；运行 `shenzhen_act_hf_real_eval_1ep_20260821.sh` | dry-run 后真实 1-episode scheduler eval |
| 20 | 按本次 `shenzhen_act_hf_eval_verify_20260821.sh` 的字段检查新 resolved result；再运行 postflight | 前者硬编码本次 timestamp，只作 provenance/template，fresh rerun 先替换实际 result path |

这些 command files 是**本次运行的精确 provenance 与可审阅复现模板**，不是永久不变的一键镜像：
`shenzhen_install_robotwin_official_20260821.sh` 会核对当时 XPolicyLab main `c07a096...`，而 official
installer 自身还会更新 submodule；asset downloader 本次取得的 HF revision 是 `a967b852...`，但 official
wrapper 没有公开 revision 参数。未来 upstream 前移时，先以本文 locks/ledger 对照，而不是删掉 SHA gate
强行运行。若要求字节级长期重建，应另做 revision-pinned source/asset snapshot；本次没有伪装成这种镜像。

安装、下载、命令结果和 SHA256 分别见：

- [`evidence/01_LIVE_AUDIT_20260821.md`](evidence/01_LIVE_AUDIT_20260821.md)
- [`evidence/02_SOURCE_AND_DOWNLOAD_LEDGER.md`](evidence/02_SOURCE_AND_DOWNLOAD_LEDGER.md)
- [`evidence/03_ROBOTWIN_ACT_ENV_LEDGER.md`](evidence/03_ROBOTWIN_ACT_ENV_LEDGER.md)
- [`evidence/04_ACT_PIPELINE_LEDGER.md`](evidence/04_ACT_PIPELINE_LEDGER.md)

## 5. 关键 artifact 与完整性

| Artifact | Server/local path | 完整性 |
|---|---|---|
| official ACT weights | `/data/chenyiteng/models/robotwin2-hf-a967b852/act_ckpt/act-adjust_bottle/demo_clean-50/policy_last.ckpt` | 335,907,442 B；SHA256 `edfb0125103e67465cc2852ea1683acc2ce1060d02b81ba6f1113b5420b40690` |
| official stats | 同 leaf 的 `dataset_stats.pkl` | 10,664 B；SHA256 `a79964a7cce7a02cd172fb669ef12c8c2f5cedd2c4bd11adfd86c4d90cb239c4` |
| debug train weights | `$RUN/06_act_train_smoke_1epoch_not_for_eval/{policy_epoch_1_seed_0.ckpt,policy_last.ckpt}` | 只作训练机制证据 |
| collect manifest | `$RUN/03_collect_manifest.txt` | 临时数据删除前已记录 tree/bytes/hash/schema/seed |
| eval result | RoboTwin `eval_result/.../demo_clean-50/<timestamp>/{_result.txt,episode0.mp4}` | `_result.txt=1.0` |
| 本地视频 | [`evidence/act_adjust_bottle_official_hf_seed100001_20260821.mp4`](evidence/act_adjust_bottle_official_hf_seed100001_20260821.mp4) | 172,523 B；SHA256 `b9ee562adfbf9eff3c23af8b7ca6dff72bcfbc3fb0edadef5f333fa27c229340` |

`$RUN` 在本文中指：

```text
/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1
```

## 6. 已删除与仍保留

- 已删除：本轮唯一临时采集根
  `/data/.../RoboTwin/data/sz_collect_smoke_1ep_20260821` 与对应临时 task config；删除前完成验收，
  约 9 MiB，无法从服务器原路径恢复。
- 保留：官方 assets、clean-50 raw/processed data、official checkpoint、1-epoch debug output、全部日志、
  manifest、compatibility patch 与真实 eval/video。
- standalone source 的完成态不是 clean：top-level 记录 official installer 将 XPolicyLab 从 parent pin
  `c371...` 前移到 `c07a...`；`envs/robot/planner.py` 保留 20 行 H100/CUDA12.1 compatibility diff。
  两项均有 ledger/patch/hash，不能误当未知污染，也不要在复现前静默 reset。

## 7. 下一阶段边界

下一阶段按 [`04_LATEST_RLINF_PI0_PPO_NEXT_PLAN.md`](04_LATEST_RLINF_PI0_PPO_NEXT_PLAN.md)：

1. latest RLinf 独立 source/env；
2. `adjust_bottle + 精确 pi0` official SFT fixed-8 eval；
3. official PPO 精确一次 optimizer-step smoke；
4. 保存 `global_step_1`，fresh process reload 后 fixed-8 eval；
5. 完成后才讨论 fixed-128、完整 PPO、pilot/formal 与 RLT 适配。

当前 standalone assets 可以按该计划做独立 reflink/copy 复用；standalone source、ACT env/checkpoint、
processed ACT HDF5 不能替代 RLinf compatibility tree、OpenPI model 或在线 PPO rollout。
