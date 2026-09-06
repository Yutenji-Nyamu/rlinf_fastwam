# RoboTwin 官方源码与下载流水账

## 本阶段目标与路径

- 日常账号：`chenyiteng`。
- 原生验证线：当前 RoboTwin 2.0 `main`，与后续 RLinf 兼容树严格分离。
- 源码/相对资产根：`/data/chenyiteng/projects/robotwin-native/RoboTwin`。
- 精确 RoboTwin commit：`30954692d06ba7e89f7a6b76064f4062c488fa81`。
- XPolicyLab commit：以该 RoboTwin commit 的 gitlink 为准；clone 后现场打印并冻结。
- 联网命令在同一 shell 中显式加载 `/etc/profile.d/mihomo-proxy.sh`，不假定 Paramiko command shell 自动继承代理。
- 不下载整个 RoboTwin 1.38 TB 数据集；先审阅官方脚本并核对每个最小资产包的下载量、展开量与目标剩余空间。

## 操作记录

> 从源码目标预检和 recursive clone 起按实际顺序追加。

### SZ-SRC-001 — 当前官方主线 recursive clone 与双 SHA 冻结

- 时间：2026-08-21 17:01–17:02 CST
- 远端账号/初始目录：`chenyiteng` / `/home/chenyiteng`
- 完整 command file：`local_scripts/remote_commands/shenzhen_clone_robotwin_native_20260821.sh`
- SHA256：`e017e702d44d5933e5221597a7ac685767e729acf6f88ac120cf6de61cf2d066`
- 目标：`/data/chenyiteng/projects/robotwin-native/RoboTwin`；执行前确认目标不存在、父目录本人可写，`/data` 可用约 3.2 TiB。
- 联网预检：同一 shell 中代理为 `http://127.0.0.1:7890`；官方 `refs/heads/main` 仍指向 `30954692d06ba7e89f7a6b76064f4062c488fa81`。GitHub API 报仓库逻辑 size 50,788 KiB，仅作为 clone 前量级参考。
- 完整动作：

```bash
mkdir -p /data/chenyiteng/projects/robotwin-native
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git /data/chenyiteng/projects/robotwin-native/RoboTwin
git -C /data/chenyiteng/projects/robotwin-native/RoboTwin checkout --detach 30954692d06ba7e89f7a6b76064f4062c488fa81
git -C /data/chenyiteng/projects/robotwin-native/RoboTwin submodule sync --recursive
git -C /data/chenyiteng/projects/robotwin-native/RoboTwin submodule update --init --recursive
```

- 结果：退出码 0；RoboTwin 为 detached HEAD `30954692d06ba7e89f7a6b76064f4062c488fa81`，XPolicyLab 为 `c37109c500be67d0dea6b36bf7337bbd26e763cd`；worktree clean，落盘 207 MiB。
- 决策：这棵树仅承担当前官方 standalone RoboTwin/ACT 验证；不把旧 `RLinf_support` 内容覆盖进来，也不在此处移植 RLinf 私有算法。

### SZ-SRC-002 — 锁定源码中的安装、资产与 ACT 合同审阅

- 时间：2026-08-21 17:03 CST
- 远端目录：`/data/chenyiteng/projects/robotwin-native/RoboTwin`
- command files：
  - `local_scripts/remote_commands/shenzhen_inspect_robotwin_official_scripts_20260821.sh`，SHA256 `f45c1708dd7cc089406a92c01e43acb9fc42a397cbfac68b64c7ab78043431ab`；退出码 0。
  - `local_scripts/remote_commands/shenzhen_inspect_robotwin_contracts_20260821.sh`，SHA256 `2283e90f733c22d5135a91f591ebcf92c3bc01c239add763b0d5a396788982ac`；退出码 0。
- 全程只读；打印各文件 hash、行数与源码，没有执行文档或脚本里的安装/下载指令。
- 关键合同：
  - `scripts/_install.sh` 会安装 requirements、PyTorch3D、editable XPolicyLab，修改当前 Python 环境内 SAPIEN/MPLib 两个文件，并 clone CuRobo v0.7.8；它还会调用 `update_xpolicylab.sh`，把 committed pin 更新到当时的 XPolicyLab `origin/main`。因此它不是严格 pin-preserving，也不是完全幂等。
  - 当前 parent commit 的 XPolicyLab pin 为 `c37109c...`；官方脚本审阅时 XPolicyLab main 为 `c07a096...`。ACT 相关文件在这两端的 hash 相同，但若执行 update 仍会造成顶层 submodule gitlink dirty；安装前需明确是否保留 parent pin。
  - `scripts/_download_assets.sh` 通过 HF 下载 `background_texture.zip`、`embodiments.zip`、`objects.zip`，解压后删除三个 zip，再改 embodiment path；属于约 14.96 GB 压缩下载与多处文件写入，不能在未预估展开空间时盲跑。
  - `download_xpolicylab_data.sh <task>` 默认仅取对应 `demo_clean.zip`，规范化为 `data/demo_clean/<task>/aloha_agilex/{data,video,instruction}`，默认保留 archive；不带 task 会发现并下载全部任务，因此本轮绝不省略 task 参数。
  - ACT 原始数据映射已经由 current source 对齐：`bench_name=demo_clean`、`ckpt_name=<task>`、`env_cfg_type=aloha_agilex`、`action_type=joint`。`process_data.py` 会把三相机 resize 到 640×480 后以未压缩 `uint8` 写入新 HDF5，必须先按实际总帧数估算展开输出。
  - 当前 `train.sh` 是单卡、batch 16、chunk 50、KL 10、hidden 512、FFN 3200、LR `1e-5`、`num_epochs=6000`、仅在第 6000 epoch 保存；这是 6000 **epochs**，不是旧网页所写 6000 steps。正式启动前必须另做可审阅的运行包，不能直接执行默认长训。
  - 默认 `demo_clean.yml` 是 50 episode；H100 又处于官方列出的 SAPIEN 采集/评估挂起风险范围。官方预采数据用于训练主线，自采仅先做复制配置后的 1-episode 底层 gate。
- 下一步：先核对 CUDA toolkit/编译链并建立两个独立 conda 环境；在任何资产/任务数据下载前再次确认 URL、压缩体积、展开上界与目标空间。

### SZ-SRC-003 — 官方三项基础资产下载与解压

- 时间：2026-08-21 17:37–18:39 CST；已完成。
- 远端目录：`/data/chenyiteng/projects/robotwin-native/RoboTwin`。
- command file：`local_scripts/remote_commands/shenzhen_download_robotwin_assets_20260821.sh`。
- SHA256：`847638815DEADB7EF204F0186562257A04655CB04BC1ED624D80A4DBEBF0FEB0`。
- 边界：同一 shell 显式加载 Mihomo proxy 与 `RoboTwin` env；再次断言双 SHA 和六个目标均不存在；原样调用官方 `scripts/_download_assets.sh`，外围 7200 秒 TERM、60 秒后 KILL 上界。官方脚本成功后会解压并删除三个 ZIP，随后逐目录核对文件数、展开大小、空间与 Git 状态。
- 启动前：`/data` 约 3.2 TiB 可用，assets 约 5.8 MiB；压缩下载精确合计 14,928,324,889 bytes，处于已核实的代理剩余额度内。
- 17:54 只读进度探针：
  - command file：`local_scripts/remote_commands/shenzhen_assets_progress_probe_20260821.sh`；SHA256 `03CB6CF382BE82C8AFDB02804572ACCAD4F44E1F263B5006EA9686DC588795E1`。
  - 第一次本地调用误把已含 `SHA256:` 的指纹传给会自动加前缀的 helper，认证前以 `expected SHA256:SHA256:...` fail-fast；服务器返回的实际指纹仍与固定值一致，没有发出远程命令。改为传入无前缀 base64 后沿同一 Paramiko 路线成功，退出码 0。
  - live process：唯一官方 `_download_assets.sh` 及其 7200 秒 timeout 仍在；没有启动第二个 downloader。
  - live bytes：assets 4.6 GiB；`background_texture` incomplete 2,852,126,720 bytes，`objects` incomplete 1,835,008,000 bytes，`embodiments.zip` 219,859,313 bytes 已完成；`/data` 仍约 3.2 TiB 可用。
- 18:00 复用同一只读探针，退出码 0：唯一 downloader 仍在；assets 6.2 GiB，background incomplete 3,921,674,240 bytes，objects incomplete 2,495,610,880 bytes，两个大文件均继续增长；`/data` 仍约 3.2 TiB 可用。
- 18:08 复用同一只读探针，退出码 0：`objects.zip` 已完成精确 3,737,778,549 bytes，`background_texture` incomplete 为 5,840,568,320 / 10,970,687,027 bytes；唯一 downloader 仍在，assets 9.2 GiB，空间正常。
- 18:17 复用同一只读探针，退出码 0：background incomplete 8,462,008,320 / 10,970,687,027 bytes；assets 12 GiB，唯一 downloader 和空间均正常。
- 18:24 复用同一只读探针，退出码 0：background incomplete 9,227,468,800 / 10,970,687,027 bytes；传输变慢但仍增长，进程/空间正常，不干预。
- 18:30 复用同一只读探针，退出码 0：background incomplete 10,150,215,680 / 10,970,687,027 bytes，约剩 820 MB；唯一 downloader 和空间正常。
- 18:35 复用同一只读探针，退出码 0：background incomplete 10,674,503,680 / 10,970,687,027 bytes，约剩 296 MB；不改变路线。
- 最终结果：原任务自然退出 0；没有重启 downloader。local HF metadata 锁定 revision
  `a967b852afa21a9cbf19a198f7e653109042e87c`。compressed bytes 精确合计
  `14,928,324,889`；官方脚本成功解压后删除三个 ZIP，并更新 6 处 CuRobo template path。
- 最终 inventory：`assets/background_texture` 约 11 GiB / 11,000 files，`assets/embodiments`
  约 901 MiB / 229 files，`assets/objects` 约 4.4 GiB / 9,368 files；assets 总计约 16 GiB。
- 终态 source probe：`local_scripts/remote_commands/shenzhen_assets_revision_probe_20260821.sh`，
  SHA256 `FC5B95CA56D0832CF4D43E338CAC25A0BB3EA751A5E0F5E0905BC9FAC1CBEB8E`，退出码 0。
  顶层仅保留 installer 按设计造成的 `M XPolicyLab`，没有未知 tracked 修改；`/data` 仍约 3.2 TiB 可用。

### SZ-SRC-004 — official `adjust_bottle/demo_clean` 单任务数据与 schema inventory

- 时间：2026-08-21 18:40–18:42 CST；已完成。
- 下载 command file：`local_scripts/remote_commands/shenzhen_download_adjust_bottle_demo_clean_20260821.sh`；
  SHA256 `D31259A47C105443E0812F2F6569E4586113AC471EFB051A1C500119DDC59AEA`；退出码 0。
- 精确边界：只向 official downloader 传 `adjust_bottle`，没有省略 task，也没有下载其他任务。
- source archive：`data/download_cache/dataset/adjust_bottle/demo_clean.zip`，`293,694,934` bytes；按计划保留。
- extracted root：`data/demo_clean/adjust_bottle`，约 423 MiB；恰好 50 HDF5、50 MP4、50 instruction
  JSON，episode IDs 连续 `0..49`。
- inventory Python：`local_scripts/remote_commands/shenzhen_inventory_adjust_bottle_hdf5_20260821.py`；
  SHA256 `1F33FAAE0835337E46A480A03348AE15A7C0891A62238DE88C2232BD716805BE`。
- runner：`local_scripts/remote_commands/shenzhen_run_adjust_bottle_inventory_20260821.sh`；
  SHA256 `FE43166BACD0CA4ADC57A001350372597A43D85728801F5416CBE859C37E45F3`；退出码 0。
- schema 结果：总帧数 7,188，最长 episode 173；首条三相机
  `cam_head/cam_left_wrist/cam_right_wrist` 均可解出 `240×320×3 uint8`；state/action 均为 14D，
  各字段 episode 内帧数一致。source HDF5 总量约 0.407 GiB。
- preprocess 空间估算：current ACT 会把三相机转为 `640×480 uint8` 并写未压缩 HDF5；仅 image
  payload 下界为 18.509 GiB。`/data` 空间充足，允许进入已授权 preprocess。

### SZ-SRC-005 — TianxingChen official ACT checkpoint 定点下载

- 时间：2026-08-21 18:50–18:52 CST；已完成。
- command file：`local_scripts/remote_commands/shenzhen_download_official_act_adjust_bottle_20260821.sh`；
  SHA256 `0902719E9E35C8C4566ED1B307E3CB952571022FACD848251CF07A8206A4668E`；退出码 0。
- repo/revision：`TianxingChen/RoboTwin2.0` dataset repo @
  `a967b852afa21a9cbf19a198f7e653109042e87c`；`allow_patterns` 只有目标 leaf 的两个文件，
  `max_workers=1`，没有拉整个 HF dataset。
- target：`/data/chenyiteng/models/robotwin2-hf-a967b852/act_ckpt/act-adjust_bottle/demo_clean-50`；
  启动前整个 model target 不存在，避免覆盖。
- `policy_last.ckpt`：335,907,442 bytes；SHA256
  `edfb0125103e67465cc2852ea1683acc2ce1060d02b81ba6f1113b5420b40690`；校验 OK。
- `dataset_stats.pkl`：10,664 bytes；SHA256
  `a79964a7cce7a02cd172fb669ef12c8c2f5cedd2c4bd11adfd86c4d90cb239c4`；校验 OK。
- artifact 总量 335,918,106 bytes；target 连同 HF metadata 显示 321 MiB；下载后 `/data` 仍约
  3.2 TiB 可用。固定 `/tmp` checksum 文件在执行前审阅中改为 stdin pipe，避免共享 `/tmp` symlink/竞态。
- 信任边界：current ACT loader 会 `pickle.load(dataset_stats.pkl)`；真实 debug/eval 前需由集中批准显式
  接受 official release 的反序列化边界。
