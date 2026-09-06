# 深圳 DVAC 离线分析执行流水账（2026-08-22）

范围：只读分析既有 π0 fixed64 与 Fast-WAM adjust_bottle P1 telemetry；分析产物写入全新目录。
日常账号：`chenyiteng`；固定 SSH host-key：`SHA256:qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY`。
凭据仅注入当前 Paramiko 进程，不写入文件或本文。

## 00. 本地分析器 review 修复与契约测试

- 本地实现：`local_scripts/analyze_shenzhen_dvac_observation.py`；测试：
  `local_scripts/test_analyze_shenzhen_dvac_observation.py`。
- review 后一个连贯批次完成：run/cohort 隔离位置基线；pre-success 主汇总与 all-query descriptive
  伴随表；`query_state_phase` 与 Fast-WAM 未执行 tail 的 `UNLABELED` 边界；terminal success
  action/frame；episode-first phase bootstrap；manifest 中真实 `action_num_train_timesteps` 优先且必验；
  `success_before` 必填；empty CSV 固定 schema；默认依赖在 output 创建前 preflight；可选官方 seed-map
  只读 sidecar merge。
- 本地命令：

  ```powershell
  & 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
    -m py_compile local_scripts/analyze_shenzhen_dvac_observation.py `
      local_scripts/test_analyze_shenzhen_dvac_observation.py
  & 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
    -m unittest -v local_scripts/test_analyze_shenzhen_dvac_observation.py
  ```

- 结果：`py_compile` exit 0；6 个测试全部通过，耗时 4.147 秒。合成数据覆盖真实
  `π0 H50/M4`、`Fast-WAM H32/C24/M10`、两个独立 run、post-success、unexecuted tail、preferred/legacy
  manifest、terminal boundary、seed sidecar、empty schema 和依赖 preflight。

## 01. 真实 source / output / runtime 只读 preflight

- 精确 source：
  - π0：`/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1`
  - Fast-WAM payload：`/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2`
  - Fast-WAM video-root：`/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2`
- 预定全新 output：
  `/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-adjust-p1-v1`。
- 预定持久 packet：`/data/chenyiteng/results/dvac-observation/packets/sz-dvac-analysis-v1`。
- 本地 command file：
  `local_scripts/remote_commands/shenzhen_dvac_offline_analysis_preflight_20260822.sh`。它只读确认身份、
  三个 source 存在、output/packet 不存在、marker 与精确 bytes，并在既有 Fast-WAM venv 中 import
  `numpy/pandas/PIL/matplotlib/cv2`；不创建目录、不安装包。

（现场结果待执行后逐项追加。）

### 01.1 现场结果（23:32 CST）

- 固定 host-key Paramiko/password 路线首次连接成功，无认证或 pre-auth 重试；远端身份为
  `uid=1003(chenyiteng)`，主机 `admin`。
- 三个精确 source/video-root 均存在；预定 output 与 packet 均不存在。
- π0 marker：4 份 episode CSV、4 份 query CSV、4 份 rank NPZ；Fast-WAM marker 包含
  `run_manifest.json`、`queries.csv`、`episodes.csv` 与 query NPZ。
- `du -sb`：π0 source=`40,760,738` bytes；Fast-WAM payload=`13,561,998` bytes；video-root=
  `5,004,047` bytes。
- 首选既有 Python：`/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python`。import preflight
  全通过：NumPy `1.26.4`、pandas `2.2.3`、Pillow `12.0.0`、matplotlib `3.10.9`、OpenCV
  `4.11.0`。没有安装或修改任何包，也无需切换 RLinf venv。
- command file 本地 bytes=`1,787`，SHA-256=
  `c6ab01927dac1e3fc37fd91391e3c8b466c6981216ce7999f7a68f77a1d0fc72`；remote exit 0，
  终止标记 `PREFLIGHT_OK`。

## 02. 持久 packet 创建与分析器上传

- 分析器本地 bytes=`109,396`，SHA-256=
  `009b4ea41ab9ebc974b778e46156ee5fc4f0bf843e0fd703d3051715675ff152`。
- packet 创建、SFTP 普通上传、remote SHA 与 `py_compile` 结果待下步逐项追加；不会覆盖既有路径。

### 02.1 现场结果（23:34–23:35 CST）

- command file `shenzhen_dvac_offline_analysis_packet_create_20260822.sh` 对 packet 路径再次执行
  `test ! -e` 后，仅以 `install -d -m 0755` 创建该精确目录；exit 0。
- 通过同一固定 host-key Paramiko 的 SFTP `put` 把本地分析器普通上传为
  `/data/chenyiteng/results/dvac-observation/packets/sz-dvac-analysis-v1/analyze_shenzhen_dvac_observation.py`；
  目标此前不存在，没有覆盖。
- remote `stat` bytes=`109,396`；SHA-256 精确等于本地
  `009b4ea41ab9ebc974b778e46156ee5fc4f0bf843e0fd703d3051715675ff152`。
- 首选 Fast-WAM venv 对 remote script 执行 `python -m py_compile` 与 `--help` 均 exit 0。
- 真实 Fast-WAM `run_manifest.json` 明确为 `action_num_train_timesteps=1000`，legacy
  `num_train_timesteps` 不存在；因此执行器将采用 preferred key，不走兼容 fallback。
- upload verify 终止标记 `UPLOAD_VERIFY_OK`；未触碰两个 source、video-root 或 output。

## 03. 唯一一次真实默认离线分析

- 精确命令由 command file
  `local_scripts/remote_commands/shenzhen_dvac_offline_analysis_execute_20260822.sh` 保存；使用已核 SHA 的
  packet 分析器和既有 Fast-WAM venv，同时传入两个精确 source、一个 video-root 与唯一全新 output。
- 不传 phase annotation 与 official seed-map：本轮目标是先生成未标注首版；不回写 payload。
- 保留默认 plots/storyboards；不安装包、不启动 GPU/Ray/仿真或模型推理。

### 03.1 执行结果（23:35 CST）

- 唯一一次命令从 `2026-08-22T15:35:48+00:00` 运行到 `15:35:58+00:00`；exit 0，终止标记
  `ANALYSIS_EXECUTE_OK`。`/usr/bin/time`：wall=`9.83s`、user=`12.73s`、sys=`7.75s`、
  max RSS=`342,400 KiB`。
- stderr 只有 matplotlib 3.10 对 `boxplot(labels=...)` 的未来弃用告警；不是数据、绘图或运行失败，
  19 张 PNG 均随后通过 Pillow 完整解码验证。本轮不为消除告警重跑或覆盖 output。
- summary 主计数：2 sources、2 groups、80 episodes、336 queries、43,520 horizon rows、
  1,852 条 Fast-WAM executed-action/frame rows。pre-success 主样本为 294 queries；42 条
  post-success query 全来自 π0，只进入 all-query descriptive 伴随表。Fast-WAM post-success 为 0。
- 两个位置基线按 run/cohort 明确隔离：
  - π0 fixed64：64 episodes，42 success / 22 failure，256 queries；
  - Fast-WAM adjust P1：16 episodes，16 success / 0 failure，80 queries。
- 三个数值恒等式最大绝对误差分别为 `2.22e-16`（`y=b+r`）、`1.78e-15`
  （raw four-way）与 `4.44e-16`（`R=S+I`）；均远低于 `1e-10` 验收线。
- 两个 source 的 `max_abs_final_chain_vs_model_action=0.0`。Fast-WAM reconstruction 使用真实 manifest
  的 `action_num_train_timesteps=1000`；未启用 seed-map sidecar。
- action/video 合同：1,852 行全为 `h<=23`；16 条 terminal-success action 均被标记，且可保留对应
  terminal post-action frame。Fast-WAM 未执行 tail 的 phase 全为 `UNLABELED`。
- 本轮未提供独立 phase CSV，因此 `phase_episode_metrics*.csv` 与 `phase_summary*.csv` 均为 0 行，
  但分别保留固定 19/14 列 schema；这是预期首版，不是丢数据。annotation template 已生成。

## 04. 输出验收与下载前大小

- 精确 output：
  `/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-adjust-p1-v1`。
- `du -sb` 总计 `48,494,941` bytes，35 个文件：15 个 CSV 共 `44,748,986` bytes、1 个 JSON
  `9,994` bytes、19 个 PNG 共 `3,711,385` bytes。
- 最大表：`query_horizon.csv`=`41,898,068` bytes（43,520×51）；其余高信息表包括
  `query_metrics.csv`=`596,612` bytes（336×67）、`fastwam_action_frame_metrics.csv`=
  `1,954,648` bytes（1,852×43）、pre/all episode 各 80×34、pre/all outcome 各 10×13。
- 图像验收：15 张统计 figure、3 张 representative query strip、1 张 Fast-WAM storyboard；19 张 PNG
  全部可解码，总像素 `32,045,170`。`storyboard_index.csv` 为 1 行且状态 `created`。
- 为遵守 C 盘下载前确认，本轮没有下载，也没有在服务器落 archive。仅把 tar+gzip 写向 stdout 后由
  `wc -c` 计数：完整 35 文件包预计精确 `9,109,430` bytes（约 8.69 MiB）；去掉可重建但最关键的
  `query_horizon.csv` 后 compact 包为 `3,722,112` bytes（约 3.55 MiB）。由于完整包也小于 9 MiB，
  建议最终下载完整包；先由主协调向用户报告本地目标路径与 C 盘 live free space。
- postflight assertions 全部通过：counts、group 隔离、pre/all query 守恒、identity、`h<24`、16 条 terminal
  action、unexecuted-tail phase、CSV schema、storyboard 与 PNG decode。

## 05. 本轮 command files 与 SHA-256

| command file | SHA-256 | 结果 |
|---|---|---|
| `shenzhen_dvac_offline_analysis_preflight_20260822.sh` | `c6ab01927dac1e3fc37fd91391e3c8b466c6981216ce7999f7a68f77a1d0fc72` | exit 0 |
| `shenzhen_dvac_offline_analysis_packet_create_20260822.sh` | `f07f8c39590d3f7e84dcca83e3d3e6ab3f012d5e2533d5d0a0cf0ccdd0cfda0c` | exit 0 |
| `shenzhen_dvac_offline_analysis_upload_verify_20260822.sh` | `e9e5d1c7bf0253d15aa535135773572230087a480db452d840e4a716c9c1e86d` | exit 0 |
| `shenzhen_dvac_offline_analysis_execute_20260822.sh` | `fc156c12d30ec052273b9b0150cb6b637f5251a5a018222467a4bd69796fd66a` | exit 0 |
| `shenzhen_dvac_offline_analysis_postflight_20260822.sh` | `f1289e1a2c588b062df4223b32ed32b7f135ee8de26ab47e34911853b8187622` | exit 0 |
| `shenzhen_dvac_offline_analysis_package_size_20260822.sh` | `ebb3bc6166c1cdfff56ceb2a43c04d073f19acc979133ffdd3f19b33d4b5f7a0` | exit 0 |

本轮服务器写入仅有持久 packet 中的分析器/`__pycache__` 与上述唯一新 analysis output；两个 source、
Fast-WAM official video-root、训练/推理进程、Python 环境和 Git worktree 均未修改。

## 06. 完整派生包下载与本地核验

- 下载前已在聊天给出精确目标和容量：远端完整压缩约8.69 MiB；本地目标为
  `C:\Users\86136\Documents\rl\docs\rlinf-shenzhen-pi0-ppo-rlt\evidence\dvac-analysis-p1-fixed64-20260822`，
  当时C盘可用`37.31 GiB`。
- 远端archive command file：
  `local_scripts/remote_commands/shenzhen_dvac_offline_analysis_archive_20260822.sh`，bytes=`580`，
  SHA-256=`53e6d38d429af6d8245897c6bb9465b113ed178533ebe768e16f428b72c737da`。它再次确认output为目录、
  35个文件且archive不存在，然后执行：

  ```bash
  cd /data/chenyiteng/results/dvac-observation/analysis
  tar -czf sz-dvac-pi0-fixed64-fastwam-adjust-p1-v1.tar.gz \
    sz-dvac-pi0-fixed64-fastwam-adjust-p1-v1
  ```

- remote archive：`9,111,207` bytes，SHA-256=
  `1900d1069e4363ab7b8b6a96c2b8491e657a79074ad21e412b59a69314ce6acd`；SFTP下载由
  `local_scripts/download_shenzhen_dvac_analysis_20260822.ps1`完成，密码只存在于该交互进程。
- 本地archive同SHA；解压到上述全新目录，35 files、文件payload合计`48,470,365` bytes、19 PNG。
  Pillow逐张`verify()`为`PNG_DECODE_OK 19`。远端source、output和archive均保留，没有删除或覆盖。

## 07. `move_stapler_pad` 独立phase标注分析

- phase标注先于本任务DVAC读取完成，来源为两条预注册代表视频的raw 12-frame contact sheet；本地CSV：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/fastwam-move-stapler-phase-candidates-20260822/phase_annotations_v1.csv`，
  2,158 bytes，SHA-256=`a949e73c060aef534e37c9dcced1306cb17b684a5a8773532dcbb8b5e8dc8afc`。
  它包含success `episode_id=4`的q0--6与failure `episode_id=6`的q0--8共16行；failure后续不确定段
  故意保持`UNLABELED`，不根据DVAC补标签。
- 目标packet文件确认absent后，经固定host-key Paramiko SFTP普通上传到
  `/data/chenyiteng/results/dvac-observation/packets/sz-dvac-analysis-v1/fastwam_move_stapler_phase_annotations_v1.csv`；
  remote SHA与本地一致。source payload、official video root和全新output分别为：
  `/data/chenyiteng/results/dvac-observation/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1`、
  `/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1`、
  `/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-fastwam-move-stapler-p2-phase-v1`。
- 使用已核packet分析器`009b4ea4...ff152`及既有Fast-WAM venv，显式清空`CUDA_VISIBLE_DEVICES`，
  保留默认plots/storyboards；唯一一次分析从`2026-08-22T15:56:04+00:00`到`15:56:12+00:00`，
  exit0。wall=`8.18s`、max RSS=`336,772 KiB`；只有既知matplotlib `labels`弃用告警，没有安装、GPU、
  Ray、模型或仿真动作，也未回写source/video。
- 验收：1 source/group、16 episodes（11 success/5 failure）、162 queries、10,368 horizon rows、
  3,741 executed-action/frame rows；phase CSV命中16 query / 365 executed-action rows，派生18条
  phase-episode rows与60条phase-summary rows。两个预注册代表episode storyboard均为`created`；
  共28 files、12 PNG且Pillow全部解码，output总计`17,591,880` bytes。
- 数值/合同：三个分解恒等式最大绝对误差分别为`4.44e-16`、`1.78e-15`、`8.88e-16`；所有tensor
  finite，`x_next=x_chain[1:]`且final-chain/model-action误差0。首次只读postflight错误假定
  `storyboard_index.csv`含`episode_id`，实际权威字段是`episode_key`，因此只读验证器报KeyError；只将
  验证器改为从`episode_key`解析4/6后复跑通过，没有重跑或覆盖分析output。
- command files：preflight `177baae1...70fd`（1,034 bytes）、execute `25accd8f...72cb`
  （1,343 bytes）、schema probe `0368eaa2...088a`（335 bytes）、最终postflight
  `5211a0cd...ef1c`（2,744 bytes）。

### 07.1 outcome、phase、代表query与压缩大小只读提取（2026-08-23）

- 所有outcome都是16个episode先各自聚合后再按success/failure汇总；`success-minus-failure`：
  - `S_std`：success/failure=`0.05420/0.65330`，差`-0.59910`，95% bootstrap CI
    `[-1.01473,-0.12344]`；
  - `|I_std|`：`0.66110/0.80331`，差`-0.14221`，CI`[-0.31950,0.06987]`；
  - `|R_std|`：`0.84849/1.18366`，差`-0.33517`，CI`[-0.65356,0.04360]`。
  只有本块`S_std`差的区间未跨0；所有结果仍是本固定块观察性关联，不是DVAC因果或训练收益。
- phase结果只来自两个预注册代表episode、18条`phase_episode_metrics`，不可冒充16-episode phase统计。
  `action_phase`下`S_std / |I_std|`（括号内95% episode bootstrap CI）为：
  - MOVING / approach：`-0.26336 [-0.59209,0.06537] / 0.56041 [0.48859,0.63224]`；
  - OPERATING / grasp-contact：`-0.38197 [-0.46577,-0.29817] / 0.52221 [0.47802,0.56641]`；
  - MOVING / transport-or-rotate：`0.22916 [0.03884,0.41949] / 0.58061 [0.45030,0.71093]`；
  - OPERATING / place-release：`0.66277 [0.36785,0.95769] / 0.88817 [0.71013,1.06622]`；
  - TRANSITION / verify：`0.28049 / 0.47843`，仅1个episode，不给CI。
- 典型success ep4的逐query `S_std(L3)` q0--6为
  `[0.065,-0.466,-0.017,0.094,0.368,0.460,-0.579]`；典型failure ep6 q0--16为
  `[-0.616,-0.568,-0.361,-0.235,0.760,0.079,-0.341,2.596,0.618,2.645,0.564,5.352,-0.175,1.620,1.067,2.777,1.113]`。
  failure q9--16虽可计算DVAC，但独立视频标注故意保持`UNLABELED`，不回看数值补phase。
- `L=3`与`L=5`的162-query rank correlation=`0.9591369`。只把完整output以tar+gzip写到stdout并
  `wc -c`，得到`4,084,323` bytes（约3.90 MiB）；没有创建archive。该完整包已经很小，建议原样下载，
  无需删`query_horizon.csv`做compact版本。
- 只读schema probe SHA=`fd073fc8...b6669`；指标/stream-compression command SHA=
  `dd320eec...d1f93`。没有写新分析产物或干扰Fast-WAM/GRPO后台进程。

### 07.2 完整phase派生包（2026-08-23）

- 用户确认下载目标与C盘余量后，在analysis父目录普通创建且不覆盖：
  `/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-fastwam-move-stapler-p2-phase-v1.tar.gz`。
- archive为`4,084,323` bytes，SHA-256=
  `285be632c973256331e060423c0de1235b28c67f27db0924901cc5bc553eb898`；与此前stdout-only压缩计数精确一致。
  source analysis目录和archive均保留，没有删除或覆盖。
- command file：`local_scripts/remote_commands/shenzhen_dvac_move_stapler_phase_archive_20260823.sh`，
  381 bytes，SHA-256=`eee24e1727abefc51cd0012f83749f12e5701399fa7e65c88d93bd8053e370d9`；exit0，
  marker=`PHASE_ARCHIVE_OK`。

### 07.3 本地下载与解码核验（2026-08-23）

- SFTP普通下载到
  `C:\Users\86136\Documents\rl\docs\rlinf-shenzhen-pi0-ppo-rlt\evidence\dvac-analysis-fastwam-move-stapler-phase-20260823.tar.gz`；
  本地bytes=`4,084,323`，SHA-256与远端同为
  `285be632c973256331e060423c0de1235b28c67f27db0924901cc5bc553eb898`。
- 解压到同名全新目录（去掉`.tar.gz`）：28 files、文件payload合计`17,571,400` bytes、12 PNG；
  Pillow逐张`verify()`为`PNG_DECODE_OK 12`。远端source、analysis output与archive均保留。
- phase标注前使用的两张raw contact sheet也已保存在
  `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/fastwam-move-stapler-phase-candidates-20260822/`：success
  `460,873` bytes、SHA-256=`8439f464b0e50ab5b9ac3b6cec3fccbb631483d6937652b01194c92cfb7d3188`；
  failure `478,232` bytes、SHA-256=`1cf55389c64f7639c2e90ca5f900239f55a922d51abbe12f05826b51eddc6f63`。

## 08. π0 fixed-64 + Fast-WAM四任务统一分析

状态：`COMPLETE_EXIT0_DOWNLOADED_AND_VISUALIZED`。

- 等`pick_diverse_bottles`自然完成后，在全新output统一只读π0 fixed-64与Fast-WAM
  `adjust_bottle / move_stapler_pad / turn_switch / pick_diverse_bottles`共5个source；不覆盖前两批分析。
- 精确command file：
  `local_scripts/remote_commands/shenzhen_dvac_all_four_tasks_execute_20260823.sh`，1,971 bytes，SHA-256=
  `62df980d849f54cba586a590b9c63b32fd8b3e32a12816e0065c592f3d56a3f9`。它使用既有已核分析器与Fast-WAM
  venv，令GPU不可见，读取4个official video root；只把既有move-stapler独立phase CSV用于精确匹配行，
  其余任务保持`UNLABELED`。
- 目标output固定为
  `/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-four-tasks-v1`；当前尚未执行，
  不提前把进行中的pick任务当作完整统计集。

### 08.1 五source统一分析exit0

- `pick_diverse_bottles`于`2026-08-23 00:24:45 CST`自然完成12/16；Fast-WAM四任务由此固定为
  64 episodes / 49 success / 15 failure、503 queries/NPZ、1,509 PNG、64完整MP4，三任务fatal=0且
  wrapper/children退出、GPU3释放。随后才执行08节既定command file，没有读取进行中payload。
- 统一分析自然exit0，marker=`SZ_DVAC_ALL_FOUR_TASKS_ANALYSIS_OK`；wall=`28.02s`、max RSS=
  `458,156 KiB`、GPU不可见。唯一输出为08节全新目录；matplotlib只给出既知`labels→tick_labels`弃用告警。
- 验收：5 source/group、128 episodes、759 queries（717 pre-success主样本、42 post-success伴随样本）、
  70,592 horizon rows、11,528 Fast-WAM executed-action/frame rows；输出68 files，`du -sb=94,668,712`
  bytes。三个分解恒等式最大误差分别为`4.44e-16 / 1.78e-15 / 8.88e-16`。
- outcome主结果：π0 adjust、Fast-WAM move与pick的`S_std` success-minus-failure分别为
  `-0.436 / -0.599 / -0.541`且95% CI均不跨0；Fast-WAM turn则为`+0.564`且CI不跨0。
  四块`abs(I_std)`差的CI均跨0。该反号说明S含任务/阶段状态，不是跨任务同号的不确定性标尺。
- Fast-WAM L3/L5 query排名相关为adjust/move/turn/pick=`0.969/0.959/0.946/0.980`。L3位置曲线跨度
  分别为`0.743/0.597/0.564/1.775 log units`；π0为`1.148`，所有MAD floor计数为0。

### 08.2 archive、本地完整包与统一图

- archive command file：
  `local_scripts/remote_commands/shenzhen_dvac_all_four_tasks_archive_20260823.sh`，428 bytes，SHA-256=
  `c096e3d1387490cc06650f7969a9ac87a3e86c9e619bc74f80a247fdfbf63ba5`。远端archive为
  `20,410,342` bytes，SHA-256=`c63bf6fd486cfa1eba2406c64e6a49ff0a473e831a1d1dd623798d5ee1316ac3`。
- 下载前聊天说明目标、压缩/展开大小与C盘`37.26 GiB`余量；SFTP脚本
  `local_scripts/download_shenzhen_dvac_all_four_tasks_20260823.ps1`（1,475 bytes，SHA-256=
  `21d4e524cbe6977d0de35edda87fd6373377f130835b9d3e0c47c79d49747200`）下载同SHA archive。
  解压到`docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dvac-analysis-all-four-tasks-20260823`：archive内容
  68 files / `94,631,848` bytes / 52 PNG，Pillow=`PNG_DECODE_OK 52`。
- 为把跨任务结论放进一张图，本地renderer首次使用bundled Python时因该runtime没有matplotlib而在import
  处退出，未创建文件。窄处理是复用服务器既有Fast-WAM venv：renderer
  `local_scripts/render_shenzhen_dvac_cross_task_20260823.py`，3,942 bytes、SHA-256=
  `094c48931b23e005e378a76a54b9353638717c0d98a170665f36226f96d99687`，上传到既有analysis packet；
  CPU-only command file `shenzhen_dvac_cross_task_figure_20260823.sh`为633 bytes、SHA-256=
  `b98016ae548ce50b5bc5702c895ff51db431a75a08b1a31723477f8400194d82`。
- 统一图自然exit0，157,252 bytes、SHA-256=
  `4f43e6439e80fdbf54f1cc530721edadf49d225f9c4b0ad97f39c4eb3cf0fb57`；单文件SFTP下载到本地完整包的
  `figures/cross_task_outcome_and_position_summary.png`。因此本地工作目录最终为69 files / 53 PNG；
  原archive仍保持分析器原生68文件内容，没有重打或覆盖。
