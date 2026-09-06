# 2026-08-23 深圳 GRPO v2 训练现场刷新流水

范围：按用户要求，以和上一轮相同口径只读刷新既有 GRPO formal-100 v2。检查进程、最新完整
step、训练/评估标量、checkpoint/日志/视频清单、GPU 与主存连续遥测，并在 Windows 本地生成轻量
CSV、JSON 和 PNG。不会停止、重启或修改训练，不改远端配置，不下载 checkpoint 或视频正文。

## REFRESH-001 — 上下文与权限边界

- 完整读取 `PROJECT_CONTEXT.md`、`HANDOFF.md` 与专题单一事实源
  `00_INDEX_AND_IMPLEMENTATION_PLAN.md`。
- 上一版图表冻结于完整 Step 27，最后只读续点为 2026-08-23 10:50 CST 的完整 Step 29；两者仅作
  定位线索，本轮所有动态结论必须由服务器现场重新确认。
- 日常账号使用 `chenyiteng`，固定 host key 的 Paramiko 密码通道；密码只注入当前进程。
- 当前授权仅覆盖只读检查和本地证据整理；不含任何进程控制、训练重启、配置修改或远端清理。

## REFRESH-002 — 本地 Python 入口纠正

- 第一次调用使用 PATH 中的 WindowsApps `python.exe`，本地在建连前以“系统无法访问此文件”失败；远端
  命令没有执行。
- 读取 Codex workspace dependency 路径后，固定改用
  `C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`。
  没有更换 SSH helper、host key、账号或认证路线。

## REFRESH-003 — 13:32 CST 最新状态

- command file：`local_scripts/remote_commands/shenzhen_grpo_v2_final_status_20260823.sh`，2,213 bytes，
  SHA-256 `aa461b4d20d4aa1053afb78438e7816077164096f049be65e230313b71fbcfe5`。
- 固定 host-key Paramiko、账号 `chenyiteng`，exit 0，marker `SZ_GRPO_V2_FINAL_STATUS_OK`。
- driver/observer均alive；精确 4 actor + 4 rollout + 4 env、1 GCS、1 raylet。
- 最新连续完整 Step 36，随后 Step 37 rollout 到 `1/4`。Step 36 为512 trajectories，
  success/KL/clip/grad/ratio-abs=`0.9589844/0.017/0.062/10.562/0.073`；fatal/nonfinite=`0/0`。
- GPU4–7即时约`61.2/62.4/61.5/61.4 GiB`；cgroup约`1.647 TiB`、swap约`5.95 GiB`、host
  MemAvailable约`386 GiB`，memory events全0。checkpoint/eval计数=`3/12`。

## REFRESH-004 — 13:33 CST 产物与连续资源只读盘点

- command file：`local_scripts/remote_commands/shenzhen_grpo_v2_curve_artifact_audit_20260823.sh`，
  SHA-256 `e66c20bd24d855253f4fd361af3814f5cc119e60e7644b752ae3b815490f7400`。
- exit 0，marker `SZ_GRPO_V2_CURVE_ARTIFACT_AUDIT_OK`。run根52 GiB；
  `global_step_10/20/30`各约18 GiB；train/eval MP4=`580/12`，bytes=`146,668,706/1,622,557`。
- `metrics/driver/resource`=`207,762/279,278/71,650` bytes，TensorBoard event=`88,940` bytes；
  `/`与`/data`可用约`234 GiB/3.0 TiB`。
- resource CSV共843行；观测全段峰值cgroup=`1663.6 GiB`、最高单卡=`74.1 GiB`。日志fatal与
  metrics非有限token均为0。
- 该旧command file的粗 `pgrep -f` GCS/raylet计数仍会匹配当前command-shell正文，显示2/2；不采用
  该值。REFRESH-003的进程名精确计数1/1为本轮权威值。

## REFRESH-005 — 轻量原件下载

- 下载前 Windows C: free=`39,515,119,616` bytes；远端四个目标合计不足0.7 MiB，不构成磁盘风险。
- downloader：`local_scripts/download_shenzhen_grpo_v2_live_step36_20260823.ps1`，SHA-256
  `6e25fa81c0f5cef3aa28945d4224324a1a3ed13e5dcdEBc73caf53368df288b5`；一次密码注入、四次固定
  host-key SFTP，全部exit 0，不覆盖已有文件。
- 下载到`evidence/grpo_v2_live_step36_20260823/`：`metrics.log` 207,762 B、`resource.csv`
  71,650 B、`driver.log` 279,278 B、TensorBoard event 88,940 B。没有下载checkpoint或视频。

## REFRESH-006 — 本地解析、制图与结果

- renderer：`local_scripts/render_shenzhen_grpo_live_step36_20260823.py`，最终 SHA-256
  `c4c8bc48f6f75ef3227ba201aa521976004ed030f028ad6397e5e31371f3d6b1`，使用bundled Python/Pillow，
  exit 0；要求连续Step1–36、所有必需字段finite、step/分钟资源对齐误差不超过31秒。
- 生成2个CSV、1个summary JSON与3张手机可读PNG；目视核对标题、坐标、图例、fixed-eval marker与
  Step36终点，无裁切或数据轴错位。
- GRPO Step1–36均值/最新5步/最新10步=`87.81/94.49/93.79%`；PPO同轴为
  `86.17/90.82/89.94%`。fixed64 GRPO Step10/20/30=`57/60/62`，PPO=`58/62/58`。
- GRPO Step36 KL/clip/grad/ratio-abs=`0.017/0.062/10.562/0.073`；全段finite。
- GRPO/PPO同轴median whole-step=`1378.85/1551.55 s`、rollout=`1352.35/1518.40 s`、actor=
  `22.80/22.94 s`；速度差几乎全部在rollout。
- cgroup对齐Step1/10/20/30/36=`302.8/892.2/1382.8/1580.5/1658.4 GiB`；Step21–30与31–36
  拟合增速=`14.01/6.87 GiB/step`，增长放慢但未平台化。

## REFRESH-007 — 13:38 CST 主存归因与换页活动

- command file：`local_scripts/remote_commands/shenzhen_grpo_v2_memory_breakdown_step36_20260823.sh`，
  SHA-256 `7781f9c3b32be7ee3a0ad5183c263ed5e5e42bae394733ff362409a1c89ea8cd`。
- 固定host-key Paramiko、`chenyiteng`，exit 0，marker
  `SZ_GRPO_V2_MEMORY_BREAKDOWN_STEP36_OK`。所有12个worker及GCS/raylet仍alive。
- 4个EnvWorker合计PSS约`1540.4 GiB`，占即时cgroup `93.55%`；4 actor/4 rollout PSS约
  `33.3/17.8 GiB`。cgroup anon/file/shmem约`1580.3/61.2/23.2 GiB`，主体是EnvWorker匿名私有内存。
- 即时cgroup=`1646.6 GiB`、host available约`385 GiB`、swap=`5.95 GiB`；3个`vmstat`样本
  `si=so=0`，cgroup与host memory PSI 10/60/300秒均0，memory events仍全0。
- 一分钟GPU利用率四卡均值全段平均/中位/P90/最近15分钟=`13.1/1.5/40.0/13.8%`；任一卡达到80%
  的分钟占24.1%。这是仿真等待与推理burst交替，不是driver空转。

完整解释与三张图见上级
[`20_GRPO_STEP36_LIVE_METRICS_AND_RESOURCE_REFRESH_20260823.md`](../20_GRPO_STEP36_LIVE_METRICS_AND_RESOURCE_REFRESH_20260823.md)。

## REFRESH-008 — 13:44 CST 最终只读续点

- 再次执行REFRESH-003同一已核command file，exit 0、marker正常。
- 最新完整表仍为Step36；Step37 rollout已到`3/4`。全部12个worker、driver/observer、GCS/raylet仍
  alive，fatal/nonfinite=`0/0`。
- GPU4–7即时约`62.7/63.8/63.3/60.7 GiB`；cgroup约`1647.8 GiB`、host available约
  `383.8 GiB`、swap约`5.95 GiB`，memory events仍全0。本轮到此停止只读取数，不等待Step37完成。

## REFRESH-009 — 本地交付QA

- renderer `py_compile` exit 0；派生metric/resource分别为36/843行，summary完整step=36，fixed-eval
  精确为`0.890625/0.9375/0.96875`。
- 三张PNG尺寸为`1440×1360`、`1440×1950`、`1440×2480`，均已目视核对。
- 主文档、流水账、SSOT与HANDOFF中的本地Markdown链接全部存在；新增文档无UTF-8 replacement char，
  无Typora旧公式分隔符。
- 本目录当前不是Git worktree，`git diff --check`无法使用；改用逐文件尾空白/编码/链接/解析与图片QA，
  没有据此创建Git仓库或改变现有目录状态。
