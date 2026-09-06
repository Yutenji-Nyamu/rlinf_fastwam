# RLT / DSRL formal 只读训练分析流水账（2026-08-24）

范围：刷新当前训练、分析指标与产物、绘制全历史训练/资源图。服务器只读；没有进程控制、代码修改、安装、
下载 checkpoint 或视频。

## A01：读取当前单一事实源

- 完整读取根 `PROJECT_CONTEXT.md`、`HANDOFF.md` 与专题
  `docs/rlinf-shenzhen-rlt-dsrl-port/00_INDEX_AND_MIGRATION_PLAN.md`。
- 采用现场优先：旧文档中的 Step 5/12 只作起点，不作为当前状态。

## A02：首次 live 状态

命令入口：

```text
local_scripts/remote_commands/shenzhen_rlt_v3_dsrl_v2_live_status_20260824.sh
```

结果：2026-08-24 10:14 CST，两个 wrapper 均 alive；RLT driver停在 Step25 save，DSRL已到 Step145；
GPU 4/5约17 GiB且util 0，GPU 6/7约34 GiB且在工作；host available约1.9 TiB。

## A03：统一产物、错误、checkpoint 与资源盘点

命令入口：

```text
local_scripts/remote_commands/shenzhen_rlt_dsrl_live_inventory_20260824.sh
```

结果：

- RLT Stage 1 `global_step_2000`约22,097,680,569 bytes。
- RLT Stage 2 `global_step_25`仅创建不完整目录；无有效 shard。
- DSRL `global_step_65/130`各约33,593,519,684/685 bytes。
- Traceback/OOM/WorkerCrashed/nonfinite均0；`ERROR`模糊匹配来自oneDNN信息文字。
- `/data`约3.04 TB可用，主机RAM充足。

## A04：本地容量与轻量下载

- 目标：
  `C:\Users\86136\Documents\rl\docs\rlinf-shenzhen-rlt-dsrl-port\evidence\formal-live-analysis-20260824\`
- 下载前 C: 可用约36.01 GiB。
- 只下载12个轻量文件：三阶段driver/resource/resolved、三个TensorBoard event；合计约6.46 MB。
- 未下载21/63 GiB checkpoint、视频或Ray全量日志。

使用固定host-key的Paramiko SFTP，密码只注入当前进程，未写入脚本或文档。

## A05：RLT TensorBoard与checkpoint stall收窄

命令入口：

```text
local_scripts/remote_commands/shenzhen_rlt_tensorboard_resource_summary_20260824.sh
local_scripts/remote_commands/shenzhen_rlt_stage2_checkpoint25_stall_audit_20260824.sh
local_scripts/remote_commands/shenzhen_rlt_checkpoint25_code_and_files_20260824.sh
local_scripts/remote_commands/shenzhen_rlt_checkpoint_smoke_compare_20260824.sh
```

结果：

- Stage 2 train scalar只有24点；eval scalar 1点，fixed20=`0/20`。
- 01:39:19 CST以后RLT GPU util的3,000多个observer点全部为0。
- 两rank进入`dcp.save`的FSDP optimizer-state提取阶段，尚未写第一个DCP shard。
- 最强相关差异是formal首次save发生在`update_step=0`；只读证据不足以写成唯一根因。

## A06：DSRL专线核对

只读核对完整step、评估、checkpoint sidecar与资源窗口。10:19 CST完整到Step147：

- global resident 4,303，累计planned updates 76,460。
- 最新fixed12为11/12；最近3次均值86.11%。
- Step65/130 sidecar分别记录update step 34,020/67,900，strict-resume私有状态齐全。
- 稳态GPU约34 GiB/card；checkpoint约36.6 GiB；启动短峰约61.75 GiB。

## A07：离线解析和制图

命令：

```text
C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe \
  local_scripts\render_shenzhen_rlt_dsrl_formal_live_20260824.py
```

脚本检查：

- RLT Stage1必须恰好连续1–2000。
- RLT Stage2与DSRL只保留完整、连续的Metric Table。
- warm-up中缺失的optimizer量保持空值，不补0。
- 5/20-step rolling只在窗口完整后开始；fixed eval只画真实离散点。
- resource使用observer原始10秒点，不插值。

产物：3份逐步CSV、1份RLT eval CSV、`summary.json`、5张独立PNG和渲染stdout。

## A08：QA

- 五张PNG均由Pillow成功打开并做人工视觉核验。
- 图中文字、图例、坐标与结论可读；未把RLT alive误写为训练正常。
- 主文档使用Typora兼容Markdown；本轮没有数学公式定界符问题。
- 服务器仍只读，没有停止RLT、干预DSRL或修改shared Ray。

## A09：最终现场刷新

再次运行A02同一只读状态脚本。2026-08-24 10:35 CST：

- RLT仍停在Step25 checkpoint，GPU4/5约17.4 GiB且util 0；没有新shard、exit marker或错误。
- DSRL已完整到Step150/200；Step147--150 train success依次为100%、75%、50%、75%，
  global resident为4,391，GPU6/7约34.1/34.4 GiB且继续工作。
- host available约1.9 TiB，旧actor数0；两个wrapper均alive。
- 原始现场输出保存为`formal-live-analysis-20260824/live_status_final.txt`。

本次刷新仍未控制进程或改动服务器文件。

## A10：RLT v4 Step250终态轻量收束

- 现场确认Stage2 v4 `250/250`自然完成，`exit_code=0`，开始/结束UTC为
  `03:42:09/11:35:51`，墙钟`28,422s`；`global_step_250`存在。
- 通过固定host-key Paramiko只下载10个小文件：driver/metrics/resource、TensorBoard event、resolved YAML、
  command/manifest和起止/exit marker；原始总量约5.2 MiB，未下载checkpoint权重。
- 解析得到连续Step1--250；在线AC从Step137开始，最终train=`7/8`、fixed20=`18/20`、
  `update_step=104000`、actor/critic loss=`-0.0900/0.0014`。
- 原始资源显示GPU4/5峰值`21.4/21.7 GiB`、平均利用率`25.2%`、host available最低`1309 GiB`，
  OOM/OOM-kill=0。
- 生成三张主图、逐步CSV、fixed20 CSV、summary JSON与README；ZIP为
  `exports/shenzhen_rlt_stage2_v4_final250_light_evidence_20260824.zip`，875,060 bytes、18项。
