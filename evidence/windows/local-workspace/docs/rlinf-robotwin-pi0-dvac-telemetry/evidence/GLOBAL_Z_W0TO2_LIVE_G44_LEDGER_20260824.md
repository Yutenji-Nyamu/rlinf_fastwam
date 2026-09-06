# Global-z `[0,2]` g44现场、产物与分析逐指令流水账

日期：2026-08-24  
范围：AutoDL只读现场、轻量SFTP下载、本地离线分析、可视化与文档更新；没有停止、重启或修改训练。
密码只注入当前Paramiko进程环境，不写入文件或本文。

## 1. 前置阅读

按工作区规则完整读取：

```powershell
Get-Content PROJECT_CONTEXT.md
Get-Content HANDOFF.md
Get-Content docs/rlinf-robotwin-pi0-dvac-telemetry/00_INDEX_AND_PLAN.md
```

结果：当前active AutoDL run为global-z `[0,2]` 100-step formal；本轮授权为查看训练、产物与资源，不干预训练。

## 2. 第一次AutoDL只读刷新

复用远程只读脚本：`tmp/idea2_global_z_w0to2_live_refresh_20260824.sh`，SHA256：
`3B153FF7DFF57DD6D9E49477DBF7712F45A1374F7EFBCF8AD3A8CB95FC26DCF5`。

调用形态：

```powershell
$env:SEETA_SSH_PASSWORD = <current-process-only>
& <bundled-python> local_scripts/remote_exec_autodl.py run `
  --command-file tmp/idea2_global_z_w0to2_live_refresh_20260824.sh
Remove-Item Env:SEETA_SSH_PASSWORD
```

结果（`2026-08-24T13:40:02+08:00`）：

- identity=`autodl-container-nekaqbwt43-6ce5babb`，`/root`，UID0；
- wrapper/driver/observer PID=`820640/820644/820645`均alive；六个核心worker均ALIVE；
- 完整到g44，g45 rollout 2/16；g44 success/KL/clip/grad=`88.672%/.032/6.8%/18.527`；
- fatal=0；Traceback 2个，均为known optional CuRobo probe；CUDA/cgroup OOM和OOM-kill均为0；
- run/runtime=`39G/69M`；88 NPZ；g10/g20/g30/g40 checkpoint；
- GPU现场约`25.73/26.33 GiB`；cgroup约`236.8/240 GiB`；host available约833 GiB；
- 数据盘约余630 GiB。

连接与命令均一次成功，没有认证、banner或EOF问题。

## 3. 产物树补充检查

新建只读脚本：`tmp/idea2_global_z_artifact_inventory_g44_20260824.sh`，SHA256：
`F080B2F44675E43EA83E3DCC25CF0286F9559B266303FED543DC5FECC8C28099`。

命令：

```powershell
$env:SEETA_SSH_PASSWORD = <current-process-only>
& <bundled-python> local_scripts/remote_exec_autodl.py run `
  --command-file tmp/idea2_global_z_artifact_inventory_g44_20260824.sh
Remove-Item Env:SEETA_SSH_PASSWORD
```

结果：

- 两rank各44个连续NPZ，最后均为`rollout_step0043.npz`；
- g40 checkpoint含两个约5.20 GB DCP shard和约1.03 MB metadata；
- control trace包含`frames.csv` 13,303 bytes、MP4 15,263 bytes、metadata 870 bytes；
- runtime另有`process_rss.tsv` 53,778,091 bytes、resources CSV约8.74 MB；
- TensorBoard config 7,992 bytes、event 119,020 bytes。

## 4. 轻量快照下载

下载器：`tmp/idea2_global_z_download_live_g44_20260824.py`，SHA256：
`1954F5D61B7CF826F471567EAB9BEB2666BBA1B8212CCDA6271FAB97855FA41F`。

调用：

```powershell
$env:SEETA_SSH_PASSWORD = <current-process-only>
& <bundled-python> tmp/idea2_global_z_download_live_g44_20260824.py
Remove-Item Env:SEETA_SSH_PASSWORD
```

第一次下载15类运行文件与event，`10,638,422 bytes`。产物树检查确认固定抽样control trace很小后，
用`apply_patch`把TensorBoard config和control trace的CSV/JSON/MP4加入同一下载器，再以相同命令复跑。
最终下载20个文件、`10,687,115 bytes`到：

```text
docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/global_z_w0to2_live_g44_20260824/raw
```

包含：g44 metrics、两rank完整runner CSV/state/manifest、双rank`rollout_step0043.npz`、TensorBoard
config/event、driver/wrapper/observer、resolved config、launch command、resources CSV和一条control trace。
未下载checkpoint正文、`process_rss.tsv`或88个历史NPZ。

## 5. 本地离线分析与作图

分析器：
`docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/global_z_w0to2_live_g44_20260824/analyze_global_z_g44.py`，
SHA256=`0D8EA50AF27734EC56CC1359FB6868AE2F34801B29014C2C91E5EA06764B5153`。

命令：

```powershell
& C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/global_z_w0to2_live_g44_20260824/analyze_global_z_g44.py
```

结果：exit0。生成五run同轴训练表、DVAC rank/step表、g44 future-h表、资源表、artifact schema、summary和四张PNG。
主合同复核：

- metrics完整到g44，主数值全finite；
- 双rank最新NPZ齐全，逐点`w=1+0.5*clipped_z`最大误差`5.96e-8`；
- 最新181个loss-valid query、9050个action权重点；
- control trace 115行，首次success映射到frame114/query2/`h≈13.72`。

针对资源CSV另执行只读统计：

```powershell
& <bundled-python> -c "import pandas as pd; ..."
```

结果：去重后29,831个逐秒样本；本run首次新增`memory.events.max`发生于
`2026-08-24T11:03:52+08:00`；末端max增量78,665、OOM/OOM-kill增量0；有223个样本达到至少
`239.9 GiB`，峰值`239.99996 GiB`。

## 6. 可视化检查

四张图逐张用本地图片查看器检查，标题、图例、坐标与线条均可读，无需返工：

| 文件 | SHA256 |
|---|---|
| `GLOBAL_Z_FIVE_RUN_SUCCESS_G44.png` | `D1D2A98A3945E76F8CCA5C944E19421DEB86481056116124094CBE712F93197C` |
| `GLOBAL_Z_FIVE_RUN_OPTIMIZATION_G44.png` | `EE50D84B2AB2C041362449ED26266C1422F247386C55211CA297ECA3EFBE7B15` |
| `GLOBAL_Z_METHOD_DIAGNOSTICS_G44.png` | `1EFC48723ABA3C51DA585D37BC4A41527756F0C0D02FBCA8E823C434124A758F` |
| `GLOBAL_Z_RESOURCES_G44.png` | `4C9DEABCD445723AA6E70E43292BB140B2987942EA9C8C408C0BD69EBB799266` |
| `SUMMARY_G44.json` | `A20CC74EF5460C48E700EAED939D409AF1E9F63C6E6E437037BA94C9AF4873A5` |

资源图D中的红线是`memory.events.max`相对run首样本的增量；它不是OOM计数。OOM与OOM-kill两条线保持0。

## 7. 文档更新

使用`apply_patch`新增：

- `30_GLOBAL_Z_W0TO2_LIVE_ANALYSIS_G44_20260824.md`；
- 本逐指令账；
- g44下载器、产物只读清单脚本、复现分析器与派生材料。

并更新专题索引与根`HANDOFF.md`的当前step、资源事实和30号文档路由。训练进程未被干预。

## 8. 最后现场刷新

文档整理后再次运行第2节同一只读刷新脚本。结果（`2026-08-24T13:50:35+08:00`）：

- 仍完整到g44，g45 rollout已到10/16；三层控制进程和六个核心worker继续ALIVE；
- fatal=0；两个Traceback仍都是known optional CuRobo probe；OOM/OOM-kill=0；
- NPZ仍为88，checkpoint仍为g10/g20/g30/g40，说明在途step尚未被误算为完整产物；
- GPU现场约`27.25/28.47 GiB`；
- cgroup现场`239.62/240 GiB`，anonymous约`151.13 GiB`、file/cache约`86.34 GiB`；
  max-event累计`246,695`，相对run首样本增量`81,328`，OOM/OOM-kill仍为0；
- host available约829.9 GiB，数据盘仍约余630 GiB。

结论：训练仍在继续，未被本轮检查干预；资源上cgroup内存余量已经很小，后续刷新应继续同时读取
`memory.current`与`memory.events`，但本轮未设置任何自动停止或外置控制。
