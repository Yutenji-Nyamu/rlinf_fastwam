# Global-z `[0,2]` g35/g36 现场刷新与分析逐指令流水账

日期：2026-08-24  
范围：AutoDL只读现场、轻量下载、本地离线分析与文档更新；未停止、重启或修改服务器训练。
密码只进入当前Paramiko进程环境，不写入文件或本文。

## 1. 前置阅读

按工作区规则完整读取：

```powershell
Get-Content PROJECT_CONTEXT.md
Get-Content HANDOFF.md
Get-Content docs/rlinf-robotwin-pi0-dvac-telemetry/00_INDEX_AND_PLAN.md
```

结果：当前唯一active AutoDL run为global-z `[0,2]` 100-step formal；本轮授权是查看训练、产物与资源，
不包含停止或修改。

## 2. 第一次AutoDL只读刷新

远程只读脚本：`tmp/idea2_global_z_w0to2_live_refresh_20260824.sh`，SHA256：
`3B153FF7DFF57DD6D9E49477DBF7712F45A1374F7EFBCF8AD3A8CB95FC26DCF5`。

调用形态：

```powershell
$env:SEETA_SSH_PASSWORD = <current-process-only>
& <bundled-python> local_scripts/remote_exec_autodl.py run `
  --command-file tmp/idea2_global_z_w0to2_live_refresh_20260824.sh
Remove-Item Env:SEETA_SSH_PASSWORD
```

关键结果（10:14 CST）：

- identity=`autodl-container-nekaqbwt43-6ce5babb`，`/root`，UID0；
- wrapper/driver/observer PID=`820640/820644/820645`均alive；
- 完整到g35，g36 rollout 12/16；两EnvWorker、两actor、两rollout worker均ALIVE；
- fatal=0，OOM/OOM-kill=0；
- run/runtime=`30G/55M`；70 NPZ、g10/g20/g30 checkpoint；
- GPU约26.7 GiB/card；cgroup约235.2 GiB，host available约845 GiB。

问题与解决：没有连接或命令失败；Paramiko复用既有固定host-key和低层密码认证路线。

## 3. 轻量快照下载

下载器：`tmp/idea2_global_z_download_live_g35_20260824.py`，SHA256：
`7266EBBF108B7E71A38D5E62B6C4276079B75CA82C89F76992F9E2BFA37AB9C8`。

调用：

```powershell
$env:SEETA_SSH_PASSWORD = <current-process-only>
& <bundled-python> tmp/idea2_global_z_download_live_g35_20260824.py
Remove-Item Env:SEETA_SSH_PASSWORD
```

结果：下载15类小文件与1个TensorBoard event，共`8,834,749 bytes`，目标：

```text
docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/global_z_w0to2_live_g35_20260824/raw
```

包含：g35 `metrics.log`、两rank完整runner CSV/state/manifest、双rank
`rollout_step0034.npz`、driver/wrapper/observer、resolved config、launch command、resources.csv和event。
未下载checkpoint正文或全部历史NPZ。

## 4. 本地离线分析与可视化

分析器：
`docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/global_z_w0to2_live_g35_20260824/analyze_global_z_g35.py`，
SHA256=`260D658AD30ACF23E9C41D413BA6ED4888C0CE18BC4E942DC80472A45BF1B455`。

命令：

```powershell
& C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/global_z_w0to2_live_g35_20260824/analyze_global_z_g35.py
```

首次结果：exit0，生成五run训练表、DVAC rank/step表、g35 future-h表、资源表、summary和四张PNG。
视觉检查发现优化图标题过长；只缩短标题后用同一命令复跑，exit0。方法数值与CSV/NPZ合同一致：
双rank、80个loss-valid query、公式误差约`5.96e-8`，全部数组finite。

最终高信息量图SHA256：

| 文件 | SHA256 |
|---|---|
| `GLOBAL_Z_FIVE_RUN_SUCCESS_G35.png` | `109D1B62367B7AC0EC2D001A0502924F05225B3EDAACC46BA5980491783C5D50` |
| `GLOBAL_Z_FIVE_RUN_OPTIMIZATION_G35.png` | `05BE29B149A834916B43F0B947462BC73305F5EBAFD958B1738CD71B0E2427F2` |
| `GLOBAL_Z_METHOD_DIAGNOSTICS_G35.png` | `898151EBC50ADE65C873D91C5844B2B1092768BBA69F3AF8FBA1CBE44DBF0D94` |
| `GLOBAL_Z_RESOURCES_G35.png` | `5DA5F2FBBCC3ECD30704D3EFBC1C1B0A98FAABC5CFBA8196233AD496E4DE0E73` |
| `SUMMARY_G35.json` | `6686968B33DAE8EF3131D54DCE837E0F31EE6E783EFF318F98A21D32F2765785` |

四张图均用本地图片查看器逐张检查：标题、图例、坐标、线条与注释可读。

## 5. 最后AutoDL只读刷新

再次运行第2节同一只读脚本。结果（`2026-08-24T10:25:38+08:00`）：

- 已完整g36，g37 rollout 3/16；
- g36 success/KL/clip/grad/ratio=`96.875%/0.036/16.8%/22.180/1.023`；
- 72 NPZ，仍为g10/g20/g30三份checkpoint；
- wrapper/driver/observer及六个核心worker均alive，fatal=0；
- GPU现场约`25.6/25.4 GiB`；
- cgroup=`235.97/240 GiB`，历史峰值`237.86 GiB`；本轮`max/OOM/OOM-kill`增量仍为0；
- host available约844 GiB，AutoDL数据盘约余640 GiB。

结论：训练正常继续。本轮没有执行任何写服务器状态、停止进程或训练配置变更。
