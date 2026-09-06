# R-only formal live snapshot through Global Step 35

只读现场时间：`2026-08-22T15:00:47+08:00` 至 `15:01:25+08:00`。

## 现场结论

- 最新完整训练记录为 `Global Step 35/100`；探针结束时 wrapper、driver、observer 仍存活，Step 36
  rollout 已到 `8/16`。
- 两卡即时显存约 `27.1/27.5 GiB`；资源 CSV 全程峰值为 `30.37/30.22 GiB`。
- cgroup `memory.max=240 GiB`；只读探针时 current 约 `223.3 GiB`，CSV 瞬时峰值约
  `237.41 GiB`。`low/high/max/oom/oom_kill` 全为 0。
- 数据盘可用约 `727 GiB`。run 约 `30G`，其中 g10/g20/g30 三份 DCP 各约 `9.7G`；runtime
  约 `93M`。DVAC 共有 70 个 rank-local NPZ。
- driver 中两条 `Traceback` 均来自启动期可选 CuRobo planner import；后续已连续完成 35 个 Global
  Step，未发现 CUDA OOM、NCCL/Ray fatal、空间耗尽或进程退出。

## 训练与方法快照

- g1–35 training-rollout success：R-only `85.27%`，历史 GRPO 同轴 `86.75%`。
- 最近 5 步：`91.80%` vs `92.11%`；最近 10 步：`91.02%` vs `90.70%`。
- g1–35 mean approximate KL：`0.0380` vs `0.0446`；joint-query clip fraction：
  `0.1371` vs `0.1468`；pre-global-clip gradient norm：`33.28` vs `32.52`。
- g2–35 R-only weight：mean `0.9981`，mean p05/median/p95=`0.708/1.030/1.199`；
  `38.13%` action-h 降权、`61.87%` 升权；下/上 residual cap 命中约 `0.38%/7.22%`。
- g2–35 正/负 advantage query 的 mean weight 为 `0.988/1.019`，即当前累计趋势仍是负向
  update 略强于正向 update。
- 最新 g35 双 rank NPZ 中，267 个 loss-valid query 的 p05/median/p95 为
  `0.645/1.025/1.200`；正/负 advantage query mean weight 为 `0.973/1.010`。

这些 success 都是 on-policy training rollout，不是 fixed-ID checkpoint evaluation。

## 本地证据

- `analysis/TRAIN_AND_METHOD_G35.png`：训练 success、KL、PPO clip、pre-global-clip norm 和 R-only
  权重现场图。
- `analysis/RESOURCES_G35.png`：GPU、cgroup RAM、host RAM 与 disk 曲线。
- `analysis/SUMMARY_G35.json`：本文全部数值的机器可读汇总。
- `analysis/{TRAIN,DVAC_RANK,DVAC_STEP,LATEST_HORIZON}_*.csv`：逐步及 latest-h 派生表。
- `raw/`：只拉取的小型 metrics、driver、resource CSV、双 rank runner CSV、g35 NPZ、rolling state
  与 resolved config；未复制 checkpoint。
- `analyze_live_g35.py`：可复现分析与 Pillow 绘图脚本。

## 逐指令记录

1. `python local_scripts/remote_exec_autodl.py run --command-file
   tmp/idea2_r_only_formal_live_refresh_20260822.sh`
   - 默认 Windows `python.exe` 是不可执行商店占位符，未触及服务器。
   - 改用 Codex bundled Python 后身份探针通过，固定 host-key 与密码认证通过，返回 g35、进程、
     metrics、产物、GPU/RAM/disk 与 fatal 扫描。
2. `... run --command-file tmp/idea2_r_only_formal_live_refresh_detail_20260822.sh`
   - 第一条审计的 checkpoint glob 少了一层 run-name 目录，因此 checkpoint 段为空。
   - 第二条用 `find ... -name global_step_*` 只读定位，确认 g10/g20/g30 均存在且各约 9.7G；同时
     读取 Traceback 上下文、cgroup limit/events 与资源 extrema。
3. 通过同一 helper 的 `get` 子命令分别 SFTP 拉取 10 个小文件；全部成功。
4. bundled Python 运行 `analyze_live_g35.py`。第一次因 workspace parent 层级写成 `parents[4]`
   找不到复用脚本；改为 `parents[3]` 后成功生成全部 CSV/JSON/PNG。
5. 使用本地 image viewer 逐张视觉复核两张 PNG，标题、图例、坐标和曲线均可读。

远端命令均为只读；密码只注入执行进程，没有写入本文件、脚本或服务器。
