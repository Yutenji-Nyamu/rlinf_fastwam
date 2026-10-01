# 深圳3 Wan Goal 继续实施 · 2026-10-01

用户起床后要求继续昨晚工作并检查三机。08:43实查12条原RLT都存活并继续推进：SZ1约629–642轮，SZ2约1349–1409轮，SZ3约126–132轮。SZ1、2保持原训练；SZ3在准备完成后借物理4–7卡继续Wan Goal单视角π0.5。0–3无新的任务安排。

## 配方和验收

沿用RLinf `d34d4c320d08cb982de034aa9a011f08dc0fa217`及昨晚固定模型和独立环境；不重下载。OFT两次真实更新已由独立证据验收，原退出阶段monitor故障单独保留；本次复用`learning-reconciled.json`，不重跑OFT。

|阶段|环境N / GRPO组G / rollout轮R|最长步数L / 执行动作块C|global / micro batch|预算与存储|
|---|---|---|---|---|
|π0.5 smoke|32 / 8 / 1|320 / 8|1280 / 64|2轮，每轮完整checkpoint|
|π0.5正式|64 / 8 / 8|320 / 8|2048 / 128|1000轮，每40轮存储；从原SFT开始|

两阶段均为用户已批准的头部RGB＋腕图mask，原动作维度与GRPO语义不变。smoke需正常退出、两轮有限有效梯度、完整CP1/CP2及实际权重变化才进入正式。WM内部奖励与真实LIBERO成功率分别记录。启动入口`scripts/prepare_wake_resume.py --launch`；实际子入口`private_ray_driver.py --config wan_goal_pi05_headonly_{smoke,formal}_sz3`，各自独立Ray端口63844/63845。

本次输出`/data/chenyiteng/projects/wan-goal-sz3/runs/wan-goal-sz3-20261001-r3`；源码、模型、缓存及临时目录均在数据盘。仅GPU4–7，offload及actor/env/rollout placement沿原配置。失败或用户停止后按精确owned身份清理并续原Dojo；Dojo完成/失败释放后续原四RLT，总预算仍3000轮。共享Ray、其他用户和无关任务保持原状。

## 新资源交接

昨晚v2 owner已结束，但用户另行恢复了RLT，所以其历史`pipeline-final.json.rlt_dispatched=false`仍保留。当前权威恢复证据是v2目录`user-rlt-return.json`与旧cycle `resumed-dispatched.json`，08:43又实查四条真实进展。新prepare仅补齐该精确外部归还回执路径；仍核cycle、continuation、释放文件SHA、实际driver/namespace/完整checkpoint，不复用旧stop或restore。

新bridge为`P/scripts/wm-bridge-20261001-v3`，新cycle为`P/rlt-cycle-sz3-wan-goal-20261001-wake-v1`，新continuation为`D/continuation-20261001-wan-goal-v3`。P是SZ3 Dojo项目，D是原6300回合目录。新owner负责WM→原Dojo→原RLT的唯一归还链。`w006`准备成功并保留2313个Dojo回合，四条原RLT绑定最新完整CP125；`w007`于08:55唯一启动owner。禁止重放prepare/launch/stop。

## 现场证据

本地细日志`local_logs/wan-goal-20261001/steps/`：w001–w003三机只读快照；w004核旧owner已结束、外部RLT归还存在、13份运行源码SHA仍匹配、环境已审通过、63843/63844/63845未占用。SZ3根盘余5.57GB，数据盘余7.22TB；缓存/输出继续放数据盘。后续只追加真实执行证据。

08:57 `w011`确认原RLT释放后π05 smoke已启动，独立Ray63844；actor/env/rollout实际placement均物理4–7。模型正在加载，尚无真实更新验收。
