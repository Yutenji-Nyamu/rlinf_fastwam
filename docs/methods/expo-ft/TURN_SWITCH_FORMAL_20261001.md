# SZ2 EXPO 换拨开关任务 · 2026-10-01

2026-10-02后续：本页为原始换任务合同。初评9/20=45%已完成，随后环境析构SIGSEGV，尚无在线动作/学习。当前修复和完整恢复转到[原生关闭修复](NATIVE_REPAIR_20261002.md)，新scope `formal-turn-switch-repair-20261002`，原方法/任务/预算保持；不要重放本页旧owner或stop/resume。

用户授权：挑官方400步、论文π0.5约60–70%、最好已有成功训练历史的任务，替换瓶子EXPO正式训练。

选 `turn_switch`。RoboTwin [官方step limit](https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/env_cfg/task_config/_eval_step_limit.yml)为400；[Fast-WAM v2 Table3](https://arxiv.org/html/2603.16666v2#A1.T3)的π0.5 Clean62%/Randomized54%。此前本工作区[π0.5 RLT成功基线](../../server-admin/PI05_RLT_TURN_SWITCH_SUCCESSFUL_BASELINE_20260925.md)末5固定评估均66%、末次65%、最高85%。这些数值用于选任务；当前Sidney原基座在200动作/C10下的实际起点须重新评估，不能把论文分数当本次结果。

## 启动前配置与边界

- SZ2物理4–7；任务 `turn_switch`、fallback prompt `turn the switch`；原始Sidney π0.5与新EXPO组件，不续旧任务checkpoint或replay。
- 原正式配方完整继承：200动作、H50/C10/14D、Euler10；8base+8edit、4卡DP、B64；10回合warmup，不积欠学习；随后每40真实动作Q20、FM/editor/温度各1。
- 总预算20,000在线真实动作，含warmup，demo及评估另计；action expert及投影训练、视觉语言冻结、无LoRA、无图像增强。方法参数和更新频率保持。
- 示范复用历史官方Clean50：HF revision `9dc9299c163db059931898a9f0852098a61155a1`，50条4863帧。SZ1→SZ2直接复制52份parquet/metadata，逐文件SHA256核同；新补`prepared.json`注明来源，不假称历史原件已有该回执。
- 复用既有 `turn_switch_{train,eval}_rlt_20260923.json`，eval20条、4×5、200动作，保存seed SHA。原种子表没有开关专家重筛，沿用此前成功实验的可复现名单。官方榜、论文、模型卡的权重/协议不等同。
- 输出scope `/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-20261001`；独立source/run、owner/cycle、branch与RLT恢复namespace。共享Ray及其他用户原状。

命令（由独立owner设置固定GPU UUID及原生环境）：

```bash
/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin/python -u -B \
  /data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-20261001/source/examples/embodiment/train_expo_formal.py \
  --inputs /data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-20261001/inputs.json \
  --run /data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-20261001/run \
  --max-physical-actions 20000 --enable-evaluation
```

停止条件沿用：预算完成、非finite、配置/源码/资源身份漂移、进程故障或心跳失联。先清理本EXPO后代并核4–7释放，才恢复原四RLT。旧EXPO已按原机制清理；按用户随后“直接切正式、RLT只做备用”授权，核旧release及四个临时RLT driver/namespace身份后精确结束旧等待监督进程，新owner直接接管，不等待RLT完成训练首轮。不重放任何旧stop/resume意图，旧pinned源码不热改。

核心算法、模型、replay与driver源码保持原字节；仅owner/resources的scope和前轮cycle路由修改。服务器21项现有CPU检查通过，50条4863帧实际demo读取与H50/14D采样通过，CUDA未初始化。

2026-10-02 00:06:50 owner839628启动，00:07:35 driver942043已进入原始π0.5加载。新cycle确认四RLT driver/namespace全部停止、4–7释放；完整CP1575/1600/1550/1550保留作退出后的恢复点。输入SHA `b81b67dcf5b7c21cab9579b81aefc35a18bc52f29c108fbb5be1455899e58ded`。独立owner持续监督，不依赖工作SSH。

00:13只读复核：`EXPO_RUNNING`、`evaluation_started`，首组4环境已实际推进每环境110动作，心跳约10秒、无failure。起点评估尚未完成，在线训练动作与学习更新仍为0；初评后自动进入10回合warmup再学习。源码/配置/轻量回执已推 `codex/sz2-expo-ft-turn-switch-20261001@67cff0224a55a3387bc09a6ec92484da55bca901`，远端SHA一致。动态原始回执在E盘 `expo-task-switch-20261001-2320/sz2/new-status2.out`；本段是现场后续记录，未计入该Git提交。
