# SZ2 EXPO-FT smoke 运行包

用户授权：自主实现、smoke、就地解决必要问题并重试；EXPO优先、RLT低优先级。最新卡位授权在另一窗口已核对原用户消息：深圳2物理4–7，原四RLT先退出，结束后续回原累计3000预算。无需重复请求执行许可。

- source：`/data/chenyiteng/projects/expo-ft-sz2-20261001/source`，独立branch `codex/sz2-expo-ft-20261001`，从SZ2已跑通 `e5cb9aa4c767490a66cba3c4e155c76f1193f199` 接入。
- 环境：同原生`pick_diverse_bottles`、既有seed表、MPLIB、200真实动作，π0.5 H50/C10/14D/pad32/ODE10、原Sidney权重及norm。模型权重、norm、配置与新增源码均校验SHA256。
- 方法：normalized C10×14上8 base＋8 bounded edits（delta=.2），combined action不额外tanh，一次native decode；10Q，selection/TD分别随机2Q-min，editor为10Q均值。base真实FM只更新action expert/projection，冻结视觉/语言，无LoRA；独立可训练torchvision ResNet50/GN共享三camera，无作者augmentation；variable-K折扣、成功terminal停止bootstrap。完整依据及移植差异见实现合同。
- 短预算：1真实环境，critic B4，FM B1，候选microbatch4；fresh1episode及新进程resume1episode，每episode结束后各1 update call，顺序为20 critic→1 FM→1 editor/temperature；不代表每chunk/动作的正式UTD预算。FM优先最后一个在线成功H50窗口，否则已核验成功clean50演示；如实注明来源。正式B64、warmup、采集/更新cadence与并行规模尚未验收。
- 卡位：先用授权集合中的物理4；RLT在借卡前锁定完整CP、原实配及唯一恢复owner，smoke结束/失败后归还。不同窗口按SZ2/SZ3执行。
- 输出：项目根`inputs.json`、`source-manifest.json`；`runs/smoke-v*/{fresh,resume}/{resolved.json,events.jsonl,checkpoint.json,complete.json,command.log}`；唯一owner及final回执。大checkpoint留服务器，不传Git。
- 停止边界：非有限loss/参数、真实更新未发生、恢复哈希不等、环境/导入故障、非预期占卡、每phase超过90分钟，均退出当前attempt，精确清理并保留证据。常规可定位故障按新attempt重试，不扩大方法预算。

当前尝试的单phase命令（通过以fresh/resume/final回执为准）：

```bash
/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin/python -u -B \
  /data/chenyiteng/projects/expo-ft-sz2-20261001/source/examples/embodiment/train_expo_ft.py \
  --inputs /data/chenyiteng/projects/expo-ft-sz2-20261001/inputs.json \
  --run /data/chenyiteng/projects/expo-ft-sz2-20261001/runs/smoke-v2/fresh \
  --episodes 1 --batch-size 4 --candidate-microbatch 4
```

resume采用同命令加fresh checkpoint路径，另起进程、验证完整学习状态后再执行与更新。starting模型全文件SHA/norm身份、base参数filter和attention运行策略均进入严格contract。`smoke-v1`的cuDNN SDPA故障后，v2复用已跑通DV50策略，仅关闭cuDNN SDPA；普通cuDNN卷积和flash/efficient/math SDPA保留。owner注册BOOT/UID/PID/start，持久flock，TERM/finally清理；不操作共享Ray基础进程或无关namespace。当前没有完整成功回执，不能称smoke已通过。
