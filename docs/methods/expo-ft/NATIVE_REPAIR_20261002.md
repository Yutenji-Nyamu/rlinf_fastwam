# SZ2 EXPO 开关任务原生崩溃修复 · 2026-10-02

状态：原版原生崩溃已复现，修复版三轮N4→N1全部通过；正式driver2892740于2026-10-02 12:35启动，12:40已核完整恢复并提交首回合59动作成功，第二回合推进中。当前实际训练进度以新scope的`run/status.json`为准，唯一控制路由为`current.json`。

## 原始故障与范围

旧 scope `formal-turn-switch-20261001` 初始固定评估完成20回合、9次成功（45%），`complete.json` 已写出；训练事件仍停在 `evaluation_started`，owner 捕获 driver `-11/SIGSEGV`。在线回合、真实动作、Q/FM/editor/温度更新均为零。`coredumpctl` 未安装、普通用户无完整系统内核日志权限，原日志没有故障点的 native 栈。

代码顺序为：评估 `env.offload()` → 写完成回执 → 函数返回并释放局部环境 → 恢复 CUDA/RNG → `evaluation_finished`。故障窗口位于此边界，不能归因于后续训练环境初始化、warmup 或学习率。也没有证据据此认定 OOM。

现场 RoboTwin `VectorEnv` 使用 `ThreadPoolExecutor`；原 `close()` 关闭场景、清环境列表、GC和CUDA cache，但没有显式 `shutdown()`。候选修复只在 EXPO 持有的环境上，先等待线程池结束，再关闭场景；训练和评估两处调用同一关闭函数，新增边界日志和 `faulthandler`。未改共享 RoboTwin/SAPIEN 安装、驱动、Ray、模型或方法配方。上游依据与不匹配的问题见 [来源记录](NATIVE_CRASH_SOURCES_20261002.md)。

2026-10-02原版原生复现：不加载policy、不做学习，N4执行两次reset及短step后，`offload()`返回、四卡CUDA同步均通过；释放函数局部环境时发生`SIGSEGV/-11`。`Fatal Python error`与主线程的`function_return_after`打印交错，尚未GC/RNG恢复。证据将问题限定到原生环境退出/析构边界，强支持线程退出与场景释放顺序问题；没有C++栈，不能指认具体SAPIEN/PhysX TLS析构器。

12:34:59修复版检查通过，12:35:04独立进程以0退出且GPU释放验证通过。三轮N4→N1共6个环境生命周期、每个两次reset及短step、函数返回/GC/四卡CUDA RNG恢复全部通过；原版首个N4函数返回即-11。这是针对已复现关闭故障的单项修复对照，不代表所有模拟器崩溃都已排除。诊断没有policy、没有学习，不计正式动作预算；正式真实动作和学习状态另行记录。

## 执行合同

- 新 scope：`/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-repair-20261002`；源码分支 `codex/sz2-expo-ft-repair-20261002`，基于 `67cff0224a55a3387bc09a6ec92484da55bca901`。
- 2机物理4–7，任务 `turn_switch`，200动作、H50/C10、8base+8edit、4卡DP、B64；10回合warmup，每40真实动作Q20＋FM/editor/温度各1；20,000在线真实动作，评估另计；action expert、不用LoRA、无图像增强。全部继承。
- 新唯一owner197649/start371818311。新cycle `rlt-cycle-expo-turn-switch-repair-20261002-v1`；只停止当前四个有身份锚点的 RLT driver/namespace，完整清理后才进入原配置native诊断。共享Ray、其他机器及0–3卡保持。
- 四RLT现已完整退出，备用恢复点分别为gpu4/5/6/7的1850/1875/1825/1825。因原最新CP缺少部分replay负载，原helper从核同供体构建独立恢复CP并严格验收，原件保留；不是从更早轮数重新开始。
- 顺序：原版诊断 → 有界修复门禁 → 修复版N4→N1生命周期检查 → 完整checkpoint恢复正式。每个后代PID均登记UID/boot/start并用pidfd清理。任何失败或正式退出后，核4–7释放才恢复原四RLT并验证首轮；无重复stop/resume。
- 诊断日志和临时文件位于数据盘；仅此owner及其后代禁用core dump，避免占用近满根盘，未改系统设置。

正式命令由owner设置原GPU UUID与原生库环境后执行：

```bash
EXPO_PY -X faulthandler -u -B SOURCE/examples/embodiment/train_expo_formal.py \
  --inputs SCOPE/inputs.json --run SCOPE/run --max-physical-actions 20000 \
  --enable-evaluation --resume SCOPE/run/checkpoint-latest.pt
```

## 状态迁移与CPU验证

原checkpoint是 `fresh-base`，约2.94GB。已CPU全量核对旧输入、源码manifest、合同、finite及全部payload digest，零在线数据/零学习均成立。迁移只改变外层目录合同与replay的 `root/root_identity`；base/core/cadence/rng/progress digest原样，replay其余逐字段digest原样。新replay经实际 `FormalReplay.load_state_dict()` 和状态往返校验，原checkpoint、replay和评估回执保留。

已完成初评逐种子、动作数、task、原基座0更新、模型候选数和config SHA核对后字节复制；保持checkpoint中 `evaluation_initial=False`，由原评估缓存校验流程正常提交结果，不伪造训练进度。

服务器定向CPU检查：生命周期3项、owner身份/路径/门禁7项、原driver提交边界10项、迁移拒绝与挂载别名9项通过。实际迁移回执 `checkpoint-migration.json` 为 `ok=true`、`cuda_initialized=false`；这不代替native验证。

## 证据位置

轻量原始证据在本机 E盘 `.../expo-repair-20261002/`；服务器新scope包含 `current.json`、各阶段日志、`native-*-events.jsonl`、`checkpoint-migration.json`、`rlt-cycle-*/`及正式 `run/`。旧错误回执、旧初评及原源码不覆盖。

12:40正式核验：`resume_verified`逐payload digest通过，原45%初评通过原合同缓存校验并正常提交；首回合59真实动作成功且完整checkpoint已提交，第二回合150动作进行中。心跳约2秒、owner/driver存活，无failure/final；仍是10回合warmup，更新0符合合同，不据首回合成功推断学习提升。4–7持有EXPO进程，原RLT退出证明已核。根盘约2.8GiB可用，临时文件/输出/CP均指向约6.3TiB可用的数据池；没有扩展清理范围。

发布内容为本次审过的源码、文档和[轻量原始证据](evidence/native-repair-20261002.json)，不含模型/数据/大checkpoint。Git独立分支[codex/sz2-expo-ft-repair-20261002](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz2-expo-ft-repair-20261002/docs/methods/expo-ft)。后续沿每25回合固定评估观察长期稳定性和训练提升；正式退出后由197649唯一清理并续RLT。
