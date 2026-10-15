# EXPO Clean B64：官方熵目标尺度换算

2026-10-15用户授权：SZ1物理4/5卡停止Editor-U τ2，得出结论并推送日志，改跑B64 Clean；RLT逐卡候补按原流程。已执行，旧τ2收束提交a05ee4790。

## 最小变更

- `signal=null`：使用已有Clean路径，关闭U采集/加权及关联dropout、退火；不更改FM、critic、alpha更新算法。
- `core.target_entropy=-295.32130774077405`：等价于官方`adjust_target_entropy=True`，即 `-D/2 + D*ln(edit_scale)`，D=10×14=140，edit_scale=0.2。既有配置字段支持显式目标，无需修改算法源码。
- 另改独立run的训练/评估输出路径和说明元数据，精确差异见config-diff.json。port_source_manifest全体仍与旧已验收源码一致。

官方出处：[pd-perry/expo-ft 023cf9c](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L286)。此为官方支持的选项，改变原默认；不是断言论文所有实验均开启该选项，也不是已证明的稳定性修复效果。

## 继承设置与启动

任务turn_switch，种子42与原训练/固定评估种子表；从同一原始π0.5和新优化器开始，无resume。B64，两卡共同分摊；候选8+8，H50/C10/ODE10；Q20→FM1→Editor1→alpha1；10回合预热、之后40动作/学习；60000真实动作含预热、不含评估；初始/每10回合/最终20条评估。学习率、修正范围±0.2、64GiB回放缓存和渲染绑定均继承。

正式运行配置：`examples/embodiment/config/expo/clean_entropy_b64_sz1_60000.json`。
运行目录：`/data/chenyiteng/results/expo-clean-entropy-b64-sz1-g45-60000-1015`。
操作事务：`/data/chenyiteng/deployment-20261015/expo-clean-entropy-b64-v1`。
源码：`/data/chenyiteng/projects/expo-fm-u-20261010/source`（原已验收快照，无运行源码修改）。

```sh
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B /data/chenyiteng/projects/expo-fm-u-20261010/source/examples/embodiment/train_expo_formal.py --inputs /data/chenyiteng/results/expo-clean-entropy-b64-sz1-g45-60000-1015/inputs.json --run /data/chenyiteng/results/expo-clean-entropy-b64-sz1-g45-60000-1015/run --max-physical-actions 60000 --enable-evaluation
```

资源：CUDA绑定物理4/5的UUID；图形绑定物理4。复用已有pair_owner与统一owner。60000动作完成或用户停止则退出；既有数值异常及越卡保护保持，清理仅本进程及其后代，确认C/G释放后独立接回各卡RLT。

## 验收与边界

服务器CPU校验：正式validate_inputs、全源码pin、Clean无加权、官方公式值通过，无CUDA初始化。参数差异仅上述方法和路径项。原τ2正常SIGTERM/exit143，末次完整恢复权重保留，启动新Clean前两卡无C/G；6/7的plan字段和driver身份保持。

新driver896595完成加载及初始checkpoint，`resumed=false`；原生初评已真实推进到第一组每环境100动作。计算在4/5、图形在4，无失败回执。这是启动/初评验收，尚非新学习结果或完整稳定性结论。

GPU4/5原RLT候补均保留CP100及完整标记，配置、环境存在，未提前派发；原owner存活，两槽WAIT_EXTERNAL。退出后按各自候补继续，未重建共享Ray。

证据：prepared.json、acceptance.json、initial-eval-progress.json。operations内是已执行的一次性切换及只读核验代码，事务有唯一回执，切勿重放cutover。
