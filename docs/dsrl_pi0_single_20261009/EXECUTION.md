# DSRL π0 单卡 U / Norm 执行结果

2026-10-09：51项服务器CPU检查通过；U、Norm各完成一轮真实GPU smoke，均保存CP1并exit0、cleanup_error=null。没有启动正式DSRL。代码、历史比较和参数解释见[CONTEXT](CONTEXT.md)。

## 验收结果

| 项目 | GPU6 U | GPU7 Norm |
|---|---:|---:|
| 完成采集轮数 / 回合 | 1 / 4 | 1 / 4 |
| 采集成功回合 | 1/4 | 1/4 |
| replay chunk / 真实SAC更新 | 35 / 35 | 35 / 35 |
| 信号回放形状 | 35×20 | 35×20 |
| 回放有效信号均值 | 0.30589 | 576.31671 |
| 更新权重标准差 | 0.07704 | 0.07952 |
| dropout比例 | 0.20257 | 0.20257 |
| actor梯度范数 | 6.61607 | 6.37054 |
| 单轮耗时（包含保存） | 338.37秒 | 339.03秒 |
| 退出 / 清理 | 0 / 正常 | 0 / 正常 |

本轮随机预采集后进行35次更新；1/4不是更新后独立评估成绩，不能推断学习收益。评估关闭，未另跑Clean GPU smoke。Clean单卡配置已提供，原SAC路径由CPU检查覆盖。

CP核验：world_size=1、policy_phase=1、update_step=35；replay schema2、resident/inserted=35；U/Norm严格signal_spec与trainer一致；双trick配置入恢复合同；FP32 target shadow有限，actor/target/optimizer/replay文件齐全。两路全部损失和梯度有限；退出后精确driver消失、对应namespace无actor、6/7无遗留C/G进程。[机器回执](smoke-passed.json)。

## 配置、命令和产物

源码：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dsrl-pi0-single-u-n-20261009`。
事务：`/data/chenyiteng/deployment-20261009/dsrl-pi0-single-u-n-v1`，以下简称D。
输出：`/data/chenyiteng/results/rlinf-dsrl-pi0-single/pi0-u-both-g6-smoke-20261009-v2`及`pi0-norm-both-g7-smoke-20261009-v2`。

```sh
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B \
  /data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops/driver.py \
  --plan /data/chenyiteng/deployment-20261008/bc-signal-tau-v1/plan.json \
  --request /data/chenyiteng/deployment-20261009/dsrl-pi0-single-u-n-v1/g6-v2.json
```

GPU7使用g7-v2.json。driver按request注入既有scope环境和命名空间，实际命令/输出见[prepared-v2](prepared-v2.json)、[launched-v2](launched-v2.json)，每路runtime保留resolved.yaml、environment.json、driver.log、started/finished.json。停止条件为本路异常、非有限值、资源越界或一轮完成；未改共享Ray。

[configs](configs/)保存Clean/U/Norm的正式预算配置（未启动）和两路smoke配置。正式预算继承原π0 N4、B256、C20、ODE4、warmup500、UTD20、200轮；smoke仅改为一轮、warmup1、UTD1、关闭评估、保存CP1。所有相对历史及smoke差异均有逐项JSON。

历史SZ1 π0实仓HEAD 4ec52bd8且工作区干净；两机现有模型config/index和stats文件相对路径SHA一致。π0实际stats为physical-intelligence/robotwin/norm_stats.json（649ed92b），已显式覆盖旧π0.5环境变量。没有重新传模型或哈希完整8GB权重。保留历史π0的旧reset环境，已知指令问题作为后续独立对照，未混进本次迁移。

真实6个worker加载的8份关键源码与本地补丁匹配；模型/stats实参正确，环境HEAD0008ae68、vector SHA7afbdfc7与历史π0路线一致。计算/图形上下文均在指定6/7。证据见[source-manifest](source-manifest.json)、[runtime-identity](runtime-identity.json)。

## 检查与异常记录

51项CPU检查涵盖真实SAC actor梯度、microbatch等价、critic/alpha不变、回放/恢复、target shadow、U4/2和Norm4主动作/RNG不变、双trick退化与恢复；[cpu-checks](cpu-checks.json)。首次lint发现两份旧Norm文件缺版权头，补齐后Ruff format/check通过，[lint-checks](lint-checks.json)。

首次资源准备使用了错误的stats候选路径，未启动GPU即修正为现有资产实际路径。v1启动在节点组名称校验阶段退出：保留名node不可用；改为dsrl_pi0_g6/g7并用ClusterConfig CPU校验后，v2执行上述唯一一轮GPU smoke。v1日志保留，不计作通过的smoke，也没有追加训练轮数。

## 回卡

原候补RLT暂停前按PID/UID/start/namespace核对，两路完整CP150已保留。旧driver未写finished文件，以精确driver消失、namespace无actor、C/G清空三项独立证据确认释放，不虚构退出码：[released](released.json)。

6/7已归还原RLT，从各自CP150重新派发，20:31已建立真实C/G上下文且仅在指定各卡，当前初始化、尚未核验恢复后的新训练轮。owner=1025410；GPU6/7 driver=1025537/1025548；命名空间rlt-sz3-g6/g7-after-dsrl-pi0-smoke-1009-v1。4/5 WM调度槽保持原状。精确身份、request和回卡启动阶段见[returned](returned.json)、[return-verified](return-verified.json)。
