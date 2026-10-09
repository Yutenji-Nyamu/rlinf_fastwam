# DSRL π0 单卡 Clean / Uτ2 正式启动

2026-10-09。用户授权SZ3物理6/7正式运行，原RLT逐卡候补。继承已验收c2c64706源码，不重复smoke。

- 6卡Clean；7卡U（ODE4/2历史采集信号），仅将原U温度2.5改为2；dropout=.2、私有seed42、R1→R200退火保留。两组从相同π0底座新启，resume_dir=null，不载入smoke CP。
- 原成功π0配置：adjust_bottle、N4 train/N4 eval、H50/C20、ODE4、B256/MB256、replay25000、warmup500、UTD20、200轮；评估每13、保存每65及结束。沿用原环境/seed/旧reset路由；本轮不混入π0.5修复环境。
- CPU只检查本次resolved配置、源码SHA、scope/ClusterConfig/恢复合同；前次51项CPU和U/Norm真实单卡smoke已通过。
- GPU仅物理6/7；复用逐卡EGL marker/profile/bootstrap与Ray明确物理卡placement，启动前核对UUID及scope四份SHA。共享Ray和4/5 WM槽保持。
- 源码：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dsrl-pi0-single-u-n-20261009`。
- 事务D：`/data/chenyiteng/deployment-20261009/dsrl-pi0-clean-u2-formal-v1`；优先request为D/priority-g6.json、priority-g7.json；runtime保留resolved.yaml/environment.json/driver.log/started/finished.json。
- 输出：`/data/chenyiteng/results/rlinf-dsrl-pi0-single/pi0-clean-g6-formal200-20261009-v1`、`pi0-u-tau2-both-g7-formal200-20261009-v1`。

两条命令分别为：
```sh
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B /data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops/driver.py --plan /data/chenyiteng/deployment-20261008/bc-signal-tau-v1/plan.json --request /data/chenyiteng/deployment-20261009/dsrl-pi0-clean-u2-formal-v1/priority-g6.json
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B /data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops/driver.py --plan /data/chenyiteng/deployment-20261008/bc-signal-tau-v1/plan.json --request /data/chenyiteng/deployment-20261009/dsrl-pi0-clean-u2-formal-v1/priority-g7.json
```

停止/回卡：200轮正常结束或本路异常退出；原owner仅在精确driver、namespace和该卡C/G均清空后启动该卡原RLT最新完整CP。两卡独立，不互相等候，不新增看护、缓存限制或自动重试训练。

## 现场记录

正式配置与scope/源码检查已通过。原RLT两路完整CP150保留；23:58精确停止，exit15、cleanup_error=null；23:59核验driver消失、namespace actor=0、6/7 C/G清空，4/5原进程不变。10月10日00:00已派发两路正式训练，CPU owner1780556；两卡分别配置priority与原RLT fallback，候补尚未启动。两路driver=1780684/1780686，00:03核验各自仅在物理6/7建立actor/rollout/env上下文；π0模型及norm stats加载正确。最新采集与绑定证据见live-verified.json。

2026-10-10 00:09:07：两路均PRIORITY_RUNNING并进入Generating Rollout Epochs正式预采集；owner心跳距核验约1.35秒。每路actor/rollout/env的C/G上下文仅在对应6/7，0–3无C/G，4/5原WM进程不变；没有finished回执或本次异常。原RLT两个新fallback request均未派发，CP150完整标记SHA保留。此处仅确认已开始预采集，未声称完成首轮或已越过warmup500。

本次无算法源码改动，只发布两份实配、差异和轻量启动/候补证据；基线c2c64706已与Yutenji-Nyamu仓库核同。
