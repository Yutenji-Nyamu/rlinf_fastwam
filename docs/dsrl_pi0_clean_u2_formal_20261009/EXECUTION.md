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

## 10-10延长至400轮

用户本轮授权两组延至400。各自先完成当前200轮并保存完整CP200，确认exit0、cleanup_error=null、精确driver及同卡C/G释放后，由原逐卡owner直接接续200→400；随后才接原RLT CP150。不打断当前训练，4/5 WM原槽不变。

- 新事务：原D下`extend400-20261010/priority-g6.json`、`priority-g7.json`；命令沿用上文driver/plan，仅request换成本次路径。
- 输出：原结果根下`pi0-clean-g6-formal400-from200-20261010-v1`、`pi0-u-tau2-both-g7-formal400-from200-20261010-v1`；resume_dir分别指原200轮输出的`<原实验名>/checkpoints/global_step_200`。两份实配在本页同目录`extend400/g6-resolved.yaml`、`g7-resolved.yaml`。
- 只改max_steps=400及必要输出/恢复路径。SAC参数、优化器/调度器、alpha、target、trainer计数与replay均从完整CP恢复；N4/C20/ODE4/B256/UTD20、评估13/保存65不变。
- **U退火仍R1→R200降至0**，不因延预算改方法合同。因此R201–400继续训练，但U重加权关闭；这不是把原退火曲线拉长到400。
- 原owner1780556已精确替换为1344615，训练子进程保持。既有WAIT_EXTERNAL只加可选`priority_after_external`分支，等待成功原任务后进入PENDING_PRIORITY；其余槽默认逻辑不变，无新增后台层。最小补丁和启动证据在同目录`extend400/`。
- 16:23核验：7卡原200轮退出0且释放，已自动启动400轮driver2369148，处于模型加载；6卡原训练尚未退出，接续已排队。400轮完成或异常结束后按既有规则接原RLT；原200阶段异常则不自动以不完整CP接续。当前未把加载视为恢复首轮成功。

## 10-10 DSRL收束并切换Attn

按用户新授权收束：Clean完成R218，U完成R224，完整恢复点均保留CP200；两路exit15、cleanup_error=null，逐namespace与GPU C/G清场通过。400轮预算未跑完。原RLT CP150逐卡候补改为等待Attn。配置、曲线、结论与回执见[CLOSEOUT.md](CLOSEOUT.md)。
