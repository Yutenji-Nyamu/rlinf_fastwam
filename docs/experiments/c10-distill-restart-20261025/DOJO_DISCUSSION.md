# Dojo π0.5执行长度与N1/N10扫描：讨论稿

2026-10-25用户授权的是先讨论；本轮仅只读核对1机已跑通源码、旧结果/外层组件和官方入口，没有启动Dojo或占用额外GPU。1机4–7当前已交接新蒸馏。

## 历史依据与这次预算

历史正式采集单卡1个仿真进程＋1个JAX策略进程，25环境并行，每任务5批×25=125。1机stack_bowls真实smoke25/25完成、19成功、约37分钟；后续叠碗/方块每25回合约38–48分钟，长任务1900步单批可超过2小时。带完整三相机视频和U，不据此承诺无记录版本的时间。

旧采集覆盖54个任务配置（含random变体），为绘图按任务选择历史较好模型seed0/1/2，不能把旧合集当单一固定模型的标准结果。这次建议固定同一官方seed0 checkpoint及归一化，所有C/N条件使用同一组布局与每回合随机噪声；不依各条件成败挑布局。

建议范围：H50固定，C10/20/30/40/50×N1/N10×54配置×8回合，计划4320回合。每任务一批8并行、串行1批；两卡各领取一个任务。若坚持单卡最大已验证25并行，则每条件至少25回合，共13500回合，与“每任务不超过10回合”不能同时满足。首轮8更符合筛选目标，单任务12.5个百分点分辨率，只用于总趋势，关键任务再另议增加样本。C50作为同协议基线纳入，避免只比较较短C。

## 复用的本地入口

- 1机安装：`/data/chenyiteng/projects/robodojo-openwam-sz1/RoboDojo`；策略是XPolicyLab Pi_05/OpenPI JAX，Dojo ARX X5官方权重，不能搬RoboTwin Sidney权重或其PyTorch评估入口。
- 已验收smoke：`deployment-20261018/dojo-u-video-smoke-sz1/smoke-n25.json`，模型`RoboDojo-sim-arx_x5-joint-0`、checkpoint59999。
- 本地旧wrapper：`worktrees/pi05-policy-budget-eval-20261017/tools/dojo_u_video/{run_task,lane,policy_entry,client_entry}.py`。
- run_task已具备GPU计算/图形同卡、独立策略端口、启动超时、600/1200秒无进展判断、精确token+UID+PID/start清场；lane逐任务领取、清场后跳过报错、接RLT。Repeated query identity自动跳过及重置时progress短暂消失处理已存在。
- 成功率版本要移除依赖视频/NPZ的validate条件，直接读取原生_result.json/details成功与失败；程序错误/不稳定布局单列缺测，正常失败必须计0。native main的无限/重复重启不再额外叠加，保留现有有界流程。卡未清场则停该lane，不能叠下一任务。

## 代码核对发现的提速点

1. **关闭纯采集开销**：不加载Observer及M5旁路，不计算U/DV/多噪声/attention，不保存query数组/观测NPZ，不写三路VideoStreamWriter、不编码、不做完整视频验证。仍保留模型所需三相机观察与原生成功判定。headless已开启，不能把它算作新的提速。
2. **取消块内无用途的图像传输**：已安装Pi_05/deploy.py在每个动作之后都get_obs_batch并update_obs_batch，即使还在执行旧chunk、没有再次infer。Model只覆盖observation_window，没有时序缓存；下一次infer前又取得最新观察。这些中间图像可尝试跳过，只在真正决策边界获取/传输；保留物理步、成功检查、设备/有限值检查。短测确认动作轨迹和下一次输入一致后再采用，不能直接降低物理频率或改变相机输入。
3. **真实模型合批**：model.py/get_action_batch目前for逐环境执行policy.infer，25仿真并行不等于25模型批推理。已安装OpenPI Policy.infer有is_batched分支，故可尝试8环境合批；需要确认文本/图像transform、显式每回合噪声、活跃槽位映射及固定shape/padding。不能因batch RNG变化破坏N1/N10对照，也不能仅删除for就宣称正确。若一两次尝试没有进展，回退逐环境推理。
4. **模型常驻**：同一固定checkpoint可在任务间保留策略/JAX编译，换任务重置观察、指令和每回合噪声；仿真仍按任务退出重建，保留故障隔离。原native wrapper每任务重新加载，改成受控本地策略服务是独立小切口，不先复用脆弱的Isaac场景生命周期。N1与N10各预热一次，报告冷启动和热运行时间。
5. **记录与输出**：保留每回合任务/布局/模型seed/噪声seed/C/N/成功/动作数/总耗时/加载重置推理环境耗时/异常原因；控制台逐env逐动作print可节流，保留低频心跳及错误。压缩JSON/大量数组均无需。
6. **并行选择**：每任务8总回合时就8环境；不照搬RoboTwin单卡3份Isaac Sim。双卡先各1任务，实测显存与环境吞吐后才考虑单卡2任务×8。主要时间可能仍在重置/渲染/控制，不能由显存余量推出线性加速。

## C与N的准确实现切口

H始终50；N通过Policy._sample_kwargs/模型构造的sample_kwargs传入num_steps=1或10，当前Policy.infer只接obs/noise，直接infer(num_steps=...)并不成立。C在Pi_05/deploy.py的动作块执行循环限制前缀，不把50个动作插值压缩为10，也不变更原生每任务动作上限、控制频率、物理/传感器分辨率。两个N对同一观察用同一噪声生成完整H50，再执行前C。不同C查询时刻不同，不宣称动作时刻噪声逐一对应。

## 拟验证与时间边界

授权实施后仅选stack_bowls和一个长任务做小规模前后对照；先无录制，再块内观察裁剪/批推理，分别记录收益。验证成功率路径、原生动作计数、C实际执行、N实际调用、GPU绑定与错误跳过。Dojo的最高并行25不是这次8回合的必要起点，不重走N1/N4/N16逐档爬升。现阶段没有无记录C10实测时长，待短测后按任务上限和加载占比估整队列，不能用RoboTwin一轮1.6h直接换算。

官方结构与相同适配器依据：[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)、[Pi_05 model.py](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/Pi_05/model.py)、[Pi_05 deploy.py](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/Pi_05/deploy.py)。动态main仅作对照，实施固定本机已成功版本，不能顺手升级整个环境。
