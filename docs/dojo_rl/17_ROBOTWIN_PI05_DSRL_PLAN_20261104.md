# RoboTwin π0.5 DSRL · 仅规划

**后续用户已纠正：训练仍200动作，直接复用已有π0.5 DSRL＋U实配。本文件400动作/旧π0推导建议已被[18 历史π0.5复用方案](18_REUSE_PI05_DSRL_U_200_20261104.md)替代，勿据此启动。**

2026-11-04用户意图：放弃6卡Dojo重复噪声方向，拟换RoboTwin π0.5 DSRL；选官方400动作、原始成功率接近50%的已测任务。**本轮先讨论，不撤6卡旧任务，不启动新实验。** 5卡完整噪声Dojo和4/7 GRPO继续；新材料准备好并获启动决定后才交接，失败接原6卡RLT。

## 任务与模型

推荐`turn_switch`（拨开关），原始Sidney多任务π0.5作单任务RL底座。不是RLinf调瓶子单任务模型，也不是Dojo模型。

|候选|历史N10/H50/C50|另有C20/N10记录|官方上限|
|---|---|---|---|
|turn_switch|15/32=46.875%|5/8=62.5%|400|
|stamp_seal|14/32=43.75%|2/8=25%|400|
|rotate_qrcode|18/32=56.25%|2/8=25%|400|

首列来自10/19四噪声采集，执行仍原始N10；第二列来自10/24普通C扫描首批，两者种子/协议不能当成纯C因果对照。C20样本8局仅作先验，不能据此确认基础成功率。调瓶子经验最多，但不同π0.5底座的成绩差异大，不能混用单任务85.94%和Sidney成绩。

证据：`docs/pi05_distillation/evidence/task_screen_history.json`；服务器`/data/chenyiteng/results/pi05-distill/chunk-sweep-sz1-g67-20261024/summary-raw-c20-n10-b0.json`。官方上限：https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/env_cfg/task_config/_eval_step_limit.yml 。

## 继承与变化：不要把“其余沿用”理解成只改权重路径

旧π0主锚点：工作区`worktrees/dsrl-pi0-single-u-n-20261009/docs/dsrl_pi0_clean_u2_formal_20261009/configs/clean-g6.yaml`，已运行单卡Clean配置，历史提交ae8a8adb497e1ce47608e6c111d7365c679d4142。旧π0.5审计：根`docs/methods/ugrow/DSRL_PI05_AUDIT_20261006.md`。

|项目|旧π0实配|本轮建议及依据|
|---|---|---|
|任务/上限|调瓶子/200|拨开关/400，用户目标及原生上限；同步train/eval/rollout三处|
|模型|RLinf单任务π0，delta关节|Sidney π0.5，absolute关节、专属mean/std、三相机与指令；复用已验收Sidney加载路径|
|生成|N4/H50/C20|N10/H50/C20；N10按模型，C20保留旧DSRL决策频率。先在拟用训练入口评原模型32局|
|学习对象|冻结全部VLA，训练小噪声策略和Q|保持；无U、额外trick、专家动作监督|
|噪声|32维沿H50重复|建议完整50×32，复用本日已验收full_noise开关；actor/Q/replay全部1600维，不能只删除repeat|
|熵尺度|目标−16、初始α1|完整噪声−800、α.02，沿已跑通Dojo尺度适配；这是工程候选，不是论文证明最优|
|训练采样|4环境×1批/轮|先保持4局/轮，200轮=800训练局；不擅自换8×200翻倍预算|
|SAC|B256/micro256；UTD20；actor1e−4、Q3e−4|保持；一次更新取256条旧转移，不是每轮新采256局|
|Q/目标|10Q均值，γ.999、目标更新τ.005，backup entropy关|保持；沿旧成功算法，不引入FQL v2的改动|
|回放|25000条，预热500条|保持，单位是动作块转移；记录实际新转移和更新数|
|评估|4×3=12局，每13轮|建议初始及每13轮32局，末轮128；比旧12更易辨认趋势，是显式评估增量|
|存储与交接|完整恢复及逐卡RLT|Data最新完整续训点滚动，峰值适配器单独保护；无录像/U/大chain，保留小日志曲线；6卡原RLT候补与不存ckpt兜底复用|

400替代200且C20不变时，满长每轮4×20=80条转移，UTD20意味着最多1600次小网络更新，约旧800的两倍；提前成功会降低实际量。因此不能说时长与旧π0相同，也不能用GRPO128局/轮套DSRL。正式ETA在完整采样→更新→评估周期后估计。

## 增量切口与有限验收

1. 以已验收RoboTwin DSRL源码和本日完整噪声补丁组合；不用Dojo原生Isaac/RPC适配。Sidney加载、absolute动作、mean/std、state文本离散化沿现有实现。
2. 旧构造器硬编码`adjust the bottle`，切任务时必须同步真实动态instruction/default prompt和task seed表；复用10/07 reset先重建场景后刷新指令的修复。不把其他任务种子表冒作拨开关，不按模型成功筛种子。
3. 数据投影沿已验收SAC：每块成功0、否则−1；成功终止不bootstrap，时间截断保留bootstrap；实际执行长度用于discount，剔除结束后块。400动作/C20末块可整除，但成功提前结束仍须保留正确mask。
4. 全维补丁位置：`openpi_action_model.py`、`gaussian_policy.py`、`dsrl_transition.py`、`fsdp_sac_policy_worker.py`；已验收工具`tools/dojo_offpolicy/full_noise.py`。它不依赖Isaac，迁移仅复用方法层。
5. 未来启动前：原始独立噪声在同一RL入口32局，确认动作/指令/起点；一次短更新→保存→恢复；同时测global256/micro256的真实整步峰值与耗时。除非证据表明容量或速度有问题，不扩大测试矩阵。微批已经等于全局batch，继续加大batch会改变算法。
6. 原始Gaussian起点正常仍不保证学习阶段正常：旧噪声策略学习后经tanh限制范围；需要比较切换到learned阶段前后成功率、Q、α、噪声饱和比例。避免再次把SAC探索变化误称模型加载错误。

## 来源与边界

- DSRL论文附录C.5：完整噪声空间较大，重复32维是降维选择，不是π0.5必须遵守的模型契约。https://arxiv.org/html/2506.15799v2#A3.SS5
- 官方讨论有π0.5关节控制“独立有效、重复失败”，也有重复有效反馈，不能普遍断言。https://github.com/nakamotoo/dsrl_pi0/issues/9
- 当前Dojo完整初评20/32、重复0/32支持试完整噪声，但环境/模型不同，不能把该成功率外推RoboTwin。
- RLinf方法概述：https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html 。具体上线实现仍以我们的固定旧源码和实配为主，不整份升级最新框架。

## 顺便核实GRPO

现场`deployment-20261104/grpo-clean-entropy-g47/config/grpo_clean.yaml`：total_num_envs64、rollout_epoch2、G8，**每轮128局**，每卡32环境，合16组；global512、micro32、update_epoch2。更早256局版本是另一套配置，当前未改。
