# OpenDW 正式训练：2026-10-04 当前授权

用户最新要求：立即结束当前smoke，直接从同一SFT开始正式训练；不等长试完成、CP或非零学习信号验收。smoke→正式保持借卡，不先恢复RLT；WM运行异常才按原CP和任务接回RLT。本文替代本轮早先“长试验收后重新借卡”的计划。

| 项目 | 正式配置 |
|---|---|
| 环境N / 组G / 采样R | 64 / 8 / 8 |
| 动作长度L / 执行C / 预测H | 384 / 32 / 50 |
| 每轮轨迹 / chunk槽位上限 | 512 / 6144 |
| global / micro / 更新遍数 | 2048 / 8 / 2 |
| actor与rollout / 两WM服务 | 4、5卡 / 6、7卡 |
| WM batch | 两服务各B1排队 |
| runner.max_epochs / max_steps | 1000 / 200；实际最多200轮 |
| 保存 / 原生评估间隔 | 10 / 10 |
| 起点 | 同一SFT，resume_dir=None |

三相机、M10、noise0.5、LR5e-6、GRPO过滤、连续差分奖励、成功阈值、OpenDW/RM/reset资产和装卸屏障保持当前长smoke实配。6144是含终止后mask的槽位上限，不等于全部进入有效loss。零梯度保留为诊断，当前不作为启动或继续的失败条件，也不宣称已有效学习。

正式配置单独恢复真正原生RoboTwin评估：训练`opendw_robotwin`，评估`robotwin`。从实际clean Aloha三相机Control复制eval，使用adjust_bottle已有固定种子，N32/R1/G1、H50/C32、动作上限384，并设置局部`env.eval.enable_offload=true`。正式driver自身的原生环境初始化是本次首次真实检验；**不额外运行native GPU probe，不预填原生验收通过。** 每10轮使用RoboTwin原生check_success，不把WM视觉奖励当真实成功率。现有runner先评估再保存，评估失败可能使对应轮检查点未保存；本次不改runner顺序。

旧smoke owner源码保持冻结。交接守护持有旧周期operation.lock，按PID/start/UID精确结束旧owner，等待其WM进程、namespace和图形上下文清理；旧owner的RLT返回尝试被这把专属交接锁挡住并保留真实回执。新周期通过独立adoption回执接管原已停止的RLT及完整CP责任，不重停、不先返回再借。

新`formal/opendw_formal_owner.py`导入锁定SHA的四卡owner，当前显式`start_mode=direct_start_user_override_20261004`。其训练协议只与当前full配置及SHA比较；不要求learning report或CP。adoption必须由新生命周期`verify_adoption(stage, receipt)`核验，不能凭停止标记存在就认定接管。薄wrapper对原owner_main仅精确替换两处：初始借卡条件改为显式adoption验证，borrowed初值改为已验证接管；落`formal-owner-source-delta.json`。服务CPU加载阶段即承担返还责任，后续异常沿原完整清理与RLT恢复路径。

`post_borrow_hook.py`在借卡/接管已证明、4–7卡空后激活新私有图形scope；仅写`scope-activated.json`，明确`native_probe_verified=false`。scope片段显式透传正式driver与Ray worker。若激活后训练异常，新返还周期读取activation回执使用新scope恢复RLT；不得回用会拒绝新增profile的旧scope。后台训练延续与失败恢复由服务器owner承担，不依赖聊天窗口存活。

本轮准备文件：

- [配置生成器](../../../local_patches/opendw_smoke_20261003/formal/build_formal_config.py)
- [正式owner](../../../local_patches/opendw_smoke_20261003/formal/opendw_formal_owner.py)
- [仅激活scope的hook](../../../local_patches/opendw_smoke_20261003/formal/post_borrow_hook.py)
- [服务器CPU fixtures](../../../local_patches/opendw_smoke_20261003/formal/test_formal_plan.py)
- [原学习与资源审计](../../../local_patches/opendw_smoke_20261003/multigpu/analyze_multigpu_smoke.py)：仍可诊断；严格learning_gate保留为独立旧模式，本轮不调用。

`native_env_probe.py`是此前严格方案的独立工具，本轮不部署、不运行。正式实配、源文件SHA、接管与scope回执以及轻量日志由主线程核验后推Git；完整checkpoint及原始视频仍在服务器，不声称已异地备份。本文是代码与授权上下文，不代替新的服务器运行状态回执。
