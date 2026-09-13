# RLT DVAC反转：实现与启动验收

2026-09-13，18:04验收。**代码已推送，93项CPU测试和GPU smoke均通过；GPU7正式实验已进入初始teacher采集。**

| 项目 | 已确认状态 |
|---|---|
| 分支 / 提交 | `codex/sz-rlt-dvac-direction-20260913` / [`2be9a6d`](https://github.com/Yutenji-Nyamu/rlinf_fastwam/commit/2be9a6d542b0688f3092ea67a4f7b4c07944107e)，远端HEAD已核一致 |
| 直接基线 | 原RLT DVAC new scale1，`5e5882e6` |
| 正式预算 | 600轮、8env、B512/micro256、UTD5；原Stage1起点，全新Stage2，scale1 |
| 方法配置 | `scope=both`、`transition_updates=0`、两层α=1；成功均重1、失败权重1 |
| 切点 | 沿用原BC/Q的critic计数；课程在69999到末点，首个实际actor反转槽70000 |
| CPU验证 | 深圳93项通过，6.43秒；包含真实worker边界、恢复契约与完整batch权重测试 |
| GPU资源 | 已接替GPU7原new scale1；GPU6 scale2及其他保护进程身份保持 |
| 正式启动 | 18:01:36，18:04确认12个actor存活、环境交互和rollout生成中，fatal为空 |

新recipe：`robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_dvac_direction`。只增加方向课程及实验标识，teacher、Q目标、环境、回放格式和原BC/Q系数课程沿用对照。旧配置不注入新默认字段；新恢复契约记录课程参数与锚点。

服务器源码：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-direction-20260913`。

GPU learner smoke已通过：固定512条真实回放，其中253条成功来源；生产FSDP、损失和优化器，仅为smoke把计数设为69998，执行4次critic、2次actor。两层方向在69998均为+1、70000均为−1，成功均重两侧均约1；actor/Q参数L2变化分别0.1516/0.1940，微批复用和正权检查通过。**这是跨切点工程验收；正式训练计数从0开始。**

旧scale1于18:01:26停止，smoke最小回执于18:01:36通过，18:01:36.9立即启动正式包装进程，前后约11秒。输出目录：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-direction-phys7-fresh600-20260913-v1`；namespace：`RLinf_rlt_dvac_direction_single_gpu7_fresh600_20260913`。

18:04启动核对：实际配置与准备配置无非预期差异；三项运行时默认展开为`false`，未改变预算或方法。当前为初始teacher采集阶段，尚未到正式actor学习或方向反转点。

证据：[93项CPU测试](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/method-cutover-20260913/rlt/cpu-tests.log)、[GPU smoke](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/method-cutover-20260913/rlt/smoke-result.json)、[旧实验停止](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/method-cutover-20260913/rlt/stop-receipt.json)、[切换回执](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/method-cutover-20260913/rlt/cutover-result.json)、[18:04正式采集验收](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/method-cutover-20260913/rlt/STARTUP_VERIFIED.json)、[源码推送](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/method-cutover-20260913/rlt/source-publish-receipt.json)。

方法定义及兼容细节见[独立规划](RLT_DVAC_DIRECTION_SCHEDULE_PLAN_20260913.md)，此页仅维护实施与运行验收。
