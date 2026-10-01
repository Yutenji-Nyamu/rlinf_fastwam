# EXPO-FT SZ2 实施记录

状态：实现、真实fresh＋新进程resume smoke及原四RLT恢复首轮均通过。源码和轻量证据归属独立branch `codex/sz2-expo-ft-20261001`；最终Git远端身份由发布回执核验。2026-10-01。

## 已完成

1. 核对官方主仓库/OpenPI源码身份，固定论文及代码差异；继承SZ2已跑通Control动作、原生环境、Sidney权重/norm和成功演示。依据集中在`IMPLEMENTATION_BASIS_20261001.md`。
2. 验证SZ2固定host-key、chenyiteng UID20001、hostname h100-gpu02；建立独立source/branch/output。用户最新卡位为物理4–7，原RLT低优先级，须CP续回。
3. 实现候选编辑选优、10Q ensemble、variable-K TD、UTD20/FM/editor/temperature更新、真实图像replay和完整学习状态恢复；native后端冻结VLM prefix、训练action expert/projections。
4. 服务器CPU 7项核心检查通过，耗时5.418秒；检查候选独立性、Q采样/梯度、TD与terminal边界及state恢复。
5. 修正raw next-observation callback保留env/prompt、checkpoint 0D tensor digest展平；加入起始模型权重严格SHA manifest和源码fingerprint。
6. 原生下层controller读取证据：gripper限幅及drive velocity规则保留。replay记录提交环境的canonical command，不等同于测得关节轨迹。
7. 当前SAPIEN3.0.1的`SapienRenderer(**args)`忽略device参数；通过独立进程/子进程的`Scene`默认系统显式`RenderSystem('cuda:0')`绑定唯一可见GPU UUID，保留CPU PhysX与原渲染设定，未改共享SDK或RoboTwin文件。
8. RLT借卡CPU预审发现最近checkpoint索引10.3–11.2万、实际payload固定8万。当前源码`buffer.py:927–996`在`auto_save=False`只保存cache，却保存全索引；`sample_chunks:567–599`确为最近8万窗口。仍保留全量索引/计数，不以窗口等价为由放宽完整恢复校验。独立恢复副本保留最新模型/优化器/RNG，用同GPU既有完整CP中exact entry匹配的原payload补齐；覆盖率全量、CPU样本及原严格检查成功后才借卡。原CP不修改。
9. 四份恢复副本首次CPU全验通过：step1475/1525/1450/1475，补齐27243/30348/32365/33775条旧payload；索引全量107243/110348/112365/113775。之后stop前/后还复核最新保存轮次，不把初始预审轮次冒作最终归还轮次。
10. 首次guardian在借卡前退出：两套uv Python缺`os.pidfd_open`/`signal.pidfd_send_signal`绑定；没有停任何RLT。复用已审Linux x86_64 LP64 pidfd syscall兼容层，并在两个实际解释器各创建一个CPU子进程，pidfd SIGTERM真实退出码-15，均通过。归档初次失败日志/launch回执后重新派发唯一guardian；不回退裸PID或广义进程名清理。
11. 演示FM预审修正：原PNG是480×640，不能提前压成224方图；改由原生π0.5 transform处理。演示真实指令从parquet `task_index`对应`meta/tasks.jsonl`读取，校验完整H50窗一致；不以笼统任务名替换原始自然语言指令。
12. 13:53 guardian精确停四RLT成功，driver/ns/GPU全部释放；停止前后最终CP1500/1525/1475/1475，额外补齐最新g4/g6原文件。13:54派发`smoke-v1`，物理4，fresh1episode＋新进程resume1episode，唯一进程/来源manifest及完整输入SHA已落服务器。
13. `smoke-v1`真实环境初始化/reset成功，首个π0.5候选生成在SigLIP的cuDNN SDPA报`mha_graph.execute(...).is_good()`失败，尚未执行动作或学习更新。已释放本次GPU及全部后代进程，guardian保持可重试的PAUSED。工作区此前成功的`tools/pi05_dv50/run_inference.py`对同H100/runtime问题已用`enable_cudnn_sdp(False)`；本次沿用该独立进程兼容处理，保留flash/efficient/math后端，不改变动作、模型或预算。依据：[PyTorch后端API](https://docs.pytorch.org/docs/stable/backends.html)、[同类上游错误记录](https://github.com/pytorch/pytorch/issues/190321)。同报错提供线索，不能据此认定两者硬件/根因完全相同。
14. `smoke-v2` fresh真实episode已通过：200动作/20chunks，独立16候选；Q20、FM1、editor/temperature1，critic/editor/target/temperature均有参数变化。FM loss0.026508、允许参数抽样Δmax2.5004e-5，冻结参数无梯度且每tensor64元素样本保持；online success=0，FM来源是clean50真实成功demo，不伪称在线成功。fresh运行153.50秒、PyTorch峰值allocated14.98GiB；checkpoint保存后交给新进程resume继续验收。
15. 新进程resume通过：base/core/replay/candidate RNG四块完整tensor状态SHA严格一致；再执行200动作/20chunks、第二次真实学习更新及新checkpoint。累计Q40、FM2、editor2、temperature2；resume FM loss0.025655、Δmax2.5008e-5，online success仍0，使用第二份已核验成功demo。resume运行157.60秒、PyTorch峰值allocated14.98GiB。恢复为episode边界继续学习，不宣称模拟器逐步bitwise continuation。
16. `smoke-v2/final.json`：smoke_passed=true、全部本次后代精确清理、reserved_gpu_released=true、RLT借卡边界校验通过。外层唯一guardian已恢复派发原四RLT并进入WAITING_FIRST_ROUND，无错误；必须等四条实际首轮完成后才把资源归还标为最终通过。
17. 14:31四RLT首轮验收全部通过：物理4/5/6/7分别从CP1500/1525/1475/1475续到1501/1526/1476/1476，ready_for_online=1、critic更新400/325/400/400，replay全局计数109159/110413/114382/113855。guardian=RESTORED、无错误，原累计3000终点及源码不变，监控路由已按冻结helper切换。日志与TB暂时不一致是写入/flush时序；每文件/目录重读同值，未改验收标准或放宽条件。
18. 完整resume checkpoint的CPU元数据复核：可训练base含209个tensor、693422112个FP32 master参数。每次FM更新检测208个参数样本变化；不把样本计数冒作全元素变化比例。Git轻量证据汇总为`evidence/SZ2_SMOKE_20261001.json`（36044 bytes）与精确输入/7项测试记录，大checkpoint和完整replay留服务器。

## 轻量证据路由

- 本机证据目录：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/expo-ft-implementation`。
- `prepare-source-and-contracts.out`：原生接口/config快照；`inputs-and-rlt-anchors.out`：实配/成功演示及旧GPU进程锚点。
- `core-tests-v1.err`：7项unittest通过；`native-render-gripper-probe.out`：当前原生renderer/controller代码。
- `native-import-check.out`：实际服务器原生导入通过；`renderer-engine-device-api-v2.out`：已安装SDK精确接口。
- `rlt-current-freeze-prepare.err`、`rlt-checkpoint-completeness-probe.out`、`live-rlt-buffer-source.out`：RLT checkpoint缺失的原始证据；修复清单/验证在服务器`rlt-cycle/repair-plan-*`和`/data/chenyiteng/recovered-rlt/rlt-cycle/*/global_step_*/repair-manifest.json`。
- `rlt-current-freeze-prepare-repaired.out`：四份CPU恢复合同；`guardian-first-startup-log.out`和两份`*-python-pidfd-signal-check.out`：兼容问题及实际修复检查。
- 初始preflight遇到schema `repo` KeyError，修正读取结构后v2成功；SSH PTY单行多upload超过终端长度，改为逐文件短JSON请求，均保留失败记录。

## 正式实验尚未验收

真实环境/真实更新/新进程恢复/GPU清理/RLT续回首轮已有完整回执；正式B64、八环境规模、长训效果及收集-更新cadence尚未验证，须单独锁定配方。发布仅含24个源码/文档/小证据文件，起始模型、checkpoint与完整原始replay留服务器。
