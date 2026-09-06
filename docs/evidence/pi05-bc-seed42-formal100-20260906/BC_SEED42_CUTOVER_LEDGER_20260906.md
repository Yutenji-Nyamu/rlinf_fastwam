# 两项π0.5 BC切换：seed42与DVAC [0,2]

## 授权与执行边界

2026-09-06用户明确授权：停止现有GPU6 BC与GPU7 BC＋DVAC [0.5,1.5]；主要日志/指标/配置/图表打小ZIP，并将轻量产物推对应代码分支；从同一原Sidney SFT、空成功池启动两项新实验。两者固定rollout seed42，DVAC范围改[0,2]，方法外参数继承现有干净BC。不是固定一个噪声张量，不更换GRPO，不动其他用户/shared Ray，不删除旧checkpoint/成功池。

不重复整套GPU smoke；实现后做与本增量直接相关的随机流/权重/配置回归，正式启动后核对实际进程、参数和健康启动。seed在rollout模型初始化后、首次采集前设置；评估作用域保存/恢复训练随机状态，避免评估消耗训练序列。原FM训练、模型、采样分布/M10/ODE、预算均不改。

## 逐操作记录

1. 15:59身份探针通过：chenyiteng@admin，固定host-key验证成功。已读当前短入口及π0.5 BC正式参数SSOT，历史快照仅定位，不当现场。
2. 开始本账本；下一步先刷新两driver/namespace/GPU/RAM/disk、最新metrics与checkpoint，记录可恢复保存点，再执行定向停止。
3. 16:01只读preflight：BC69、DVAC68完整轮，latest fixed65分别20/32、21/32，无所查fatal；两项Step60所需文件在。/data余约1363GiB，RAM available约1100GiB；GRPO driver701211保持。
4. 16:04定向SIGTERM两driver2143110/2223401，清理且仅清理其Ray namespace RLinf_1/RLinf_2（job f6010000/fc010000），GPU6/7释放，其他named actors集合/GRPO driver身份保持。退出码均0是程序处理SIGTERM后的返回，不代表跑完100轮。最终完整轮69/68，Step60文件集/ZIP中央目录可读，未做恢复测试；大权重/成功池未删除。
5. 本地合并结果包1421742B、SHA256 560064d83ff648f9629b24510cda0ba079e53ce1bacbb4ac46687fb9af0dd652，图已查看，无虚构Step0。两项原日志/TB/CSV/JSON/配置/图纳入，排除checkpoint tensors/replay/video；ZIP完整性通过。
6. 第一次归档提交前，git diff --check因原始日志尾空格等报错；尚未commit/push，BC归档目录已staged、DVAC未复制。不篡改原始日志消除空白，改为按manifest校验原文并只对新说明检查格式，再从这个明确断点继续。
7. 随机数实施批次：两树同一个rollout文件各加可选seed初始化、整轮eval保存/恢复RNG；复用utils已有seed_everything/get_rng_state/set_rng_state，默认None不改变其他方法。新测试直接调用worker实际方法，不加载模型/环境；DVAC复用已有default alpha0.25，不更改信号/归一化/损失算法。
8. 旧结果归档完成：BC commit76510d20fb57003ae3d76922874d29f0af533437；DVAC eef4ff674306a9c3a8c5d0ac14b330cb2f28cfc2；personal两对应分支ls-remote一致。各30文件（含完整原始日志）保留，原始字节按包比对；没有强推或修改日志空白。
9. 两树各24行生产增量，测试文件约80行；CPU BC18、DVAC18、额外DVAC13项全过，包括已有采集、mask、真实采样器结构、归一化/权重检查。正在compose完整正式配置与做GPU6小RNG组件测试；没有模型/仿真smoke。
10. 两分支CUDA随机流组件检查各8项通过。正式配置经validate_cfg、shell语法和实际CLI逐项核对：旧BC→新BC8叶仅seed/输出；旧DVAC→新DVAC9叶另含alpha；新两项之间17叶仅7方法＋10身份/GPU/路径，种子表内容相同。合同已展示；不新增Step0评估，不改采集或更新预算。
11. 源码/测试/完整启动配置已分别推送7e2565a05e421b8583c157730f96c56095cd2170与0390121c41b528434212c6c47a5e6d430fcae166；远端SHA相同、两树clean。
12. 16:28:07/16:28:19单次正式启动，BC wrapper914842/observer914843，DVAC wrapper915663/observer915664；先等BC预留RLinf_1再启动DVAC，避免namespace竞争。launch_attempt与两run回执已落盘；GRPO701211身份及原其他named actors保持。新run不得重复启动，正在只读验收实际配置/seed输出/首轮采集。
13. 16:30首次startup只读检查：实际配置均与prepared零差异，两边seed42确认行已在，actor仍初始化，无所查fatal；不把尚未进入采集标为通过。
14. 16:32第二次startup只读检查通过：driver914848/915687，namespace RLinf_1/RLinf_2，均首轮rollout；实际配置与各自prepared零差异、新两项仍仅17叶预期差异。GPU6/7各约26.41/25.87GiB（非峰值），RAMavailable1582GiB，/data1358GiB；GRPO保持。验收JSON和结果摘要已在本地，最后只追加轻量证据提交，不改运行源码。

## 收尾断点

- 旧停止/归档ZIP及push/seed实现与测试/正式启动/实际配置验收均已完成，禁止重复执行。
- 最后追加启动JSON/本账本/结果说明到对应分支；最终push SHA回执独立保存于BC_SEED42_STARTUP_PUSH_20260906.txt，避免文档递归记录自己的提交hash。开始后即回报，不等待完整一轮或长期盯守。
